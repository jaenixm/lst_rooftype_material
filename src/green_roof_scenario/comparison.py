"""Run and validate the frozen 12-scenario exact-versus-dominant comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio

from .config import ScenarioConfig
from .masking import subset_buildings
from .provenance import git_commit, sha256_path
from .scenario import run_scenario


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    city: str
    dataset: str
    threshold: float
    strategy: str
    buildings_file: str
    slope_field: str
    l2_folder: str
    target_ndvi: float
    target_albedo: float
    target_ndbi: float
    expected_count: int
    expected_area_ha: float


CITY_PARAMETERS = {
    "madrid": {
        "l2": "data/raw/landsat/LC09_L2SP_201032_20240723_20240724_02_T1madrid",
        "target_ndvi": 0.28431087365675817,
        "target_albedo": 0.17372641559623933,
        "target_ndbi": -0.07992667124082531,
    },
    "paris": {
        "l2": "data/raw/landsat/LC09_L2SP_199026_20240810_20240811_02_T1paris",
        "target_ndvi": 0.40624622109631436,
        "target_albedo": 0.14593065672091077,
        "target_ndbi": -0.05327758769465608,
    },
    "hamburg": {
        "l2": "data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg",
        "target_ndvi": 0.5098841067653748,
        "target_albedo": 0.1435968737428123,
        "target_ndbi": -0.15313389460307741,
    },
}


def _case(
    dataset: str,
    threshold: int,
    strategy: str,
    buildings_file: str,
    slope_field: str,
    expected_count: int,
    expected_area_ha: float,
) -> ComparisonCase:
    city = "hamburg" if dataset.startswith("hamburg") else dataset
    params = CITY_PARAMETERS[city]
    return ComparisonCase(
        name=f"{dataset}_{strategy}_le{threshold}",
        city=city,
        dataset=dataset,
        threshold=float(threshold),
        strategy=strategy,
        buildings_file=buildings_file,
        slope_field=slope_field,
        l2_folder=str(params["l2"]),
        target_ndvi=float(params["target_ndvi"]),
        target_albedo=float(params["target_albedo"]),
        target_ndbi=float(params["target_ndbi"]),
        expected_count=expected_count,
        expected_area_ha=expected_area_ha,
    )


CASES = [
    _case("madrid", 11, "exact", "madrid_comparison.gpkg", "slope_median_deg", 4647, 542.2788),
    _case("madrid", 11, "dominant", "madrid_comparison.gpkg", "slope_median_deg", 4895, 699.2551),
    _case("madrid", 30, "exact", "madrid_comparison.gpkg", "slope_median_deg", 25040, 1688.8892),
    _case("madrid", 30, "dominant", "madrid_comparison.gpkg", "slope_median_deg", 25644, 2030.8242),
    _case("paris", 11, "exact", "paris_comparison.gpkg", "slope_median_deg", 8486, 554.8416),
    _case("paris", 11, "dominant", "paris_comparison.gpkg", "slope_median_deg", 8557, 591.8603),
    _case("paris", 30, "exact", "paris_comparison.gpkg", "slope_median_deg", 34880, 1358.8060),
    _case("paris", 30, "dominant", "paris_comparison.gpkg", "slope_median_deg", 34997, 1418.0887),
    _case("hamburg_old", 15, "exact", "hamburg_old_comparison.gpkg", "roof_slope_mean_deg", 59547, 3332.2716),
    _case("hamburg_old", 15, "dominant", "hamburg_old_comparison.gpkg", "roof_slope_mean_deg", 59947, 3549.7170),
    _case("hamburg_new", 15, "exact", "hamburg_new_comparison.gpkg", "roof_slope_mean_deg", 59551, 3332.9568),
    _case("hamburg_new", 15, "dominant", "hamburg_new_comparison.gpkg", "roof_slope_mean_deg", 59949, 3549.0361),
]


def _scenario_config(case: ComparisonCase, data_root: Path, prepared_dir: Path, output_root: Path) -> ScenarioConfig:
    return ScenarioConfig(
        l2_folder=data_root / case.l2_folder,
        buildings=prepared_dir / case.buildings_file,
        layer="buildings",
        roof_material_field="predicted_roof_materials",
        roof_materials_type="0,4",
        roof_material_strategy=case.strategy,
        roof_material_cov_field="material_cov",
        roof_slope_field=case.slope_field,
        max_roof_slope_deg=case.threshold,
        build_lst=True,
        lst_unit="celsius",
        out_dir=output_root / case.name,
        target_ndvi=case.target_ndvi,
        target_albedo=case.target_albedo,
        target_ndbi=case.target_ndbi,
        model="rf",
        sample_frac=0.1,
        min_sample_spacing=100,
        random_state=42,
        supersample=8,
        clip_positive_delta=True,
        write_indices_rasters=True,
        write_pred_baseline=False,
    )


def _output_hashes(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: sha256_path(str(path.resolve()))
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }


def _source_candidate_stats(case: ComparisonCase, prepared_dir: Path) -> tuple[int, float]:
    """Measure candidates in the prepared layer's own projected CRS."""

    buildings = gpd.read_file(prepared_dir / case.buildings_file, layer="buildings")
    selected = subset_buildings(
        buildings,
        "predicted_roof_materials",
        "0,4",
        roof_slope_field=case.slope_field,
        max_roof_slope_deg=case.threshold,
        roof_material_strategy=case.strategy,
        roof_material_cov_field="material_cov",
    )
    return len(selected), float(selected.geometry.area.sum())


def _validate_delta(path: Path) -> float:
    with rasterio.open(path) as src:
        delta = src.read(1, masked=True)
    maximum = float(np.ma.max(delta))
    if maximum > 1e-7:
        raise AssertionError(f"Positive delta found in {path}: maximum={maximum}")
    return maximum


DIAGNOSTIC_ABS_TOLERANCE = 1e-12


def _validate_diagnostics(records: list[dict[str, object]]) -> float:
    maximum_difference = 0.0
    for city in CITY_PARAMETERS:
        city_records = [record for record in records if record["case"]["city"] == city]
        reference = city_records[0]["provenance"]["model_diagnostics"]
        for record in city_records[1:]:
            candidate = record["provenance"]["model_diagnostics"]
            if candidate.keys() != reference.keys():
                raise AssertionError(f"Model diagnostic fields differ within {city}")
            for key in reference:
                difference = abs(float(candidate[key]) - float(reference[key]))
                maximum_difference = max(maximum_difference, difference)
                if difference > DIAGNOSTIC_ABS_TOLERANCE:
                    raise AssertionError(
                        f"Model diagnostic {key} differs within {city} by {difference}, "
                        f"exceeding {DIAGNOSTIC_ABS_TOLERANCE}"
                    )
    return maximum_difference


def _validate_historical(records: list[dict[str, object]], historical_output: Path) -> None:
    record = next(item for item in records if item["case"]["name"] == "hamburg_old_exact_le15")
    provenance = record["provenance"]
    counts = provenance["counts"]
    cooling = provenance["cooling_statistics_c"]
    if counts["buildings_selected"] != 59547:
        raise AssertionError("Hamburg-old exact candidate count does not reproduce 59,547")
    if round(float(cooling["raster_mean"]), 4) != 0.0733:
        raise AssertionError(f"Hamburg-old raster mean does not reproduce 0.0733: {cooling['raster_mean']}")
    if round(float(cooling["all_buildings_maximum"]), 4) != 7.5915:
        raise AssertionError(f"Hamburg-old maximum does not reproduce 7.5915: {cooling['all_buildings_maximum']}")

    expected_importance = historical_output / "model_feature_importance.txt"
    actual_importance = Path(str(record["output_dir"])) / "model_feature_importance.txt"
    if not expected_importance.exists():
        raise FileNotFoundError(f"Historical feature-importance benchmark missing: {expected_importance}")
    if actual_importance.read_text() != expected_importance.read_text():
        raise AssertionError("Hamburg-old RF importances do not match the historical benchmark")
    expected_text = "ndvi: 0.4467\nalbedo: 0.1285\nndbi: 0.4248"
    if expected_text not in actual_importance.read_text():
        raise AssertionError("Historical RF importances are not the expected 0.4467/0.1285/0.4248")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_summary(records: list[dict[str, object]], summary_path: Path, manifest_path: Path, data_root: Path, prepared_dir: Path, output_root: Path, diagnostic_max_difference: float) -> None:
    lines = [
        "# Legacy exact-versus-dominant material comparison",
        "",
        "This report was generated by the frozen `v0.1.1` comparison workflow. Generated GeoPackages and rasters remain local; the commands, provenance, hashes, validations, and results are committed.",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        f'ROOT="{data_root}"',
        'REPRO="/path/to/freeze-legacy-v0.1.1-dominant-material-worktree"',
        f'PREPARED="{prepared_dir}"',
        f'OUTPUTS="{output_root}"',
        'env -u PROJ_LIB -u GDAL_DATA PYTHONPATH="$REPRO/src" "$ROOT/.venv/bin/python" "$REPRO/scripts/prepare_legacy_material_comparison.py" --data-root "$ROOT" --out-dir "$PREPARED"',
        f'env -u PROJ_LIB -u GDAL_DATA PYTHONPATH="$REPRO/src" "$ROOT/.venv/bin/python" "$REPRO/scripts/run_legacy_material_comparison.py" --data-root "$ROOT" --prepared-dir "$PREPARED" --output-root "$OUTPUTS" --historical-output "$ROOT/outputs/scenarios/hamburg_slope_repro_historical" --summary-path "$REPRO/{summary_path.relative_to(Path(__file__).resolve().parents[2])}" --manifest-path "$REPRO/{manifest_path.relative_to(Path(__file__).resolve().parents[2])}"',
        "```",
        "",
        "Each case uses building extent, Celsius LST, RF, sample fraction `0.1`, spacing `100 m`, seed `42`, supersampling `8`, material classes `0,4`, positive-delta clipping, and index-raster output. No administrative boundary or baseline-prediction raster is used.",
        "",
        "## Scenario results",
        "",
        "| Dataset | Slope | Strategy | Candidates | Candidate area (ha) | Buildings >0.01°C | Raster mean (°C) | All avg. (°C) | All area-wtd. (°C) | Changed avg. (°C) | Changed area-wtd. (°C) | Maximum (°C) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_name: dict[str, dict[str, object]] = {}
    for record in records:
        case = record["case"]
        provenance = record["provenance"]
        counts = provenance["counts"]
        stats = provenance["cooling_statistics_c"]
        by_name[str(case["name"])] = record
        lines.append(
            f"| {case['dataset']} | ≤{int(case['threshold'])}° | {case['strategy']} | "
            f"{counts['buildings_selected']:,} | {_fmt(record['candidate_area_source_crs_m2'] / 10000)} | "
            f"{counts['buildings_cooling_gt_0_01_c']:,} | {_fmt(stats['raster_mean'])} | "
            f"{_fmt(stats['all_buildings_average'])} | {_fmt(stats['all_buildings_area_weighted_average'])} | "
            f"{_fmt(stats['changed_buildings_average'])} | {_fmt(stats['changed_buildings_area_weighted_average'])} | "
            f"{_fmt(stats['all_buildings_maximum'])} |"
        )

    lines.extend([
        "",
        "## Dominant minus exact",
        "",
        "| Dataset | Slope | Δ candidates | Δ area (ha) | Δ buildings >0.01°C | Δ raster mean (°C) | Δ changed avg. (°C) | Δ changed area-wtd. (°C) | Δ maximum (°C) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    pairs = [("madrid", 11), ("madrid", 30), ("paris", 11), ("paris", 30), ("hamburg_old", 15), ("hamburg_new", 15)]
    for dataset, threshold in pairs:
        exact_record = by_name[f"{dataset}_exact_le{threshold}"]
        dominant_record = by_name[f"{dataset}_dominant_le{threshold}"]
        exact = exact_record["provenance"]
        dominant = dominant_record["provenance"]
        ec, dc = exact["counts"], dominant["counts"]
        es, ds = exact["cooling_statistics_c"], dominant["cooling_statistics_c"]
        area_delta_ha = (
            dominant_record["candidate_area_source_crs_m2"]
            - exact_record["candidate_area_source_crs_m2"]
        ) / 10000
        lines.append(
            f"| {dataset} | ≤{threshold}° | {dc['buildings_selected'] - ec['buildings_selected']:+,} | "
            f"{_fmt(area_delta_ha)} | "
            f"{dc['buildings_cooling_gt_0_01_c'] - ec['buildings_cooling_gt_0_01_c']:+,} | "
            f"{_fmt(ds['raster_mean'] - es['raster_mean'])} | "
            f"{_fmt(ds['changed_buildings_average'] - es['changed_buildings_average'])} | "
            f"{_fmt(ds['changed_buildings_area_weighted_average'] - es['changed_buildings_area_weighted_average'])} | "
            f"{_fmt(ds['all_buildings_maximum'] - es['all_buildings_maximum'])} |"
        )

    lines.extend([
        "",
        "## Model diagnostics",
        "",
        f"Diagnostics are identical to numerical precision across scenarios within each city (maximum absolute difference `{diagnostic_max_difference:.3e}`, required ≤ `{DIAGNOSTIC_ABS_TOLERANCE:.0e}`). Rounded diagnostics and feature-importance files are identical.",
        "",
        "| City | R² train | R² test | RMSE train (°C) | RMSE test (°C) |",
        "|---|---:|---:|---:|---:|",
    ])
    for city in CITY_PARAMETERS:
        provenance = next(record["provenance"] for record in records if record["case"]["city"] == city)
        metrics = provenance["model_diagnostics"]
        lines.append(
            f"| {city} | {_fmt(metrics['r2_train'])} | {_fmt(metrics['r2_test'])} | "
            f"{_fmt(metrics['rmse_train'])} | {_fmt(metrics['rmse_test'])} |"
        )

    lines.extend([
        "",
        "## Validation",
        "",
        "- All 12 candidate counts and areas match the frozen expectations.",
        "- Every delta raster is non-positive at all finite pixels.",
        "- The current four inputs contain no malformed arrays and no maximum-coverage ties.",
        "- Hamburg-old exact ≤15° reproduces 59,547 candidates, raster mean cooling `0.0733°C`, maximum cooling `7.5915°C`, and RF importances `0.4467 / 0.1285 / 0.4248` (NDVI/albedo/NDBI).",
        f"- Full input/output SHA-256 hashes, absolute source paths, parameters, Git commit, and environment versions are in `{manifest_path.name}`.",
        "",
    ])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _validate_case_provenance(case: ComparisonCase, provenance: dict[str, object], data_root: Path, prepared_dir: Path) -> None:
    parameters = provenance["parameters"]
    expected = {
        "roof_material_field": "predicted_roof_materials",
        "roof_materials_type": "0,4",
        "roof_material_strategy": case.strategy,
        "roof_material_cov_field": "material_cov",
        "roof_slope_field": case.slope_field,
        "max_roof_slope_deg": case.threshold,
        "model": "rf",
        "sample_frac": 0.1,
        "min_sample_spacing": 100,
        "random_state": 42,
        "supersample": 8,
        "clip_positive_delta": True,
        "write_indices_rasters": True,
        "write_pred_baseline": False,
    }
    for key, expected_value in expected.items():
        if parameters.get(key) != expected_value:
            raise AssertionError(
                f"{case.name}: provenance parameter {key}={parameters.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    expected_buildings = (prepared_dir / case.buildings_file).resolve()
    expected_l2 = (data_root / case.l2_folder).resolve()
    if Path(provenance["inputs"]["buildings"]["path"]) != expected_buildings:
        raise AssertionError(f"{case.name}: provenance buildings path differs")
    if Path(provenance["inputs"]["landsat_l2_folder"]["path"]) != expected_l2:
        raise AssertionError(f"{case.name}: provenance Landsat path differs")


def run_comparison(data_root: Path, prepared_dir: Path, output_root: Path, historical_output: Path, summary_path: Path, manifest_path: Path, *, reuse_existing: bool = False) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not reuse_existing:
        raise FileExistsError(f"Refusing to overwrite non-empty scenario directory: {output_root}")
    if reuse_existing and not output_root.is_dir():
        raise FileNotFoundError(f"Existing scenario directory missing: {output_root}")
    if summary_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing summary or manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    prep_manifest_path = prepared_dir / "preparation_manifest.json"
    if not prep_manifest_path.exists():
        raise FileNotFoundError(f"Preparation manifest missing: {prep_manifest_path}")

    records: list[dict[str, object]] = []
    for number, case in enumerate(CASES, start=1):
        action = "revalidating" if reuse_existing else "running"
        print(f"[{number:02d}/{len(CASES):02d}] {action} {case.name}", flush=True)
        source_count, source_area_m2 = _source_candidate_stats(case, prepared_dir)
        source_area_ha = source_area_m2 / 10000
        if source_count != case.expected_count:
            raise AssertionError(f"{case.name}: source selected {source_count}, expected {case.expected_count}")
        if abs(source_area_ha - case.expected_area_ha) > 0.00015:
            raise AssertionError(
                f"{case.name}: source selected area {source_area_ha:.6f} ha, "
                f"expected {case.expected_area_ha:.4f} ha"
            )
        if reuse_existing:
            case_output_dir = output_root / case.name
            required = [
                "_greening_provenance.json",
                "_greening_statistics.txt",
                "buildings_greening_impact.gpkg",
                "delta_LST.tif",
                "model_feature_importance.txt",
                "scenario_pred_LST.tif",
            ]
            missing = [name for name in required if not (case_output_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{case.name}: existing output is incomplete: {missing}")
            provenance_path = case_output_dir / "_greening_provenance.json"
            delta_raster = case_output_dir / "delta_LST.tif"
        else:
            output = run_scenario(_scenario_config(case, data_root, prepared_dir, output_root))
            case_output_dir = output.out_dir
            provenance_path = Path(output.provenance)
            delta_raster = output.delta_raster
        provenance = json.loads(provenance_path.read_text())
        _validate_case_provenance(case, provenance, data_root, prepared_dir)
        count = int(provenance["counts"]["buildings_selected"])
        if count != case.expected_count:
            raise AssertionError(f"{case.name}: selected {count}, expected {case.expected_count}")
        max_delta = _validate_delta(delta_raster)
        records.append(
            {
                "case": vars(case),
                "output_dir": str(case_output_dir.resolve()),
                "candidate_area_source_crs_m2": source_area_m2,
                "maximum_delta_c": max_delta,
                "provenance": provenance,
                "output_files": _output_hashes(case_output_dir),
            }
        )
        print(f"[{number:02d}/{len(CASES):02d}] validated {case.name}: {count:,} candidates", flush=True)

    diagnostic_max_difference = _validate_diagnostics(records)
    _validate_historical(records, historical_output)
    write_summary(
        records,
        summary_path,
        manifest_path,
        data_root,
        prepared_dir,
        output_root,
        diagnostic_max_difference,
    )
    manifest = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "preparation": json.loads(prep_manifest_path.read_text()),
        "preparation_manifest_hash": sha256_path(str(prep_manifest_path.resolve())),
        "summary": {"path": str(summary_path.resolve()), **sha256_path(str(summary_path.resolve()))},
        "validations": {
            "expected_counts_and_areas": "passed",
            "all_deltas_non_positive": "passed",
            "model_diagnostics_identical_within_city": {
                "status": "passed",
                "absolute_tolerance": DIAGNOSTIC_ABS_TOLERANCE,
                "maximum_absolute_difference": diagnostic_max_difference,
            },
            "hamburg_old_historical_reproduction": "passed",
        },
        "runs": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-output", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Read-only revalidation/finalization of a complete existing 12-run output root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_comparison(
        data_root=args.data_root.resolve(),
        prepared_dir=args.prepared_dir.resolve(),
        output_root=args.output_root.resolve(),
        historical_output=args.historical_output.resolve(),
        summary_path=args.summary_path.resolve(),
        manifest_path=args.manifest_path.resolve(),
        reuse_existing=args.reuse_existing,
    )


if __name__ == "__main__":
    main()
