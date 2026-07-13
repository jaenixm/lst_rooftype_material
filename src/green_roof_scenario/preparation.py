"""Prepare slope-enriched comparison inputs without precomputing material choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import geopandas as gpd
import numpy as np
import shapely

from .masking import parse_array_value
from .provenance import sha256_path

CITY_SOURCES = {
    "madrid": {
        "materials": "data/buildings/madrid-cls-finale.gpkg",
        "slopes": "data/buildings/madrid-cls-finale_with_slope_dominant.gpkg",
        "slope_field": "slope_median_deg",
        "output": "madrid_comparison.gpkg",
    },
    "paris": {
        "materials": "data/buildings/paris-cls-finale-finetuned.gpkg",
        "slopes": "data/buildings/paris-cls-finale-finetuned_with_slope_dominant.gpkg",
        "slope_field": "slope_median_deg",
        "output": "paris_comparison.gpkg",
    },
}

HAMBURG_SOURCES = {
    "hamburg_old": {
        "materials": "data/processed/buildings/hamburg-cls-finale.geojson",
        "output": "hamburg_old_comparison.gpkg",
    },
    "hamburg_new": {
        "materials": "data/buildings/Hamburg-cls-with-probs.geojson",
        "output": "hamburg_new_comparison.gpkg",
    },
}
HAMBURG_SLOPES = "data/buildings/hamburg-cls-finale-with-slopes_dominant.gpkg"


def iter_geojson_properties(path: Path) -> Iterator[dict[str, object]]:
    """Incrementally decode GeoJSON features and yield only their properties."""

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as stream:
        while not started:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"GeoJSON {path} has no features array")
            buffer += chunk
            key_at = buffer.find('"features"')
            if key_at < 0:
                buffer = buffer[-64:]
                continue
            array_at = buffer.find("[", key_at + len('"features"'))
            if array_at < 0:
                continue
            buffer = buffer[array_at + 1 :]
            started = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                feature, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"Malformed or truncated GeoJSON feature in {path}") from exc
                chunk = stream.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
                raise ValueError(f"GeoJSON feature in {path} has no properties object")
            yield feature["properties"]
            buffer = buffer[end:]


def validate_material_arrays(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    """Validate all raw arrays and report scalar/multi/tie counts."""

    required = {"predicted_roof_materials", "material_cov"}
    missing = required.difference(gdf.columns)
    if missing:
        raise KeyError(f"Missing material columns: {sorted(missing)}")

    scalar = multi = ties = 0
    errors: list[str] = []
    for position, (material, coverage) in enumerate(
        zip(gdf["predicted_roof_materials"], gdf["material_cov"])
    ):
        try:
            materials = parse_array_value(material, field_name="predicted_roof_materials")
            coverages_raw = parse_array_value(coverage, field_name="material_cov")
            if len(materials) != len(coverages_raw):
                raise ValueError(f"array lengths differ ({len(materials)} != {len(coverages_raw)})")
            coverages = np.asarray([float(value) for value in coverages_raw], dtype="float64")
            if not np.all(np.isfinite(coverages)):
                raise ValueError("material_cov contains a non-finite value")
            scalar += len(materials) == 1
            multi += len(materials) > 1
            ties += int(np.count_nonzero(coverages == np.max(coverages)) > 1)
        except (TypeError, ValueError) as exc:
            if len(errors) < 10:
                errors.append(f"position={position}: {exc}")
    if errors:
        raise ValueError("Malformed material arrays: " + "; ".join(errors))
    if ties:
        raise ValueError(f"Found {ties} maximum-coverage ties; expected none in current inputs")
    return {"records": len(gdf), "scalar_records": scalar, "multi_records": multi, "ties": ties}


def _require_2d(gdf: gpd.GeoDataFrame, label: str) -> None:
    if bool(np.any(shapely.has_z(gdf.geometry.array))):
        raise ValueError(f"{label} contains 3D geometry; expected 2D")


def prepare_madrid_or_paris(city: str, data_root: Path, output_path: Path) -> dict[str, object]:
    spec = CITY_SOURCES[city]
    material_path = data_root / spec["materials"]
    slope_path = data_root / spec["slopes"]
    materials = gpd.read_file(material_path)
    slopes = gpd.read_file(slope_path)

    if len(materials) != len(slopes):
        raise ValueError(f"{city}: feature counts differ ({len(materials)} != {len(slopes)})")
    if materials.crs != slopes.crs:
        raise ValueError(f"{city}: CRS differs ({materials.crs} != {slopes.crs})")
    _require_2d(materials, f"{city} materials")
    _require_2d(slopes, f"{city} slopes")
    order_matches = shapely.equals(materials.geometry.array, slopes.geometry.array)
    if not bool(np.all(order_matches)):
        first = int(np.flatnonzero(~order_matches)[0])
        raise ValueError(f"{city}: geometry/order mismatch at position {first}")

    slope_field = str(spec["slope_field"])
    if slope_field not in slopes.columns:
        raise KeyError(f"{city}: slope field {slope_field!r} missing")
    prepared = materials.copy()
    prepared[slope_field] = slopes[slope_field].to_numpy(copy=True)
    validation = validate_material_arrays(prepared)
    prepared.to_file(output_path, layer="buildings", driver="GPKG", index=False)
    return {
        "dataset": city,
        "output": str(output_path.resolve()),
        "feature_count": len(prepared),
        "crs": str(prepared.crs),
        "material_validation": validation,
        "sources": {
            "materials": {"path": str(material_path.resolve()), **sha256_path(str(material_path.resolve()))},
            "slopes": {"path": str(slope_path.resolve()), **sha256_path(str(slope_path.resolve()))},
        },
        "output_hash": sha256_path(str(output_path.resolve())),
    }


def _as_compact_json_array(value: object, field_name: str) -> str:
    return json.dumps(parse_array_value(value, field_name=field_name), separators=(",", ":"))


def prepare_hamburg(dataset: str, data_root: Path, output_path: Path) -> dict[str, object]:
    spec = HAMBURG_SOURCES[dataset]
    material_path = data_root / spec["materials"]
    slope_path = data_root / HAMBURG_SLOPES

    geometry = gpd.read_file(material_path, columns=["gml_id"])
    slopes = gpd.read_file(slope_path)
    required_slopes = {"gml_id", "roof_slope_mean_deg"}
    missing = required_slopes.difference(slopes.columns)
    if missing:
        raise KeyError(f"Hamburg slope source missing columns: {sorted(missing)}")
    if geometry.crs != slopes.crs:
        raise ValueError(f"{dataset}: CRS differs ({geometry.crs} != {slopes.crs})")
    if geometry["gml_id"].duplicated().any() or slopes["gml_id"].duplicated().any():
        raise ValueError(f"{dataset}: gml_id must be unique in both sources")

    ids: list[str] = []
    materials: list[str] = []
    coverages: list[str] = []
    for properties in iter_geojson_properties(material_path):
        if "gml_id" not in properties:
            raise KeyError(f"{dataset}: streamed feature lacks gml_id")
        ids.append(str(properties["gml_id"]))
        materials.append(_as_compact_json_array(properties.get("predicted_roof_materials"), "predicted_roof_materials"))
        coverages.append(_as_compact_json_array(properties.get("material_cov"), "material_cov"))

    geometry_ids = geometry["gml_id"].astype(str).tolist()
    if ids != geometry_ids:
        raise ValueError(f"{dataset}: streamed gml_id order does not match geometry order")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: streamed gml_id values are not unique")
    if set(ids) != set(slopes["gml_id"].astype(str)):
        raise ValueError(f"{dataset}: gml_id sets differ between material and slope sources")

    slope_lookup = slopes.set_index(slopes["gml_id"].astype(str))["roof_slope_mean_deg"]
    prepared = gpd.GeoDataFrame(
        {
            "gml_id": ids,
            "predicted_roof_materials": materials,
            "material_cov": coverages,
            "roof_slope_mean_deg": slope_lookup.loc[ids].to_numpy(),
        },
        geometry=shapely.force_2d(geometry.geometry.array),
        crs=geometry.crs,
    )
    validation = validate_material_arrays(prepared)
    prepared.to_file(output_path, layer="buildings", driver="GPKG", index=False)
    return {
        "dataset": dataset,
        "output": str(output_path.resolve()),
        "feature_count": len(prepared),
        "crs": str(prepared.crs),
        "material_validation": validation,
        "sources": {
            "materials": {"path": str(material_path.resolve()), **sha256_path(str(material_path.resolve()))},
            "slopes": {"path": str(slope_path.resolve()), **sha256_path(str(slope_path.resolve()))},
        },
        "output_hash": sha256_path(str(output_path.resolve())),
    }


def prepare_all(data_root: Path, out_dir: Path) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty preparation directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for city, spec in CITY_SOURCES.items():
        records.append(prepare_madrid_or_paris(city, data_root, out_dir / str(spec["output"])))
    for dataset, spec in HAMBURG_SOURCES.items():
        records.append(prepare_hamburg(dataset, data_root, out_dir / str(spec["output"])))
    manifest = out_dir / "preparation_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "datasets": records}, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Repository root containing data/.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New, empty output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = prepare_all(args.data_root.resolve(), args.out_dir.resolve())
    print(manifest)


if __name__ == "__main__":
    main()
