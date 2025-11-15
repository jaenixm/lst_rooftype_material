"""High-level orchestration for the green roof scenario workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats

from .config import ScenarioConfig
from .io import read_raster, save_raster
from .l2 import build_lst_from_l2, compute_ndvi_albedo_from_l2
from .masking import roof_mask_fraction, subset_buildings
from .modeling import fit_model, predict_model, predict_partial

logger = logging.getLogger(__name__)


@dataclass
class ScenarioOutputs:
    out_dir: Path
    scenario_raster: Path
    delta_raster: Path
    buildings_layer: Path
    baseline_pred_raster: Optional[Path] = None
    roof_fraction_raster: Optional[Path] = None


def run_scenario(config: ScenarioConfig) -> ScenarioOutputs:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lst_path = config.lst
    if config.build_lst and lst_path is None:
        lst_path = out_dir / "baseline_LST.tif"

    if config.build_lst:
        lst_path, lst, lst_profile = build_lst_from_l2(
            config.l2_folder,
            out_path=lst_path,
            unit=config.lst_unit,
            keep_water=config.keep_lst_water,
        )
    else:
        if lst_path is None:
            raise ValueError("Provide lst path or enable build_lst to derive it from the Landsat L2 folder.")
        lst, lst_profile = read_raster(lst_path)

    template_profile = lst_profile

    if config.layer:
        buildings = gpd.read_file(config.buildings, layer=config.layer)
    else:
        buildings = gpd.read_file(config.buildings)
    if buildings.crs != template_profile["crs"]:
        buildings = buildings.to_crs(template_profile["crs"])

    bld_green = subset_buildings(
        buildings,
        config.roof_field,
        config.roof_types,
        keep_null_roof=config.keep_null_roof,
    )
    if config.min_roof_area > 0:
        bld_green = bld_green[bld_green.geometry.area >= config.min_roof_area].copy()

    if bld_green.empty:
        raise ValueError("No buildings match the requested roof types to green.")

    ndvi, albedo, ndbi, _ = compute_ndvi_albedo_from_l2(config.l2_folder, lst_path)

    if config.target_ndvi is None:
        valid = np.isfinite(ndvi) & np.isfinite(lst)
        veg = ndvi[valid & (ndvi > 0.3)]
        target_ndvi = float(np.median(veg)) if veg.size else 0.5
    else:
        target_ndvi = float(config.target_ndvi)
    target_albedo = float(config.target_albedo)

    pixel_size = abs(template_profile["transform"].a)
    if config.min_sample_spacing > 0:
        block_size = max(1, int(round(config.min_sample_spacing / pixel_size)))
    else:
        block_size = None

    model, metrics = fit_model(
        lst,
        ndvi,
        albedo,
        ndbi,
        frac=config.sample_frac,
        seed=config.random_state,
        model_type=config.model,
        block_size=block_size,
    )
    logger.info(
        "Model fitted (type=%s): R2 train=%.3f test=%.3f, RMSE train=%.3f test=%.3f",
        config.model,
        metrics["r2_train"],
        metrics["r2_test"],
        metrics["rmse_train"],
        metrics["rmse_test"],
    )

    baseline_pred_path = None
    if config.write_pred_baseline:
        baseline_pred = predict_model(model, ndvi, albedo, ndbi)
        baseline_pred_path = out_dir / "baseline_pred_LST.tif"
        save_raster(baseline_pred_path, baseline_pred, template_profile)

    roof_frac = roof_mask_fraction(
        bld_green,
        template_profile,
        supersample=config.supersample,
        all_touched=config.all_touched,
    )
    roof_fraction_raster = None
    if config.write_roof_fraction_raster:
        roof_fraction_raster = out_dir / "roof_fraction.tif"
        save_raster(roof_fraction_raster, roof_frac.astype("float32"), template_profile)

    f = np.clip(roof_frac, 0.0, 1.0).astype("float32")
    valid_lsts = np.isfinite(lst)
    pred_mask = (f > 0) & valid_lsts

    scen_ndvi = (1.0 - f) * ndvi + f * target_ndvi
    scen_albedo = (1.0 - f) * albedo + f * target_albedo

    scen_pred = predict_partial(model, scen_ndvi, scen_albedo, pred_mask, ndbi)
    baseline_partial = predict_partial(model, ndvi, albedo, pred_mask, ndbi)

    scen_pred_filled = scen_pred.copy()
    baseline_pred_filled = baseline_partial.copy()
    scen_pred_filled[~pred_mask] = lst[~pred_mask]
    baseline_pred_filled[~pred_mask] = lst[~pred_mask]

    delta = np.zeros(ndvi.shape, dtype="float32")
    diff = scen_pred_filled - baseline_pred_filled
    mfin = pred_mask & np.isfinite(diff)
    delta[mfin] = diff[mfin]
    delta[~valid_lsts] = np.nan

    scenario_raster = out_dir / "scenario_pred_LST.tif"
    delta_raster = out_dir / "delta_LST.tif"
    save_raster(scenario_raster, scen_pred_filled, template_profile)
    save_raster(delta_raster, delta, template_profile)

    stats = zonal_stats(
        buildings,
        delta,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    buildings = buildings.copy()
    buildings["delta_mean"] = [z["mean"] for z in stats]

    buildings_layer = out_dir / "buildings_greening_impact.gpkg"
    buildings.to_file(buildings_layer, driver="GPKG")

    provenance = (
        f"Roof types converted: {config.roof_types}\n"
        f"Target NDVI: {target_ndvi}\n"
        f"Target Albedo: {target_albedo}\n"
        f"LST source: {'built from L2 folder' if config.build_lst else lst_path}\n"
        f"Supersample: {config.supersample}\n"
        f"Model type: {config.model}\n"
        f"Used NDBI predictor: yes\n"
    )
    if config.build_lst:
        provenance += f"LST build options: unit={config.lst_unit}, keep_water={config.keep_lst_water}\n"

    provenance_path = out_dir / "_greening_provenance.txt"
    provenance_path.write_text(provenance, encoding="utf-8")

    logger.info("Scenario written to %s", out_dir)

    return ScenarioOutputs(
        out_dir=out_dir,
        scenario_raster=scenario_raster,
        delta_raster=delta_raster,
        buildings_layer=buildings_layer,
        baseline_pred_raster=baseline_pred_path,
        roof_fraction_raster=roof_fraction_raster,
    )
