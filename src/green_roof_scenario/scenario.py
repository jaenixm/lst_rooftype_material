"""High-level orchestration for the green roof scenario workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import mapping

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
    feature_importance: Optional[Path] = None


def _load_boundary(boundary_path: Path, target_crs) -> tuple[list[dict], gpd.GeoDataFrame]:
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary layer {boundary_path} is empty.")
    boundary = boundary[boundary.geometry.notnull()].copy()
    boundary = boundary.to_crs(target_crs)
    geom = boundary.geometry.unary_union
    if geom.is_empty:
        raise ValueError(f"Boundary layer {boundary_path} has no geometry after reprojecting to the raster CRS.")
    return [mapping(geom)], boundary


def _clip_raster_to_boundary(arr: np.ndarray, profile: dict, geoms: list[dict]) -> tuple[np.ndarray, dict]:
    tmp_profile = profile.copy()
    tmp_profile.setdefault("driver", "GTiff")
    tmp_profile.update(count=1)
    with MemoryFile() as memfile:
        with memfile.open(**tmp_profile) as ds:
            ds.write(arr, 1)
            clipped, out_transform = mask(ds, geoms, crop=True)
            out_profile = ds.profile.copy()
            out_profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=out_transform,
                nodata=np.nan,
                dtype="float32",
                count=1,
            )
    clipped_arr = np.ma.filled(clipped, np.nan).astype("float32")[0]
    return clipped_arr, out_profile


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

    boundary_geoms: Optional[list[dict]] = None
    boundary_gdf: Optional[gpd.GeoDataFrame] = None
    if config.boundary:
        boundary_geoms, boundary_gdf = _load_boundary(config.boundary, template_profile["crs"])

    if config.layer:
        buildings = gpd.read_file(config.buildings, layer=config.layer)
    else:
        buildings = gpd.read_file(config.buildings)
    if buildings.crs != template_profile["crs"]:
        buildings = buildings.to_crs(template_profile["crs"])
    if boundary_gdf is not None:
        buildings = gpd.clip(buildings, boundary_gdf)
        buildings = buildings[buildings.geometry.notnull()].copy()
        if buildings.empty:
            raise ValueError("No buildings intersect the provided boundary.")

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

    if boundary_geoms is not None:
        base_profile = template_profile.copy()
        lst, template_profile = _clip_raster_to_boundary(lst, base_profile, boundary_geoms)
        ndvi, _ = _clip_raster_to_boundary(ndvi, base_profile, boundary_geoms)
        albedo, _ = _clip_raster_to_boundary(albedo, base_profile, boundary_geoms)
        ndbi, _ = _clip_raster_to_boundary(ndbi, base_profile, boundary_geoms)
        if config.build_lst:
            save_raster(lst_path, lst, template_profile)

    if config.write_indices_rasters:
        save_raster(out_dir / "ndvi.tif", ndvi.astype("float32"), template_profile)
        save_raster(out_dir / "albedo.tif", albedo.astype("float32"), template_profile)
        save_raster(out_dir / "ndbi.tif", ndbi.astype("float32"), template_profile)

    if config.target_ndvi is None:
        valid = np.isfinite(ndvi) & np.isfinite(lst)
        veg = ndvi[valid & (ndvi > 0.3)]
        target_ndvi = float(np.median(veg)) if veg.size else 0.5
    else:
        target_ndvi = float(config.target_ndvi)
    target_albedo = float(config.target_albedo)
    target_ndbi = float(config.target_ndbi)

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
    feat_report = out_dir / "model_feature_importance.txt"
    feature_importance_path: Optional[Path] = None
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        names = ["ndvi", "albedo", "ndbi"]
        lines = [f"{n}: {v:.4f}" for n, v in zip(names, vals)]
        feat_report.write_text("Random Forest feature importances (sum=1):\n" + "\n".join(lines), encoding="utf-8")
        feature_importance_path = feat_report
        logger.info("Feature importances saved to %s", feat_report)
    elif hasattr(model, "coef_"):
        vals = model.coef_
        names = ["ndvi", "albedo", "ndbi"]
        lines = [f"{n}: {v:.4f}" for n, v in zip(names, vals)]
        feat_report.write_text("Linear model coefficients:\n" + "\n".join(lines), encoding="utf-8")
        feature_importance_path = feat_report
        logger.info("Linear coefficients saved to %s", feat_report)

    baseline_pred = predict_model(model, ndvi, albedo, ndbi)

    baseline_pred_path = None
    if config.write_pred_baseline:
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
    scen_ndbi = (1.0 - f) * ndbi + f * target_ndbi

    scen_pred = predict_partial(model, scen_ndvi, scen_albedo, pred_mask, scen_ndbi)

    # 2. Calculate Delta: Scenario Prediction - Baseline Prediction
    # This isolates the effect of the green roof by canceling out the model's bias.
    delta = np.zeros(ndvi.shape, dtype="float32")
    
    # Ensure we only calculate delta where we have valid predictions AND valid input data
    mfin = pred_mask & np.isfinite(scen_pred) & np.isfinite(baseline_pred) & valid_lsts
    
    # THE FIX: Subtract baseline_pred, NOT lst
    delta[mfin] = scen_pred[mfin] - baseline_pred[mfin]

    if config.clip_positive_delta:
        # Optional: keep only cooling effects; warmings are nulled out
        delta[(delta > 0) & np.isfinite(delta)] = 0.0

    # 3. Create the final "Scenario Absolute Temperature" raster
    # Instead of pasting the raw model prediction (which has bias) into the observed LST,
    # we apply the clean 'delta' to the observed 'lst'. 
    # This prevents visual "seams" between changed and unchanged pixels.
    scen_pred_filled = lst.copy()
    scen_pred_filled[mfin] = lst[mfin] + delta[mfin]

    # Handle NaNs for output
    delta[~valid_lsts] = np.nan
    scen_pred_filled[~valid_lsts] = np.nan

    scenario_raster = out_dir / "scenario_pred_LST.tif"
    delta_raster = out_dir / "delta_LST.tif"
    save_raster(scenario_raster, scen_pred_filled, template_profile)
    save_raster(delta_raster, delta, template_profile)

    baseline_stats = zonal_stats(
        buildings,
        lst,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    scenario_stats = zonal_stats(
        buildings,
        scen_pred_filled,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    delta_stats = zonal_stats(
        buildings,
        delta,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    buildings = buildings.copy()
    buildings["lst_baseline_mean"] = [z["mean"] for z in baseline_stats]
    buildings["lst_scenario_mean"] = [z["mean"] for z in scenario_stats]
    buildings["delta_mean"] = [z["mean"] for z in delta_stats]

    buildings_layer = out_dir / "buildings_greening_impact.gpkg"
    buildings.to_file(buildings_layer, driver="GPKG")

    provenance = (
        f"Roof types converted: {config.roof_types}\n"
        f"Boundary: {config.boundary if config.boundary else 'None'}\n"
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
        feature_importance=feature_importance_path,
    )
