#!/usr/bin/env python3
"""
Green Roof Scenario (Empirical LST Model)

Steps:
1) Load inputs (Landsat L2 folder to compute NDVI/Albedo/NDBI).
2) Fit a simple model: LST = a + b*NDVI + c*Albedo + d*NDBI (linear or RF).
3) Build a roof mask for selected roof types; estimate per-pixel roof fraction via supersampled rasterization.
4) Modify NDVI & Albedo only over roof pixels: new = (1-f)*orig + f*target.
5) Predict baseline vs. scenario; write ΔLST = scenario - baseline.
6) Summarize ΔLST per building and export GPKG.

Requires: numpy, rasterio, geopandas, rasterstats, scikit-learn

python3 green_roof_scenario.py \
  --lst data/lst_bremen.tif \
  --l2_folder data/LC08_L2SP_196023_20240813_20240822_02_T1 \
  --buildings results/buildings_with_lst.gpkg \
  --roof_field predictedroofmaterials \
  --roof_types "concrete, tar_paper" \
  --out_dir results_greening_0.4 \
  --supersample 4 \
  --write_pred_baseline \
  --target_ndvi 0.4 \
  --write_roof_fraction_raster
"""

import argparse
from pathlib import Path
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import mapping
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rasterstats import zonal_stats

# ---- Grid alignment check ----------------------------------------------------

def _assert_same_grid(reference_profile, other_profile, label="raster"):
    """Raise ValueError if CRS, transform, width, or height differ from reference.
    Allows small floating tolerance on transform.
    """
    ref = reference_profile
    oth = other_profile
    # CRS
    if ref.get("crs") != oth.get("crs"):
        raise ValueError(f"{label} CRS does not match LST CRS. Reproject/resample to LST grid.")
    # Shape
    if (ref.get("width"), ref.get("height")) != (oth.get("width"), oth.get("height")):
        raise ValueError(f"{label} dimensions do not match LST dimensions. Resample to LST grid.")
    # Transform (allow small tolerance)
    import numpy as _np
    ref_t = ref.get("transform")
    oth_t = oth.get("transform")
    if not _np.allclose([ref_t.a, ref_t.b, ref_t.c, ref_t.d, ref_t.e, ref_t.f],
                        [oth_t.a, oth_t.b, oth_t.c, oth_t.d, oth_t.e, oth_t.f],
                        rtol=0, atol=1e-6):
        raise ValueError(f"{label} geotransform does not match LST grid. Resample to LST grid.")

# Landsat C2 L2 surface reflectance scaling
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

def parse_args():
    p = argparse.ArgumentParser(description="Greened roof LST scenario (empirical model).")
    # Core I/O
    p.add_argument("--lst", required=True, help="Baseline LST raster (°C), aligned to Landsat grid.")
    p.add_argument("--buildings", required=True, help="Buildings file (GPKG/GeoJSON/shp).")
    p.add_argument("--layer", default=None, help="Optional layer name inside GPKG.")
    p.add_argument("--roof_field", default="predictedrooftypematerial", help="Roof type field.")
    p.add_argument("--roof_types", required=True,
                   help="Comma-separated roof types to convert to green (e.g., 'concrete,bitumen').")
    p.add_argument("--out_dir", default="results_greening", help="Output folder.")

    # NDVI/Albedo/NDBI source (Landsat L2)
    p.add_argument(
        "--l2_folder",
        required=True,
        help="Path to Landsat L2 scene folder (will compute NDVI, Albedo, and NDBI).",
    )

    # Targets & model
    p.add_argument("--target_ndvi", type=float, default=None,
                   help="Target NDVI for green roofs (default: auto from local veg).")
    p.add_argument("--target_albedo", type=float, default=0.20,
                   help="Target broadband albedo for green roofs (default: 0.20).")
    p.add_argument("--sample_frac", type=float, default=0.1,
                   help="Random sample fraction of valid pixels for model fitting (default: 0.1).")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--model",
        choices=["linear", "rf"],
        default="linear",
        help="Model type: 'linear' (default) or 'rf' (Random Forest).",
    )

    # Rasterization / blending
    p.add_argument("--supersample", type=int, default=4,
                   help="Supersampling factor to estimate roof fraction per pixel (default: 4).")
    p.add_argument("--all_touched", action="store_true",
                   help="Use all_touched=True for coarse roof mask (fast, no fractions).")

    # Summaries
    p.add_argument("--write_pred_baseline", action="store_true",
                   help="Write model baseline prediction raster.")
    p.add_argument("--keep_null_roof", action="store_true",
                   help="Keep features with NULL roof field instead of dropping them.")
    p.add_argument(
        "--write_roof_fraction_raster",
        action="store_true",
        help="Write the per-pixel roof fraction raster (0..1) to out_dir.",
    )
    return p.parse_args()

def _find_band(folder, suffix):
    import glob, os
    pats = [str(Path(folder)/f"*{suffix}.TIF"),
            str(Path(folder)/f"*{suffix}.tif")]
    for pat in pats:
        m = glob.glob(pat)
        if m:
            return m[0]
    return None


# --- Landsat L2 NDVI/Albedo helpers ------------------------------------------

def _read_rescaled_to_template(path, shape):
    """Read a single-band raster, resample to given shape, and apply Landsat SR scaling."""
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=shape, resampling=Resampling.bilinear)
    return arr.astype("float32") * SR_SCALE + SR_OFFSET


def _load_l2_bands_to_template(l2_folder, template_path):
    """Load red (B4), nir (B5) and optional swir1 (B6) from an L2 folder, aligned to template."""
    b4 = _find_band(l2_folder, "_SR_B4")
    b5 = _find_band(l2_folder, "_SR_B5")
    b6 = _find_band(l2_folder, "_SR_B6")  # optional

    if not (b4 and b5):
        raise FileNotFoundError("Could not find SR_B4 and SR_B5 in L2 folder.")

    with rasterio.open(template_path) as tmp:
        profile = tmp.profile.copy()
        shape = (tmp.height, tmp.width)

    red = _read_rescaled_to_template(b4, shape)
    nir = _read_rescaled_to_template(b5, shape)
    swir1 = _read_rescaled_to_template(b6, shape) if b6 else None

    return red, nir, swir1, profile


def _compute_indices(red, nir, swir1):
    """Compute NDVI, Albedo, and NDBI (if SWIR1 available) from rescaled bands."""
    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Albedo
    albedo = 0.356 * red + 0.130 * nir + 0.373 * swir1 + 0.0018
    albedo = np.clip(albedo, 0.0, 1.0)
    
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    ndbi = np.clip(ndbi, -1.0, 1.0)

    return ndvi, albedo, ndbi

def compute_ndvi_albedo_from_l2(l2_folder, template_path):
    """Compute NDVI (B5,B4), broadband albedo, and NDBI aligned to template."""
    red, nir, swir1, profile = _load_l2_bands_to_template(l2_folder, template_path)
    ndvi, albedo, ndbi = _compute_indices(red, nir, swir1)
    return ndvi, albedo, ndbi, profile

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        # ensure width/height present for downstream checks
        prof.update(width=src.width, height=src.height, transform=src.transform, crs=src.crs)
    return arr, prof

def save_raster(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)

def sample_model_inputs(lst, ndvi, albedo, ndbi, frac, seed, ):
    mask = np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(albedo) & np.isfinite(ndbi)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("No valid pixels for modeling.")
    rng = np.random.default_rng(seed)
    n = max(1000, int(frac * len(idx)))
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    X = np.column_stack([ndvi.flat[sel], albedo.flat[sel], ndbi.flat[sel]])
    y = lst.flat[sel]
    return X, y

def fit_model(lst, ndvi, albedo, ndbi, frac=0.1, seed=42, model_type="linear"):
    X, y = sample_model_inputs(lst, ndvi, albedo, ndbi, frac, seed)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    if model_type == "linear":
        model = LinearRegression().fit(X_train, y_train)
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ).fit(X_train, y_train)

    # Training metrics
    r2_train = model.score(X_train, y_train)
    y_train_pred = model.predict(X_train)
    rmse_train = float(np.sqrt(np.mean((y_train - y_train_pred)**2)))

    # Test metrics
    r2_test = model.score(X_test, y_test)
    y_test_pred = model.predict(X_test)
    rmse_test = float(np.sqrt(np.mean((y_test - y_test_pred)**2)))

    if model_type == "linear":
        print("Linear model fitted: LST = a + b*NDVI + c*Albedo + d*NDBI")
        print(f"Intercept (a): {model.intercept_}")
        print(f"Coefficient for NDVI (b): {model.coef_[0]}")
        print(f"Coefficient for Albedo (c): {model.coef_[1]}")
        if len(model.coef_) > 2:
            print(f"Coefficient for NDBI (d): {model.coef_[2]}")
    else:
        print("Random Forest model fitted with features:")
        feature_names = ["NDVI", "Albedo", "NDBI"]
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            print(f"  {name}: importance={imp:.3f}")
    print("--- Training performance ---")
    print(f"R² (train): {r2_train}")
    print(f"RMSE (train): {rmse_train}")
    print("--- Test performance ---")
    print(f"R² (test): {r2_test}")
    print(f"RMSE (test): {rmse_test}")

    return model

def predict_model(model, ndvi, albedo, ndbi):
    Xfull = np.column_stack([ndvi.ravel(), albedo.ravel(), ndbi.ravel()])
    yhat = model.predict(Xfull).reshape(ndvi.shape)
    return yhat

def predict_partial(model, ndvi, albedo, mask, ndbi):
    """Predict only where mask==True; elsewhere return NaN.
    ndvi, albedo: 2D arrays
    mask: boolean 2D array
    ndbi: 2D array
    """
    out = np.full(ndvi.shape, np.nan, dtype="float32")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    X = np.column_stack([ndvi.ravel()[idx], albedo.ravel()[idx], ndbi.ravel()[idx]])
    yhat = model.predict(X).astype("float32")
    out.ravel()[idx] = yhat
    return out

def roof_mask_fraction(buildings_gdf, template_profile, supersample=4, all_touched=False):
    """Return per-pixel roof fraction in [0,1]. If all_touched=True, returns 0/1 mask."""
    shapes = [mapping(geom) for geom in buildings_gdf.geometry.values]
    height, width = template_profile["height"], template_profile["width"]
    transform = template_profile["transform"]

    if all_touched or supersample <= 1:
        mask = rasterize(shapes, out_shape=(height, width), transform=transform,
                         fill=0, default_value=1, all_touched=True).astype("float32")
        return mask  # 0/1

    # Supersampled grid to approximate area fraction
    hss, wss = height*supersample, width*supersample
    # scale the transform
    a, b, c, d, e, f = transform[:6]
    up_transform = rasterio.Affine(a/supersample, b, c, d, e/supersample, f)
    up = rasterize(shapes, out_shape=(hss, wss), transform=up_transform,
                   fill=0, default_value=1, all_touched=True).astype("float32")
    # Downsample by block averaging
    up = up.reshape(height, supersample, width, supersample).mean(axis=(1,3))
    return up

def subset_buildings(gdf, roof_field, roof_types, keep_null_roof=False):
    if roof_field not in gdf.columns:
        raise KeyError(f"Roof field '{roof_field}' not in buildings.")
    if not keep_null_roof:
        gdf = gdf.dropna(subset=[roof_field]).copy()
    wanted = {t.strip().lower() for t in roof_types.split(",")}
    sel = gdf[roof_field].str.lower().isin(wanted)
    return gdf.loc[sel].copy()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Read baseline LST (°C)
    lst, lst_profile = read_raster(args.lst)
    template_profile = lst_profile

    # Load buildings and filter to roofs we’ll green
    if args.layer:
        bld = gpd.read_file(args.buildings, layer=args.layer)
    else:
        bld = gpd.read_file(args.buildings)
    if bld.crs != template_profile["crs"]:
        bld = bld.to_crs(template_profile["crs"])

    bld_green = subset_buildings(bld, args.roof_field, args.roof_types, args.keep_null_roof)
    if bld_green.empty:
        raise ValueError("No buildings match the requested roof types to green.")

    # Get NDVI, Albedo, and NDBI
    if args.l2_folder:
        ndvi, albedo, ndbi, _ = compute_ndvi_albedo_from_l2(args.l2_folder, args.lst)
    else:
        raise ValueError("This script now expects Landsat L2 input via --l2_folder so that NDVI, Albedo, and NDBI can all be computed. The --ndvi_albedo mode is no longer supported.")

    # Auto target NDVI if not given: median of vegetated pixels within LST-valid area
    if args.target_ndvi is None:
        valid = np.isfinite(ndvi) & np.isfinite(lst)
        veg = ndvi[valid & (ndvi > 0.3)]
        target_ndvi = float(np.median(veg)) if veg.size else 0.5
    else:
        target_ndvi = float(args.target_ndvi)
    target_albedo = float(args.target_albedo)

    # Fit model on baseline
    model = fit_model(lst, ndvi, albedo, ndbi, frac=args.sample_frac, seed=args.random_state, model_type=args.model)
    if args.write_pred_baseline:
        baseline_pred = predict_model(model, ndvi, albedo, ndbi)
        save_raster(out_dir / "baseline_pred_LST.tif", baseline_pred, template_profile)

    # Build per-pixel roof fraction mask for selected roofs
    roof_frac = roof_mask_fraction(bld_green, template_profile,
                                   supersample=args.supersample,
                                   all_touched=args.all_touched)
    if args.write_roof_fraction_raster:
        save_raster(out_dir / "roof_fraction.tif", roof_frac.astype("float32"), template_profile)

    # Create scenario NDVI/Albedo by blending toward targets where roofs are present
    # new = (1 - f)*orig + f*target
    f = np.clip(roof_frac, 0.0, 1.0).astype("float32")

    valid_lsts = np.isfinite(lst)
    pred_mask = (f > 0) & valid_lsts

    scen_ndvi   = (1.0 - f) * ndvi   + f * target_ndvi
    scen_albedo = (1.0 - f) * albedo + f * target_albedo

    # Predict ONLY where roofs are affected (and LST is valid)
    scen_pred = predict_partial(model, scen_ndvi, scen_albedo, pred_mask, ndbi)
    if args.write_pred_baseline:
        baseline_pred = predict_partial(model, ndvi, albedo, pred_mask, ndbi)
    else:
        baseline_pred = predict_partial(model, ndvi, albedo, pred_mask, ndbi)

    # Reuse original observed LST where no greening occurs (pred_mask == False)
    scen_pred_filled = scen_pred.copy()
    baseline_pred_filled = baseline_pred.copy()
    scen_pred_filled[~pred_mask] = lst[~pred_mask]
    baseline_pred_filled[~pred_mask] = lst[~pred_mask]

    # Delta is defined as 0 outside affected pixels, scenario-baseline inside
    delta = np.zeros(ndvi.shape, dtype="float32")
    diff = scen_pred_filled - baseline_pred_filled
    mfin = pred_mask & np.isfinite(diff)
    delta[mfin] = diff[mfin]
    # Preserve NaNs where original LST was invalid (water/clouds)
    delta[~valid_lsts] = np.nan

    # Save rasters
    save_raster(out_dir / "scenario_pred_LST.tif", scen_pred_filled, template_profile)
    save_raster(out_dir / "delta_LST.tif", delta, template_profile)

    # Summarize ΔLST per building (all buildings, so you can rank impacts)
    # (If you prefer only greened buildings, pass bld_green to zonal_stats.)
    zs = zonal_stats(bld, delta, affine=template_profile["transform"],
                     nodata=np.nan, stats=["mean"], all_touched=True)
    bld["delta_mean"] = [z["mean"] for z in zs]

    # Write summary layer
    out_gpkg = out_dir / "buildings_greening_impact.gpkg"
    bld.to_file(out_gpkg, driver="GPKG")
    print(f"Wrote scenario rasters and {out_gpkg}")

    # Tiny provenance file
    (out_dir / "_greening_provenance.txt").write_text(
        f"Roof types converted: {args.roof_types}\n"
        f"Target NDVI: {target_ndvi} (computed within LST-valid area if not provided)\n"
        f"Target Albedo: {target_albedo}\n"
        f"Supersample: {args.supersample}\n"
        f"Model type: {args.model}\n"
        f"Used NDBI predictor: yes\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()