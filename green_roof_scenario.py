#!/usr/bin/env python3
"""
Green Roof Scenario (Empirical LST Model)

Steps:
1) Load inputs (either Landsat L2 folder to compute NDVI/Albedo, or provided rasters).
2) Fit a simple model: LST = a + b*NDVI + c*Albedo  (linear by default).
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
  --out_dir results_greening \
  --supersample 4 \
  --write_pred_baseline \
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

    # NDVI/Albedo sources
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--l2_folder", help="Path to Landsat L2 scene folder (will compute NDVI/Albedo).")
    src.add_argument("--ndvi_albedo", nargs=2, metavar=("NDVI_TIF", "ALBEDO_TIF"),
                     help="Precomputed NDVI and Albedo rasters (aligned to LST).")

    # Targets & model
    p.add_argument("--target_ndvi", type=float, default=None,
                   help="Target NDVI for green roofs (default: auto from local veg).")
    p.add_argument("--target_albedo", type=float, default=0.20,
                   help="Target broadband albedo for green roofs (default: 0.20).")
    p.add_argument("--sample_frac", type=float, default=0.1,
                   help="Random sample fraction of valid pixels for model fitting (default: 0.1).")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")

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

def compute_ndvi_albedo_from_l2(l2_folder, template_path):
    """Compute NDVI (B5,B4) and broadband albedo (simple Liang-style OLI proxy) aligned to template."""
    b4 = _find_band(l2_folder, "_SR_B4")
    b5 = _find_band(l2_folder, "_SR_B5")
    b6 = _find_band(l2_folder, "_SR_B6")  # optional (useful if adding NDBI later)
    if not (b4 and b5):
        raise FileNotFoundError("Could not find SR_B4 and SR_B5 in L2 folder.")

    with rasterio.open(template_path) as tmp:
        profile = tmp.profile.copy()
        shape = (tmp.height, tmp.width)

    # Read & resample bands to template grid (if needed), apply SR scaling
    def read_rescaled(path):
        with rasterio.open(path) as src:
            arr = src.read(1, out_shape=shape, resampling=Resampling.bilinear)
            return arr.astype("float32") * SR_SCALE + SR_OFFSET

    red = read_rescaled(b4)
    nir = read_rescaled(b5)

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Broadband albedo (very simple proxy using R,NIR,SWIR1 weights).
    # If B6 missing, fall back to a 2-band proxy.
    if b6:
        swir1 = read_rescaled(b6)
        # Coeffs are a pragmatic proxy; tune with local measurements if available.
        albedo = 0.356*red + 0.130*nir + 0.373*swir1 + 0.0018
    else:
        albedo = 0.4*red + 0.35*nir + 0.0018
    albedo = np.clip(albedo, 0.0, 1.0)

    return ndvi, albedo, profile

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

def sample_model_inputs(lst, ndvi, albedo, frac, seed):
    mask = np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(albedo)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("No valid pixels for modeling.")
    rng = np.random.default_rng(seed)
    n = max(1000, int(frac * len(idx)))
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    X = np.column_stack([ndvi.flat[sel], albedo.flat[sel]])
    y = lst.flat[sel]
    return X, y

def fit_model(lst, ndvi, albedo, frac=0.1, seed=42):
    X, y = sample_model_inputs(lst, ndvi, albedo, frac, seed)
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"Linear model fitted: LST = a + b*NDVI + c*Albedo")
    print(f"Intercept (a): {model.intercept_}")
    print(f"Coefficient for NDVI (b): {model.coef_[0]}")
    print(f"Coefficient for Albedo (c): {model.coef_[1]}")
    print(f"R² on training sample: {r2}")
    return model

def predict_model(model, ndvi, albedo):
    Xfull = np.column_stack([ndvi.ravel(), albedo.ravel()])
    yhat = model.predict(Xfull).reshape(ndvi.shape)
    return yhat

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

    # Get NDVI and Albedo
    if args.l2_folder:
        ndvi, albedo, _ = compute_ndvi_albedo_from_l2(args.l2_folder, args.lst)
    else:
        ndvi, ndvi_prof = read_raster(args.ndvi_albedo[0])
        albedo, alb_prof = read_raster(args.ndvi_albedo[1])
        _assert_same_grid(lst_profile, ndvi_prof, label="NDVI")
        _assert_same_grid(lst_profile, alb_prof, label="Albedo")

    # Auto target NDVI if not given: median of vegetated pixels within LST-valid area
    if args.target_ndvi is None:
        valid = np.isfinite(ndvi) & np.isfinite(lst)
        veg = ndvi[valid & (ndvi > 0.3)]
        target_ndvi = float(np.median(veg)) if veg.size else 0.5
    else:
        target_ndvi = float(args.target_ndvi)
    target_albedo = float(args.target_albedo)

    # Fit model on baseline
    model = fit_model(lst, ndvi, albedo, frac=args.sample_frac, seed=args.random_state)
    if args.write_pred_baseline:
        baseline_pred = predict_model(model, ndvi, albedo)
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
    scen_ndvi   = (1.0 - f) * ndvi   + f * target_ndvi
    scen_albedo = (1.0 - f) * albedo + f * target_albedo

    # Predict scenario LST and delta
    scen_pred = predict_model(model, scen_ndvi, scen_albedo)
    if not args.write_pred_baseline:
        baseline_pred = predict_model(model, ndvi, albedo)
    delta = scen_pred - baseline_pred  # °C; negatives are cooling

    # Save rasters
    save_raster(out_dir / "scenario_pred_LST.tif", scen_pred, template_profile)
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
        f"Supersample: {args.supersample}\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()