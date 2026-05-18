

# Green Roof Scenario — Empirical Urban Heat Mitigation Modeling

This repository ships a reusable Python package (`green_roof_scenario`) for simulating how much **land surface temperature (LST)** would decrease if selected building roofs were converted to **green roofs**, using **remote sensing** and a **data-driven regression model**.

Local geospatial inputs and generated scenario outputs are intentionally ignored by Git:

| Path | Purpose |
|------|---------|
| `data/raw/landsat/` | Landsat Collection 2 Level-2 scene folders |
| `data/raw/citygml/` | CityGML / LoD2 source data |
| `data/raw/orthophotos/` | Local orthophoto rasters and sidecars |
| `data/processed/buildings/` | Prepared building layers |
| `data/processed/helpers/` | Green-roof references, boundaries, and parameter summaries |
| `outputs/scenarios/` | Scenario rasters, GeoPackages, provenance, and model summaries |

## Installation & CLI Usage

```bash
python -m pip install -e .            # install locally (PEP 517/518 via pyproject)
green-roof-scenario --help            # show CLI options
green-roof-scenario \
  --l2_folder data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --buildings data/processed/buildings/hamburg-cls-finale-with-slopes.gpkg \
  --layer buildings \
  --roof_material_field predicted_roof_materials \
  --roof_materials_type "0,4" \
  --roof_slope_field roof_slope_mean_deg \
  --max_roof_slope_deg 15 \
  --out_dir outputs/scenarios/hamburg_slope \
  --build_lst \
  --model rf \
  --target_ndvi 0.5098841067653748 \
  --target_ndbi -0.15313389460307741 \
  --target_albedo 0.1435968737428123 \
  --clip_positive_delta \
  --write_indices_rasters
      

```

If `--boundary` is omitted, the scenario clips the analysis to the extent of the
input building layer. Pass `--boundary path/to/boundary.gpkg` only when you want a
specific administrative or study-area boundary instead.

## Preparation Utilities

Fetch already tagged green roofs from OSM for a city boundary:

```bash
green-roof-fetch-osm \
  --city "Madrid, Spain" \
  --out data/processed/helpers/madrid_green_roofs.gpkg \
  --boundary-out data/processed/helpers/madrid_boundary.gpkg \
  --target-crs EPSG:25830
```

Or use the extent of the building layer as the Overpass search area, which keeps
the green-roof fetch aligned to the exact analysis footprint:

```bash
green-roof-fetch-osm \
  --buildings data/processed/buildings/paris-cls-finale.gpkg \
  --buildings-layer buildings \
  --out data/processed/helpers/paris_green_roofs.gpkg \
  --boundary-out data/processed/helpers/paris_building_extent.gpkg \
  --target-crs EPSG:2154
```

If Nominatim resolves the wrong object, pass the administrative relation directly:

```bash
green-roof-fetch-osm --osm-relation-id 5326784 --out data/processed/helpers/madrid_green_roofs.gpkg
```

Compute mean NDVI, NDBI, and albedo for known green roofs, plus unweighted and
area-weighted target values:

```bash
green-roof-params \
  --green-roofs data/processed/helpers/madrid_green_roofs.gpkg \
  --l2-folder data/raw/landsat/LC08_or_LC09_SCENE_FOLDER \
  --indices-out-dir outputs/scenarios/results_madrid \
  --out data/processed/helpers/madrid_green_roofs_with_params.gpkg \
  --summary-csv data/processed/helpers/madrid_green_roof_parameter_summary.csv
```

If `outputs/scenarios/results_madrid/ndvi.tif`, `outputs/scenarios/results_madrid/ndbi.tif`, and
`outputs/scenarios/results_madrid/albedo.tif` already exist, you can pass them explicitly instead:

```bash
green-roof-params \
  --green-roofs data/processed/helpers/madrid_green_roofs.gpkg \
  --ndvi outputs/scenarios/results_madrid/ndvi.tif \
  --ndbi outputs/scenarios/results_madrid/ndbi.tif \
  --albedo outputs/scenarios/results_madrid/albedo.tif \
  --out data/processed/helpers/madrid_green_roofs_with_params.gpkg \
  --summary-csv data/processed/helpers/madrid_green_roof_parameter_summary.csv
```

Use the `area_weighted_mean` values from the summary CSV as conservative green-roof
targets for `--target_ndvi`, `--target_ndbi`, and `--target_albedo`.

Enrich a CityGML-derived building layer with roof slope metrics. The main field,
`roof_slope_mean_deg`, is the area-weighted mean slope across CityGML roof planes:

```bash
python -m green_roof_scenario.citygml_slopes \
  --buildings data/processed/buildings/hamburg-cls-finale.gpkg \
  --layer buildings \
  --citygml-dir data/raw/citygml/hamburg_lod2 \
  --out data/processed/buildings/hamburg-cls-finale-with-slopes.gpkg
```

Then restrict greening to roofs with an area-weighted mean slope up to 15 degrees:

```bash
green-roof-scenario \
  --buildings data/processed/buildings/hamburg-cls-finale-with-slopes.gpkg \
  --layer buildings \
  --max_roof_slope_deg 15 \
  ...
```

Random Forest (`--model rf`) is the default choice per the methodology outlined in
`docs/methodology/Model_green_roof_effect.pdf`, but `--model linear` remains available for deterministic fits.

When `--build_lst` is set and no `--lst` path is provided, the baseline raster is
written to `<out_dir>/baseline_LST.tif` alongside the other outputs.

Programmatic use is also supported:

```python
from green_roof_scenario import ScenarioConfig, run_scenario

config = ScenarioConfig(
    l2_folder="data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg",
    buildings="data/processed/buildings/hamburg-cls-finale.gpkg",
    roof_material_field="predictedroofmaterials",
    roof_materials_type="concrete, tar_paper",
    roof_shape_field="roofshape",
    roof_shape_type="flat",
    out_dir="outputs/scenarios/results_greening_demo",
    build_lst=True,
    target_ndvi=0.4,
    model="rf",
)
run_scenario(config)
```

### Source layout

The package follows a modern `src/` layout:

| Module | Responsibility |
|--------|----------------|
| `green_roof_scenario.cli` | Argparse-based CLI entry point |
| `green_roof_scenario.config` | Dataclasses for scenario configuration |
| `green_roof_scenario.l2` | Landsat L2 helpers (LST build, NDVI/albedo) |
| `green_roof_scenario.modeling` | Sampling, regression fitting, prediction |
| `green_roof_scenario.masking` | Building filtering and roof fraction rasters |
| `green_roof_scenario.scenario` | High-level orchestration + outputs |
| `green_roof_scenario.io` | Raster IO helpers |
| `green_roof_scenario.citygml_slopes` | CityGML roof slope enrichment utility |

## Goal

Evaluate the **cooling potential** of green roof interventions **directly from satellite imagery** — fast, spatially explicit, and scientifically backed.

This approach does **not** rely on heavy physical climate models. Instead, it follows the **empirical intervention simulation approach** used in multiple **peer-reviewed urban heat studies**, such as:
- Sánchez-Cordero et al. 2025 (*Remote Sensing*)
- Joshi et al. 2023 (*Springer Urban Intelligence*)
- Calhoun et al. 2024 (*Scientific Reports*)

These studies show that **modifying NDVI and Albedo on rooftops** and re-predicting LST is a scientifically valid method for simulating urban greening scenarios.

## How It Works

1. **Input data**
   - Baseline LST raster (°C) from Landsat 8/9 Level-2 Surface Temperature.
   - Building footprints with a roof type attribute.
   - Either **Landsat Level-2 folder** (auto-compute NDVI/Albedo)  
     or **precomputed NDVI + Albedo rasters**.

2. **Fit an empirical model**
   By default the package trains a **Random Forest regressor** that maps NDVI, broadband
   albedo, and NDBI to the observed LST. A linear model is still available via `--model linear`
   if you need a simple parametric form, but the RF baseline is recommended for green-roof
   assessments.

3. **Select roof filters to “green”**
   Example (material only):
   ```
   --roof_materials_type "concrete, tar_paper"
   ```
   Example (material AND shape):
   ```
   --roof_materials_type "concrete, tar_paper" --roof_shape_type "flat,low_slope"
   ```
   Example (material AND roof slope up to 15 degrees):
   ```
   --roof_materials_type "concrete, tar_paper" --max_roof_slope_deg 15
   ```
   At least one of `--roof_materials_type`, `--roof_shape_type`, or `--max_roof_slope_deg` is required; if multiple are set the filters are ANDed. `--max_roof_slope_deg` reads `roof_slope_mean_deg` by default; use `--roof_slope_field` for another numeric slope column.

4. **Simulate greening**
   Only pixels **actually containing roofs are modified**. Their NDVI and albedo are **partially replaced with realistic vegetation values** (derived from existing green areas in the same image).

5. **Predict new scenario LST**
   ```
   delta_LST = scenario_predicted_LST - baseline_observed_LST
   ```
   → **Negative values = cooling effect**
   - Optional safeguard: `--clip_positive_delta` (cooling-only).

6. **Export results**
   - `delta_LST.tif` → pixel-level cooling map
   - `buildings_greening_impact.gpkg` → per-building mean cooling
   - Optional `ndvi.tif`, `albedo.tif`, `ndbi.tif` → input indices aligned to LST (enable with `--write_indices_rasters`)
   - Optional `roof_fraction.tif` → visualization of roof pixel influence
   - `model_feature_importance.txt` → feature importances (RF) or coefficients (linear)

## Output Overview

| File | Description |
|------|-------------|
| `scenario_pred_LST.tif` | Predicted LST *after* greening |
| `delta_LST.tif` | LST change (°C) — negative = cooler |
| `ndvi.tif`, `albedo.tif`, `ndbi.tif` | (optional) input indices aligned to LST |
| `roof_fraction.tif` | (optional) roof coverage per pixel |
| `buildings_greening_impact.gpkg` | Each building with mean ΔLST |
| `_greening_provenance.txt` | Documents roof types, NDVI target, parameters |

## Methodology Reference

The file `docs/methodology/Model_green_roof_effect.pdf` summarizes the scientific basis that guides this package.
Highlights from that document:

- **Predictor set** – we follow Verbeiren et al. 2024 and Martínez-Pérez et al. 2023 by using
  Landsat Collection 2 SR products to derive NDVI, broadband albedo (Liang 2000/2001
  coefficients), and NDBI as predictors.
- **Observed target** – Landsat ST_B10 (C2 L2) provides the calibrated LST that the model
  predicts; QA_PIXEL masks ensure clouds, shadows, and water are excluded.
- **Spatial sampling** – to limit spatial autocorrelation the training pool is thinned to a
  minimum spacing of ~100 m (≈3–4 pixels) before sampling the requested fraction.
- **Model choice** – Random Forest regression is recommended because it captures the
  non-linear responses of LST to vegetation, albedo, and built-up intensity, with typical
  scene-level performance of R²≈0.6–0.75 / RMSE≈2°C, matching the cited literature.
- **Scenario blending** – rooftop NDVI and albedo are blended toward realistic green roof
  targets (NDVI≈0.4, albedo≈0.20) proportionally to the supersampled roof fraction,
  matching the procedure detailed in the PDF.
