

# Green Roof Scenario — Empirical Urban Heat Mitigation Modeling

This repository now ships a reusable Python package (`green_roof_scenario`) plus a thin compatibility script `green_roof_scenario.py`. The package simulates how much **land surface temperature (LST)** would decrease if selected building roofs were converted to **green roofs**, using **remote sensing** and a **data-driven regression model**.

## Installation & CLI Usage

```bash
python -m pip install -e .            # install locally (PEP 517/518 via pyproject)
green-roof-scenario --help            # show CLI options
green-roof-scenario \
  --l2_folder data/LC09_L2SP_196023_20250621_20250622_02_T1 \
  --buildings data/building_footprint.gpkg \
  --roof_field predictedroofmaterials \
  --roof_types "concrete, tar_paper" \
  --out_dir results_greening_demo \
  --build_lst \
  --model rf \
  --min_roof_area 100     

```

Random Forest (`--model rf`) is the default choice per the methodology outlined in
`Model_green_roof_effect.pdf`, but `--model linear` remains available for deterministic fits.

When `--build_lst` is set and no `--lst` path is provided, the baseline raster is
written to `<out_dir>/baseline_LST.tif` alongside the other outputs.

Programmatic use is also supported:

```python
from green_roof_scenario import ScenarioConfig, run_scenario

config = ScenarioConfig(
    l2_folder="data/LC09_L2SP_196023_20250621_20250622_02_T1",
    buildings="results/buildings_with_lst.gpkg",
    roof_field="predictedroofmaterials",
    roof_types="concrete, tar_paper",
    out_dir="results_greening_demo",
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

3. **Select roof types to “green”**
   Example:
   ```
   --roof_types "concrete, tar_paper"
   ```

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

The file `Model_green_roof_effect.pdf` summarizes the scientific basis that guides this package.
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
