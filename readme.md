

# Green Roof Scenario — Empirical Urban Heat Mitigation Modeling

This project contains the script `green_roof_scenario.py`, which simulates how much **land surface temperature (LST)** would decrease if selected building roofs were converted to **green roofs**, using **remote sensing** and a **data-driven regression model**.

## 🎯 Goal

Evaluate the **cooling potential** of green roof interventions **directly from satellite imagery** — fast, spatially explicit, and scientifically backed.

This approach does **not** rely on heavy physical climate models. Instead, it follows the **empirical intervention simulation approach** used in multiple **peer-reviewed urban heat studies**, such as:
- Sánchez-Cordero et al. 2025 (*Remote Sensing*)
- Joshi et al. 2023 (*Springer Urban Intelligence*)
- Calhoun et al. 2024 (*Scientific Reports*)

These studies show that **modifying NDVI and Albedo on rooftops** and re-predicting LST is a scientifically valid method for simulating urban greening scenarios.

## 🔧 How It Works

1. **Input data**
   - Baseline LST raster (°C) from Landsat 8/9 Level-2 Surface Temperature.
   - Building footprints with a roof type attribute.
   - Either **Landsat Level-2 folder** (auto-compute NDVI/Albedo)  
     or **precomputed NDVI + Albedo rasters**.

2. **Fit an empirical model**
   The script learns how **LST depends on NDVI and albedo** in your own city:
   ```
   LST = a + b × NDVI + c × Albedo
   ```

3. **Select roof types to “green”**
   Example:
   ```
   --roof_types "concrete, tar_paper"
   ```

4. **Simulate greening**
   Only pixels **actually containing roofs are modified**. Their NDVI and albedo are **partially replaced with realistic vegetation values** (derived from existing green areas in the same image).

5. **Predict new scenario LST**
   ```
   delta_LST = scenario_predicted_LST − baseline_predicted_LST
   ```
   → **Negative values = cooling effect**

6. **Export results**
   - `delta_LST.tif` → pixel-level cooling map
   - `buildings_greening_impact.gpkg` → per-building mean cooling
   - Optional `roof_fraction.tif` → visualization of roof pixel influence

## ✅ Output Overview

| File | Description |
|------|-------------|
| `scenario_pred_LST.tif` | Predicted LST *after* greening |
| `delta_LST.tif` | LST change (°C) — negative = cooler |
| `roof_fraction.tif` | (optional) roof coverage per pixel |
| `buildings_greening_impact.gpkg` | Each building with mean ΔLST |
| `_greening_provenance.txt` | Documents roof types, NDVI target, parameters |

## 📚 Why This Is Scientifically Valid

This method is directly aligned with **recent remote sensing literature**, which uses **remote sensing + empirical regression models** for green roof simulations — instead of heavy simulation tools like ENVI-met.

It is:
- ✅ **Fast and scalable**
- ✅ **Fully satellite-based**
- ✅ **Quantitative and explainable**
- ✅ **Defensible for urban planning and policy**

## 👤 Author / Contact

Your name / institution / email here.