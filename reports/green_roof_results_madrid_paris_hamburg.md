# Green roof scenario results for Paris, Madrid, Hamburg and Munich

This short report summarizes the modeled land-surface-temperature (LST) effect of replacing selected roof materials with green-roof parameter values in Paris, Madrid, Hamburg and Munich. The analysis used Landsat Collection 2 Level-2 data from `/Users/jaenix/Documents/lst_rooftype_material/data/raw/landsat` and Random Forest scenario models with NDVI, NDBI and albedo predictors. Positive cooling values mean `baseline LST - scenario LST`, in degrees Celsius.

## Landsat scene and weather context

Weather values are hourly Open-Meteo historical reanalysis values for the nearest local hour to the Landsat scene center time. The Landsat cloud cover and sun elevation are read from the local MTL metadata.

| City | Landsat product | Acquisition date and time | Nearest weather hour | Weather at nearest hour | Landsat cloud cover | Sun elevation |
|---|---|---:|---:|---|---:|---:|
| Paris | `LC09_L2SP_199026_20240810_20240811_02_T1` | 2024-08-10 10:40 UTC / 12:40 CEST | 13:00 CEST | 24.4 °C, 55% RH, 0.0 mm precipitation, 42% cloud cover, 4.8 km/h wind | 6.16% | 53.08° |
| Madrid | `LC09_L2SP_201032_20240723_20240724_02_T1` | 2024-07-23 10:55 UTC / 12:55 CEST | 13:00 CEST | 33.5 °C, 22% RH, 0.0 mm precipitation, 0% cloud cover, 5.8 km/h wind | 0.00% | 62.42° |
| Hamburg | `LC09_L2SP_196023_20250621_20250622_02_T1` | 2025-06-21 10:20 UTC / 12:20 CEST | 12:00 CEST | 24.6 °C, 37% RH, 0.0 mm precipitation, 1% cloud cover, 6.1 km/h wind | 0.04% | 57.82° |
| Munich | `LC09_L2SP_193026_20250702_20250703_02_T1` | 2025-07-02 10:03 UTC / 12:03 CEST | 12:00 CEST | 31.4 °C, 29% RH, 0.0 mm precipitation, 0% cloud cover, 9.3 km/h wind | 0.11% | 60.27° |

## Green-roof reference parameters

The target green-roof parameters were taken from detected/reference green roofs and applied as area-weighted means in the scenario runs.

| City | Valid reference roofs | Target NDVI | Target NDBI | Target albedo |
|---|---:|---:|---:|---:|
| Paris | 37 | 0.4062 | -0.0533 | 0.1459 |
| Madrid | 18 | 0.2843 | -0.0799 | 0.1737 |
| Hamburg | 155 | 0.5099 | -0.1531 | 0.1436 |
| Munich | 286 | 0.4218 | -0.0545 | 0.1532 |

## Scenario results

Paris, Madrid and Munich used roof material classes `0,4`. Hamburg was run twice: once without a roof-slope constraint and once with `roof_slope_mean_deg <= 15`.

| City / scenario | Buildings in analysis | Selected candidate roofs | Candidate roof area | Buildings with cooling > 0.01 °C | Raster mean cooling | Area-weighted cooling, all buildings | Area-weighted cooling, cooled buildings | Max building cooling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paris | 106,391 | 61,347 | 2,103.6 ha | 81,096 (76.2%) | 0.1364 °C | 0.2257 °C | 0.2804 °C | 3.4862 °C |
| Madrid | 121,006 | 42,169 | 2,397.9 ha | 82,225 (68.0%) | 0.1459 °C | 0.6163 °C | 0.8588 °C | 7.5013 °C |
| Hamburg, without slope filter | 194,799 | 73,813 | 3,628.8 ha | 78,181 (40.1%) | 0.0769 °C | 0.7837 °C | 1.2268 °C | 7.5915 °C |
| Hamburg, slope <= 15° | 194,799 | 59,547 | 3,332.3 ha | 68,529 (35.2%) | 0.0733 °C | 0.7541 °C | 1.2415 °C | 7.5915 °C |
| Munich | 128,620 | 2,500 | 531.2 ha | 6,431 (5.0%) | 0.0187 °C | 0.0888 °C | 0.4151 °C | 4.1130 °C |

## Interpretation

Madrid shows the strongest average building-level cooling among the four city runs, with an area-weighted cooling of 0.6163 °C across all buildings and 0.8588 °C among buildings with a detectable cooling response. This likely reflects both the hot, dry acquisition conditions and the strong modeled role of the built-up index: the Madrid Random Forest assigns the highest feature importance to NDBI (0.5417).

Paris has the highest share of buildings with a detectable modeled cooling response, but the cooling magnitude is more moderate. The rerun with the fine-tuned Paris building layer selected 61,347 candidate roofs. The raster-wide mean cooling is 0.1364 °C, while the area-weighted building cooling is 0.2257 °C. The Paris model weights NDBI and NDVI similarly, with feature importances of 0.4316 and 0.4071.

Hamburg has the largest candidate roof area, but moderate raster-wide cooling because the effect is distributed over a larger and more heterogeneous analysis area. Adding the roof-slope filter reduces the candidate set from 73,813 to 59,547 roofs and lowers the number of buildings with detectable cooling by about 9,652. The area-weighted cooling among affected buildings remains slightly higher with the slope filter, indicating that the remaining flatter roofs are still effective scenario targets.

Munich was observed during the hottest weather context in this comparison, but the scenario affects a much smaller share of the classified building stock. Only 2,500 roofs matched material classes `0,4`, giving a low city-wide raster mean cooling of 0.0187 °C and an all-building area-weighted cooling of 0.0888 °C. Among buildings with a detectable response, however, the area-weighted cooling is 0.4151 °C. The Munich Random Forest also assigns the highest feature importance to NDBI (0.4810), followed by NDVI (0.3992).

Overall, the results support the expected pattern that green-roof conversion can reduce modeled roof/building LST, with the largest building-level cooling in Madrid and the strongest slope-screened target refinement in Hamburg. Munich shows that the city-wide effect can remain small even under very hot, cloud-free conditions when the selected candidate roof classes cover only a small part of the building stock. The weather context matters for interpretation: Madrid and Munich were observed during very hot, cloud-free conditions, while Paris had milder air temperature and more modeled hourly cloud cover near the overpass despite low Landsat scene cloud cover.

## Sources

- Local Landsat and scenario outputs: `/Users/jaenix/Documents/lst_rooftype_material/data/raw/landsat`, `/Users/jaenix/Documents/lst_rooftype_material/outputs/scenarios/paris`, `/Users/jaenix/Documents/lst_rooftype_material/outputs/scenarios/madrid`, `/Users/jaenix/Documents/lst_rooftype_material/outputs/scenarios/hamburg`, `/Users/jaenix/Documents/lst_rooftype_material/outputs/scenarios/hamburg_slope`, `/Users/jaenix/Documents/lst_rooftype_material/outputs/scenarios/munich`.
- Local green-roof parameter summaries: `/Users/jaenix/Documents/lst_rooftype_material/data/processed/helpers/*_green_roof_parameter_summary.csv`.
- Historical weather: Open-Meteo Historical Weather API, https://open-meteo.com/en/docs/historical-weather-api.
