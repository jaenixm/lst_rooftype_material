# Changelog

All notable package changes are documented here.

## 0.2.0 - 2026-07-10

- Reproject Landsat surface-reflectance bands to the exact baseline LST grid instead of resizing by array shape.
- Convert raster nodata values to `NaN` before modeling.
- Exclude non-finite predictor pixels from partial scenario prediction.
- Validate programmatic and CLI configuration values early.
- Correct `--keep_null_roof`, supersampling transforms, and the CLI supersample default.
- Calculate roof areas in a metre-based CRS and mark selected buildings in the output layer.
- Include train/test metrics and full scenario settings in output reports.
- Add synthetic unit and end-to-end tests.
- Add repository ignore rules and a reproducible setup and usage guide.

## 0.1.0

- Initial reusable package and command-line tools for Landsat green-roof scenarios, OSM preparation, parameter estimation, and CityGML slope enrichment.
