"""Vector/raster masking helpers."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

__all__ = ["roof_mask_fraction", "subset_buildings"]


def roof_mask_fraction(
    buildings_gdf: gpd.GeoDataFrame,
    template_profile: dict,
    *,
    supersample: int = 4,
    all_touched: bool = False,
) -> np.ndarray:
    shapes = [mapping(geom) for geom in buildings_gdf.geometry.values]
    height, width = template_profile["height"], template_profile["width"]
    transform = template_profile["transform"]

    if all_touched or supersample <= 1:
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=True,
        ).astype("float32")
        return mask

    hss, wss = height * supersample, width * supersample
    a, b, c, d, e, f = transform[:6]
    up_transform = rasterio.Affine(a / supersample, b, c, d, e / supersample, f)
    up = rasterize(
        shapes,
        out_shape=(hss, wss),
        transform=up_transform,
        fill=0,
        default_value=1,
        all_touched=True,
    ).astype("float32")
    up = up.reshape(height, supersample, width, supersample).mean(axis=(1, 3))
    return up


def subset_buildings(
    gdf: gpd.GeoDataFrame,
    roof_field: str,
    roof_types: str,
    *,
    keep_null_roof: bool = False,
) -> gpd.GeoDataFrame:
    if roof_field not in gdf.columns:
        raise KeyError(f"Roof field '{roof_field}' not in buildings.")
    df = gdf.copy()
    if not keep_null_roof:
        df = df.dropna(subset=[roof_field]).copy()
    wanted = {t.strip().lower() for t in roof_types.split(",")}
    sel = df[roof_field].str.lower().isin(wanted)
    return df.loc[sel].copy()
