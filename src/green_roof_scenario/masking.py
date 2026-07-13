"""Vector/raster masking helpers."""

from __future__ import annotations

import ast
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

__all__ = ["parse_array_value", "roof_mask_fraction", "select_material_value", "subset_buildings"]


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
    roof_material_field: str | None,
    roof_materials_type: str | None,
    *,
    roof_shape_field: str | None = None,
    roof_shape_type: str | None = None,
    roof_slope_field: str | None = None,
    max_roof_slope_deg: float | None = None,
    keep_null_roof: bool = False,
    roof_material_strategy: str = "legacy",
    roof_material_cov_field: str = "material_cov",
) -> gpd.GeoDataFrame:
    material_filter = bool(roof_materials_type and roof_materials_type.strip())
    shape_filter = bool(roof_shape_type and roof_shape_type.strip())
    slope_filter = max_roof_slope_deg is not None

    if not material_filter and not shape_filter and not slope_filter:
        raise ValueError("Provide at least one roof material, shape, or slope filter to select roofs.")

    df = gdf.copy()
    mask = np.ones(len(df), dtype=bool)

    if material_filter:
        if not roof_material_field:
            raise ValueError("roof_material_field must be provided when roof_materials_type is set.")
        if roof_material_field not in df.columns:
            raise KeyError(f"Roof material field '{roof_material_field}' not in buildings.")
        if roof_material_strategy == "legacy":
            # Intentionally retain the historical parser byte-for-byte for legacy runs.
            normalized_mat = df[roof_material_field].apply(_normalize_roof_value)
        elif roof_material_strategy in {"exact", "dominant"}:
            if roof_material_cov_field not in df.columns:
                raise KeyError(f"Roof material coverage field '{roof_material_cov_field}' not in buildings.")
            selected_values: list[str | None] = []
            errors: list[str] = []
            for position, (index, row) in enumerate(df.iterrows()):
                coverage = row[roof_material_cov_field]
                try:
                    selected_values.append(
                        select_material_value(
                            row[roof_material_field],
                            coverage,
                            strategy=roof_material_strategy,
                        )
                    )
                except ValueError as exc:
                    if len(errors) < 10:
                        errors.append(f"index={index!r} (position={position}): {exc}")
            if errors:
                suffix = "" if len(errors) < 10 else " (showing first 10)"
                raise ValueError(
                    f"Malformed roof material records for strategy '{roof_material_strategy}'{suffix}: "
                    + "; ".join(errors)
                )
            normalized_mat = pd.Series(selected_values, index=df.index, dtype="object")
        else:
            raise ValueError("roof_material_strategy must be one of: legacy, exact, dominant")
        wanted_mat = {t.strip().lower() for t in roof_materials_type.split(",") if t.strip()}
        mat_mask = normalized_mat.isin(wanted_mat)
        if not keep_null_roof:
            mat_mask &= normalized_mat.notna()
        mask &= mat_mask

    if shape_filter:
        if not roof_shape_field:
            raise ValueError("roof_shape_field must be provided when roof_shape_type is set.")
        if roof_shape_field not in df.columns:
            raise KeyError(f"Roof shape field '{roof_shape_field}' not in buildings.")
        normalized_shape = df[roof_shape_field].apply(_normalize_roof_value)
        wanted_shape = {t.strip().lower() for t in roof_shape_type.split(",") if t.strip()}
        shape_mask = normalized_shape.isin(wanted_shape)
        if not keep_null_roof:
            shape_mask &= normalized_shape.notna()
        mask &= shape_mask

    if slope_filter:
        if not roof_slope_field:
            raise ValueError("roof_slope_field must be provided when max_roof_slope_deg is set.")
        if roof_slope_field not in df.columns:
            raise KeyError(f"Roof slope field '{roof_slope_field}' not in buildings.")
        normalized_slope = df[roof_slope_field].apply(_normalize_numeric_value)
        slope_mask = normalized_slope.notna() & (normalized_slope <= float(max_roof_slope_deg))
        mask &= slope_mask

    return df.loc[mask].copy()


def parse_array_value(value: object, *, field_name: str) -> list[object]:
    """Parse an array-like property while preserving scalar values as one element.

    Native sequences, JSON/Python literals, comma-separated strings, and scalars are
    accepted. Empty and null values are invalid because exact/dominant selection must
    be auditable rather than silently guessing.
    """

    if isinstance(value, np.ndarray):
        parsed = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} is empty")
        if stripped.startswith("[") or stripped.startswith("("):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    decoded = ast.literal_eval(stripped)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"{field_name} is not a valid serialized array") from exc
            if not isinstance(decoded, (list, tuple, np.ndarray)):
                raise ValueError(f"{field_name} serialized value is not an array")
            parsed = list(decoded)
        elif "," in stripped:
            parsed = [item.strip() for item in stripped.split(",")]
        else:
            parsed = [stripped]
    else:
        try:
            is_null = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_null = False
        if is_null:
            raise ValueError(f"{field_name} is null")
        parsed = [value]

    if not parsed:
        raise ValueError(f"{field_name} is empty")
    for item in parsed:
        if isinstance(item, str) and not item.strip():
            raise ValueError(f"{field_name} contains an empty element")
        try:
            is_null = bool(pd.isna(item))
        except (TypeError, ValueError):
            is_null = False
        if is_null:
            raise ValueError(f"{field_name} contains a null element")
    return parsed


def select_material_value(
    material_value: object,
    coverage_value: object | None,
    *,
    strategy: str,
) -> str | None:
    """Return the exact or dominant normalized material for one building record."""

    materials = parse_array_value(material_value, field_name="roof material")
    normalized = [str(item).strip().lower() for item in materials]
    if strategy not in {"exact", "dominant"}:
        raise ValueError("select_material_value supports only exact or dominant strategies")

    coverages_raw = parse_array_value(coverage_value, field_name="material coverage")
    if len(materials) != len(coverages_raw):
        raise ValueError(
            "roof material and material coverage lengths differ "
            f"({len(materials)} != {len(coverages_raw)})"
        )
    try:
        coverages = np.asarray([float(item) for item in coverages_raw], dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("material coverage contains a non-numeric value") from exc
    if not np.all(np.isfinite(coverages)):
        raise ValueError("material coverage contains a non-finite value")
    if strategy == "exact":
        return normalized[0] if len(normalized) == 1 else None
    return normalized[int(np.argmax(coverages))]


def _normalize_roof_value(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return _normalize_roof_value(parsed)
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def _normalize_numeric_value(value: object) -> float | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return _normalize_numeric_value(parsed)
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
