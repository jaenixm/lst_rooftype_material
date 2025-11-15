"""Command-line interface for the green roof scenario package."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config import ScenarioConfig
from .scenario import run_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greened roof LST scenario (empirical model).")
    parser.add_argument("--lst", default=None, help="Baseline LST raster (°C), aligned to Landsat grid.")
    parser.add_argument(
        "--build_lst",
        action="store_true",
        help="Build the baseline LST raster directly from the provided Landsat L2 folder.",
    )
    parser.add_argument(
        "--lst_unit",
        choices=["celsius", "kelvin"],
        default="celsius",
        help="Unit for any LST raster built from the L2 folder (default Celsius).",
    )
    parser.add_argument(
        "--keep_lst_water",
        action="store_true",
        help="When building LST internally, keep QA water pixels instead of masking them.",
    )
    parser.add_argument("--buildings", required=True, help="Buildings file (GPKG/GeoJSON/shp).")
    parser.add_argument("--layer", default=None, help="Optional layer name inside GPKG.")
    parser.add_argument(
        "--roof_field",
        default="predictedrooftypematerial",
        help="Roof type field in the buildings layer.",
    )
    parser.add_argument(
        "--roof_types",
        required=True,
        help="Comma-separated roof types to convert to green (e.g., 'concrete,bitumen').",
    )
    parser.add_argument("--out_dir", default="results_greening", help="Output folder.")
    parser.add_argument(
        "--l2_folder",
        required=True,
        help="Path to Landsat L2 scene folder (used to compute NDVI, Albedo, and NDBI).",
    )
    parser.add_argument(
        "--target_ndvi",
        type=float,
        default=0.4,
        help="Target NDVI for green roofs (default: median of vegetated pixels).",
    )
    parser.add_argument(
        "--target_albedo",
        type=float,
        default=0.20,
        help="Target broadband albedo for green roofs (default: 0.20).",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.1,
        help="Random sample fraction of valid pixels for model fitting (default: 0.1).",
    )
    parser.add_argument(
        "--min_sample_spacing",
        type=float,
        default=100.0,
        help="Approximate minimum spacing between training samples in meters (0 disables thinning).",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model",
        choices=["linear", "rf"],
        default="rf",
        help="Model type: 'rf' (default) or 'linear' (Linear Regression).",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="Supersampling factor to estimate roof fraction per pixel (default: 4).",
    )
    parser.add_argument(
        "--all_touched",
        action="store_true",
        help="Use all_touched=True for coarse roof mask (fast, no fractions).",
    )
    parser.add_argument(
        "--write_pred_baseline",
        action="store_true",
        help="Write model baseline prediction raster to the output folder.",
    )
    parser.add_argument(
        "--keep_null_roof",
        action="store_true",
        help="Keep features with NULL roof field instead of dropping them.",
    )
    parser.add_argument(
        "--write_roof_fraction_raster",
        action="store_true",
        help="Write the per-pixel roof fraction raster (0..1) to the output folder.",
    )
    parser.add_argument(
        "--min_roof_area",
        type=float,
        default=0.0,
        help="Minimum roof area in square meters to consider for greening (default: 0).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ScenarioConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ScenarioConfig(
        l2_folder=args.l2_folder,
        buildings=args.buildings,
        roof_field=args.roof_field,
        roof_types=args.roof_types,
        out_dir=args.out_dir,
        lst=args.lst,
        build_lst=args.build_lst,
        lst_unit=args.lst_unit,
        keep_lst_water=args.keep_lst_water,
        layer=args.layer,
        target_ndvi=args.target_ndvi,
        target_albedo=args.target_albedo,
        sample_frac=args.sample_frac,
        min_sample_spacing=args.min_sample_spacing,
        random_state=args.random_state,
        model=args.model,
        supersample=args.supersample,
        all_touched=args.all_touched,
        write_pred_baseline=args.write_pred_baseline,
        keep_null_roof=args.keep_null_roof,
        write_roof_fraction_raster=args.write_roof_fraction_raster,
        log_level=args.log_level,
        min_roof_area=args.min_roof_area,
    )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    log_level = getattr(logging, config.log_level)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_scenario(config)


if __name__ == "__main__":  # pragma: no cover
    main()
