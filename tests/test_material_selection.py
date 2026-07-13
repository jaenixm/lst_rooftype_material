from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from green_roof_scenario.cli import parse_args
from green_roof_scenario.masking import parse_array_value, select_material_value, subset_buildings


def frame(materials, coverages):
    return gpd.GeoDataFrame(
        {
            "predicted_roof_materials": materials,
            "material_cov": coverages,
            "slope": [5.0] * len(materials),
        },
        geometry=[Point(i, 0) for i in range(len(materials))],
        crs="EPSG:3857",
    )


class ParseArrayValueTests(unittest.TestCase):
    def test_native_serialized_comma_and_scalar_inputs(self):
        cases = [
            (np.asarray([0, 4]), [0, 4]),
            ([0, 4], [0, 4]),
            ((0, 4), [0, 4]),
            ('[0, "4"]', [0, "4"]),
            ("(0, 4)", [0, 4]),
            ("0,4", ["0", "4"]),
            (4, [4]),
            ("4", ["4"]),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_array_value(value, field_name="test"), expected)

    def test_invalid_empty_arrays(self):
        for value in ([], np.asarray([]), "", "[]", "0,"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_array_value(value, field_name="test")


class MaterialStrategyTests(unittest.TestCase):
    def test_exact_excludes_multi_material_records(self):
        result = subset_buildings(
            frame(["0", "0,4", "4"], ["1", "0.7,0.3", "1"]),
            "predicted_roof_materials",
            "0,4",
            roof_material_strategy="exact",
            roof_material_cov_field="material_cov",
        )
        self.assertEqual(result.index.tolist(), [0, 2])

    def test_dominant_uses_true_argmax(self):
        self.assertEqual(select_material_value("4,1", "0.2,0.8", strategy="dominant"), "1")

    def test_dominant_tie_uses_first_maximum(self):
        self.assertEqual(select_material_value("4,0,1", "0.5,0.5,0", strategy="dominant"), "4")

    def test_invalid_mismatch_and_nonfinite_fail(self):
        for materials, coverage in [("0,4", "1"), ("0,4", "nan,1"), ("0", "inf")]:
            with self.subTest(materials=materials, coverage=coverage), self.assertRaises(ValueError):
                select_material_value(materials, coverage, strategy="dominant")

    def test_exact_also_validates_coverage(self):
        with self.assertRaises(ValueError):
            select_material_value("0", "nan", strategy="exact")

    def test_legacy_behavior_is_unchanged(self):
        result = subset_buildings(
            frame(["0,4", "[0, 4]"], ["0.5,0.5", "[0.5,0.5]"]),
            "predicted_roof_materials",
            "0",
            roof_material_strategy="legacy",
        )
        self.assertEqual(result.index.tolist(), [1])


class CliTests(unittest.TestCase):
    def test_material_strategy_and_coverage_field_reach_config(self):
        config = parse_args(
            [
                "--l2_folder",
                "scene",
                "--buildings",
                "buildings.gpkg",
                "--roof_materials_type",
                "0,4",
                "--roof_material_strategy",
                "dominant",
                "--roof_material_cov_field",
                "cov",
            ]
        )
        self.assertEqual(config.roof_material_strategy, "dominant")
        self.assertEqual(config.roof_material_cov_field, "cov")

    def test_default_strategy_is_legacy(self):
        config = parse_args(
            ["--l2_folder", "scene", "--buildings", "buildings.gpkg", "--roof_materials_type", "0"]
        )
        self.assertEqual(config.roof_material_strategy, "legacy")


if __name__ == "__main__":
    unittest.main()
