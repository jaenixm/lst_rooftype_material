from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from green_roof_scenario.preparation import iter_geojson_properties, validate_material_arrays


class PreparationTests(unittest.TestCase):
    def test_geojson_properties_are_streamed_in_order(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"gml_id": "a", "material_cov": [1]}, "geometry": None},
                {"type": "Feature", "properties": {"gml_id": "b", "material_cov": [0.6, 0.4]}, "geometry": None},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.geojson"
            path.write_text(json.dumps(payload))
            self.assertEqual([row["gml_id"] for row in iter_geojson_properties(path)], ["a", "b"])

    def test_validation_reports_scalar_multi_and_no_ties(self):
        gdf = gpd.GeoDataFrame(
            {
                "predicted_roof_materials": ["0", "4,1"],
                "material_cov": ["1", "0.6,0.4"],
            },
            geometry=[Point(0, 0), Point(1, 0)],
        )
        self.assertEqual(
            validate_material_arrays(gdf),
            {"records": 2, "scalar_records": 1, "multi_records": 1, "ties": 0},
        )

    def test_validation_rejects_ties(self):
        gdf = gpd.GeoDataFrame(
            {"predicted_roof_materials": ["0,4"], "material_cov": ["0.5,0.5"]},
            geometry=[Point(0, 0)],
        )
        with self.assertRaisesRegex(ValueError, "ties"):
            validate_material_arrays(gdf)


if __name__ == "__main__":
    unittest.main()
