from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from green_roof_scenario.config import ScenarioConfig
from green_roof_scenario.provenance import environment_versions, sha256_path
from green_roof_scenario.scenario import run_scenario


class FakeModel:
    feature_importances_ = np.asarray([0.4, 0.2, 0.4])


class ProvenanceTests(unittest.TestCase):
    def test_sha256_file_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("abc")
            record = sha256_path(str(path.resolve()))
        self.assertEqual(record["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertIn("python", environment_versions())

    def test_scenario_writes_auditable_json_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buildings_path = root / "buildings.gpkg"
            buildings_path.write_bytes(b"input")
            l2_folder = root / "l2"
            l2_folder.mkdir()
            (l2_folder / "band.tif").write_bytes(b"band")
            output_dir = root / "out"
            gdf = gpd.GeoDataFrame(
                {
                    "predicted_roof_materials": ["0"],
                    "material_cov": ["1"],
                    "slope": [5.0],
                },
                geometry=[box(0, 0, 2, 2)],
                crs="EPSG:3857",
            )
            array = np.ones((2, 2), dtype="float32") * 30
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:3857",
                "transform": Affine(1, 0, 0, 0, -1, 2),
                "nodata": np.nan,
            }

            def fake_save(path, *_args, **_kwargs):
                Path(path).write_bytes(b"raster")

            config = ScenarioConfig(
                l2_folder=l2_folder,
                buildings=buildings_path,
                roof_material_field="predicted_roof_materials",
                roof_materials_type="0,4",
                roof_material_strategy="exact",
                roof_material_cov_field="material_cov",
                roof_slope_field="slope",
                max_roof_slope_deg=15,
                out_dir=output_dir,
                model="rf",
            )
            metrics = {"r2_train": 0.8, "r2_test": 0.7, "rmse_train": 1.0, "rmse_test": 1.2}
            with (
                patch("green_roof_scenario.scenario.gpd.read_file", return_value=gdf),
                patch("green_roof_scenario.scenario.build_lst_from_l2", return_value=(output_dir / "baseline_LST.tif", array, profile)),
                patch("green_roof_scenario.scenario.compute_ndvi_albedo_from_l2", return_value=(array / 100, array / 100, array / 100, profile)),
                patch("green_roof_scenario.scenario._clip_raster_to_boundary", side_effect=lambda values, prof, _geoms: (values, prof)),
                patch("green_roof_scenario.scenario.fit_model", return_value=(FakeModel(), metrics)),
                patch("green_roof_scenario.scenario.predict_model", return_value=np.zeros((2, 2), dtype="float32")),
                patch("green_roof_scenario.scenario.predict_partial", return_value=-np.ones((2, 2), dtype="float32")),
                patch("green_roof_scenario.scenario.roof_mask_fraction", return_value=np.ones((2, 2), dtype="float32")),
                patch("green_roof_scenario.scenario.save_raster", side_effect=fake_save),
                patch(
                    "green_roof_scenario.scenario.zonal_stats",
                    side_effect=[[{"mean": 30.0}], [{"mean": 29.0}], [{"mean": -1.0}]],
                ),
            ):
                outputs = run_scenario(config)

            provenance = json.loads(Path(outputs.provenance).read_text())
            self.assertEqual(provenance["parameters"]["roof_material_strategy"], "exact")
            self.assertEqual(provenance["parameters"]["roof_material_cov_field"], "material_cov")
            self.assertEqual(provenance["counts"]["buildings_analyzed"], 1)
            self.assertEqual(provenance["counts"]["buildings_selected"], 1)
            self.assertEqual(provenance["model_diagnostics"], metrics)
            self.assertIn("sha256", provenance["inputs"]["buildings"])
            self.assertIn("git_commit", provenance)
            self.assertIn("environment", provenance)


if __name__ == "__main__":
    unittest.main()
