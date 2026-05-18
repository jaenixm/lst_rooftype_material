"""Compatibility wrapper for the reusable green-roof parameter CLI.

Use:
    green-roof-params --green-roofs data/helpers/hamburg_green_roofs.gpkg \
      --ndvi results_greening_hamburg/ndvi.tif \
      --ndbi results_greening_hamburg/ndbi.tif \
      --albedo results_greening_hamburg/albedo.tif \
      --out data/helpers/hamburg_green_roofs_with_params.gpkg
"""

from pathlib import Path
import sys

repo_src = Path(__file__).resolve().parents[1] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

from green_roof_scenario.green_roof_params import main


if __name__ == "__main__":
    main()
