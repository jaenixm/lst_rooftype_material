"""Configuration dataclasses for running scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["ScenarioConfig"]


@dataclass(slots=True)
class ScenarioConfig:
    l2_folder: Path
    buildings: Path
    roof_field: str
    roof_types: str
    out_dir: Path = Path("results_greening")
    lst: Optional[Path] = None
    build_lst: bool = True
    lst_unit: str = "celsius"
    keep_lst_water: bool = False
    layer: Optional[str] = None
    target_ndvi: Optional[float] = 0.4
    target_albedo: float = 0.20
    sample_frac: float = 0.1
    min_sample_spacing: float = 100.0
    random_state: int = 42
    model: str = "rf"
    supersample: int = 4
    all_touched: bool = False
    write_pred_baseline: bool = False
    keep_null_roof: bool = False
    write_roof_fraction_raster: bool = False
    log_level: str = "INFO"
    min_roof_area: float = 0.0

    def __post_init__(self) -> None:
        self.l2_folder = Path(self.l2_folder)
        self.buildings = Path(self.buildings)
        self.out_dir = Path(self.out_dir)
        if self.lst is not None:
            self.lst = Path(self.lst)
