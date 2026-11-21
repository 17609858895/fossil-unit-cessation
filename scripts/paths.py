"""Data locations shared by the analysis commands."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("CEMS_ARCHIVES", ROOT / "data_raw" / "cems"))
WRI_ARCHIVE = Path(os.environ.get("CEMS_WRI_ARCHIVE", ROOT / "data_raw" / "wri" / "globalpowerplantdatabasev130.zip"))
DP = Path(os.environ.get("CEMS_PANEL", ROOT / "data_processed"))
RES = Path(os.environ.get("CEMS_RESULTS", ROOT / "results"))
RES.mkdir(parents=True, exist_ok=True)
