"""Common locations for false-positive split generators and saved split files."""

from pathlib import Path


FP_ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
GENERATED_SPLITS_ROOT = FP_ANALYSIS_ROOT / "splits" / "generated"

HCRL_SPLITS_DIR = GENERATED_SPLITS_ROOT / "hcrl"
SONATA_SPLITS_DIR = GENERATED_SPLITS_ROOT / "sonata"
OBD_SPLITS_DIR = GENERATED_SPLITS_ROOT / "obd"

HCRL_SPLITS_JSON = HCRL_SPLITS_DIR / "splits_config.json"
SONATA_SPLITS_JSON = SONATA_SPLITS_DIR / "splits_config.json"
OBD_SPLITS_JSON = OBD_SPLITS_DIR / "splits_obd.json"
