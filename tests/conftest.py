"""Make the flat src/ modules importable (scripts are run as `python src/<script>.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
