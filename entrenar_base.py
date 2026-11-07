"""Punto de entrada: python entrenar_base.py piloto|cola ... (ver src/wdsd_e1/entrenar.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wdsd_e1.entrenar import main  # noqa: E402

if __name__ == "__main__":
    main()
