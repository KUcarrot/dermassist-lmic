"""Evaluate on BCN20000 (external validation).

Requires BCN20000 dataset from ISIC Archive:
  https://api.isic-archive.com/collections/249/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.evaluation.bcn20000 import main

if __name__ == "__main__":
    main()
