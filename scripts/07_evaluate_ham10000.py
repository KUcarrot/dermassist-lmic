"""Evaluate on HAM10000 test split (in-distribution)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.evaluation.ham10000 import main

if __name__ == "__main__":
    main()
