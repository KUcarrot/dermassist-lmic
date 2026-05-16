"""Generate 5,000 LMIC-specialized training samples for Gemma 4.

Outputs JSONL file. Pre-generated dataset is available at:
  https://huggingface.co/datasets/KUcarrot/dermassist-lmic-training-data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.data.synthetic_data import main

if __name__ == "__main__":
    main()
