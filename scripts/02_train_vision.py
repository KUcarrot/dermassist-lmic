"""Train EfficientNet-B4 Vision Classifier on HAM10000."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.vision.train import main

if __name__ == "__main__":
    main()
