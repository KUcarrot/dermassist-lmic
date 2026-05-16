"""Run the Gradio demo UI (offline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.ui.gradio_app import main

if __name__ == "__main__":
    main()
