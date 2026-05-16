"""Fine-tune Gemma 4 E4B with LoRA on LMIC samples.

Pre-trained LoRA adapter is available at:
  https://huggingface.co/YOUR_USERNAME/dermassist-lmic-gemma4-lora
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.llm.train_lora import main

if __name__ == "__main__":
    main()
