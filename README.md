# DermAssist LMIC

> **Offline-first AI dermatology screening assistant for Sub-Saharan Africa and other low- and middle-income countries (LMICs).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemma 4](https://img.shields.io/badge/Gemma-4%20E4B-blue.svg)](https://ai.google.dev/gemma)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Hackathon](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-orange.svg)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)

In Sub-Saharan Africa, fewer than **1 dermatologist serves every 1,000,000 people**. DermAssist LMIC brings frontier dermatology AI to clinics 200 km from the nearest specialist, runs entirely offline on a single laptop, and is specifically fine-tuned for LMIC patient contexts including patients with albinism (1000x increased skin cancer risk).

---

## Demo

**Video demo:** [YouTube link](https://youtube.com/watch?v=YOUR_VIDEO_ID)

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/KUcarrot/dermassist-lmic)

![UI Screenshot](docs/images/ui_screenshot.png)

---

## Key Results

The system was validated on two datasets to test cross-dataset robustness:

| Metric | HAM10000 (in-distribution) | BCN20000 (external) |
|---|---|---|
| Vision Classifier accuracy | 60.0% | 28.3% |
| Urgency-recommendation consistency | 100.0% | 100.0% |
| Hallucination-free output | 100.0% | 98.3% |
| Safety disclaimer inclusion | 100.0% | 100.0% |
| **Overall safety pass rate** | **100.0%** | **98.3%** |

**Key finding:** The system maintains safety guarantees under significant distribution shift, even when the upstream Vision Classifier accuracy drops by 32 percentage points. This is the result of deliberate "safety-by-design" through LMIC-specialized fine-tuning.

See [docs/EVALUATION.md](docs/EVALUATION.md) for detailed methodology and analysis.

---

## Architecture

```
                  [Skin Lesion Image]
                          |
                          v
              [DullRazor Hair Removal]
                          |
                          v
              [Vision Classifier]
              (EfficientNet-B4)
                          |
                          v
          [Patient Context]
                  |
                  v
          [RAG Retrieval] <-- (DermNet, BAD, WHO)
                  |
                  v
          [Gemma 4 E4B + LoRA]
          (LMIC-specialized)
                  |
                  v
   [Urgency | Recommendation | Patient Summary | Limitations]
```

### Components

- **Vision Classifier:** EfficientNet-B4 fine-tuned on HAM10000 (10,015 dermatoscopic images, 7 classes)
- **Hair Removal:** DullRazor algorithm (Lee et al., 1997) for image preprocessing
- **RAG Knowledge Base:** SQLite + BAAI/bge-m3 embeddings, indexed from DermNet, BAD guidelines, and WHO LMIC dermatology protocols
- **LLM:** Gemma 4 E4B (4-bit quantized) + LoRA adapter fine-tuned on 5,000 LMIC-specialized samples
- **UI:** Gradio with Noto Sans font (offline, multi-language ready)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16 GB+ VRAM (tested on RTX 4080)
- 30 GB disk space for models and data

### Installation

```bash
git clone https://github.com/KUcarrot/dermassist-lmic.git
cd dermassist-lmic
pip install -e .
```

This installs DermAssist LMIC and all dependencies in editable mode.

### Download Pre-trained Models

The Vision Classifier weights and RAG database must be built locally (see Reproducing Results). The Gemma 4 LoRA adapter is available on Hugging Face Hub:

```bash
# Pre-trained LoRA adapter
huggingface-cli download KUcarrot/dermassist-lmic-gemma4-lora \
    --local-dir models/gemma/lora_adapter/final_adapter

# Training data (5,000 LMIC samples)
huggingface-cli download KUcarrot/dermassist-lmic-training-data \
    --repo-type dataset \
    --local-dir data/training
```

### Run the Demo

```bash
python scripts/06_run_demo.py
```

Open http://localhost:7860 in your browser. To verify offline operation, disconnect your network — the UI status should change to "OFFLINE MODE".

Use the example images in `examples/`:
- `sample_nv.jpg` — benign nevus (expected: routine triage)
- `sample_bcc.jpg` — basal cell carcinoma (expected: urgent triage)

---

## Reproducing the Results

To reproduce the full pipeline from scratch:

```bash
# 1. Prepare HAM10000 data (requires Kaggle credentials)
python scripts/01_prepare_data.py

# 2. Train Vision Classifier (~2 hours on RTX 4080)
python scripts/02_train_vision.py

# 3. Build RAG knowledge base
python scripts/03_build_rag.py

# 4. Generate LMIC-specialized training data (5,000 samples)
python scripts/04_generate_training_data.py

# 5. Fine-tune Gemma 4 E4B with LoRA (~4 hours)
python scripts/05_train_gemma_lora.py

# 6. Evaluate on HAM10000 test split
python scripts/07_evaluate_ham10000.py

# 7. External validation on BCN20000
python scripts/08_evaluate_bcn20000.py
```

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed data setup instructions.

---

## Project Structure

```
dermassist-lmic/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── src/dermassist/         # Core package
│   ├── data/               # Data preprocessing, schema, synthetic generation
│   ├── vision/             # Vision Classifier (training, evaluation)
│   ├── rag/                # RAG knowledge base construction
│   ├── llm/                # Gemma 4 LoRA training, JSON parsing
│   ├── pipeline/           # End-to-end inference pipeline
│   ├── ui/                 # Gradio UI with Noto Sans
│   └── evaluation/         # HAM10000 / BCN20000 evaluation
│
├── scripts/                # Executable entry points (01-08)
├── configs/                # Configuration (paths, hyperparameters)
├── docs/                   # Architecture, evaluation, data preparation
├── fonts/                  # Noto Sans (offline rendering)
├── examples/               # Sample images for quick testing
└── tests/                  # Unit tests (optional)
```

Large files excluded from this repository:
- `data/` — datasets (download from Kaggle / ISIC Archive)
- `models/` — trained weights (download from Hugging Face Hub)
- `outputs/` — evaluation results (regenerated by scripts)

---

## Why "Safety-by-Design"

Most medical AI systems rely on classifier confidence as the primary safety signal. DermAssist LMIC takes a different approach:

1. **Patient context awareness:** The LLM considers patient risk factors (albinism, HIV+) and resource constraints (specialist distance, biopsy availability) — not just classifier output.

2. **Conservative referral patterns:** Borderline confidence cases automatically escalate to specialist consultation through LMIC-specialized training corpus.

3. **Mandatory safety disclaimers:** Every output explicitly states AI limitations, even on high-confidence cases.

4. **Cross-dataset robustness:** External validation on BCN20000 demonstrated 98.3% safety pass rate despite Vision Classifier accuracy dropping to 28.3%.

This means the system maintains patient safety even when the Vision Classifier encounters unfamiliar populations (Sub-Saharan African skin types not in training data).

---

## Limitations

- **Geographic bias:** Trained on HAM10000 (Austria/USA) and validated on BCN20000 (Spain). True validation on Sub-Saharan African populations requires regional partnerships.
- **Hardware requirements:** RTX 4080 tier GPU needed. Future work: optimization for edge devices via Gemma 4 E2B.
- **Inference time:** ~65 seconds per case. Acceptable for non-emergency screening but not real-time triage.
- **Vision domain adaptation:** External validation revealed substantial classifier degradation on out-of-distribution data, requiring domain-specific retraining before deployment in target populations.

---

## Citation

If you use this work, please cite:

```bibtex
@software{dermassist-lmic-2026,
  author = {Donggeun Kim},
  title = {DermAssist LMIC: Offline-First AI Dermatology Screening for Sub-Saharan Africa},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/KUcarrot/dermassist-lmic}},
  note = {Kaggle Gemma 4 Good Hackathon submission}
}
```

---

## Acknowledgments

- HAM10000 dataset (Tschandl et al., 2018)
- BCN20000 dataset (Hernández-Pérez et al., 2024)
- Gemma 4 model (Google DeepMind)
- DermNet, British Association of Dermatologists for medical knowledge sources
- African Teledermatology Project for LMIC clinical context
- Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Noto Sans fonts are licensed under [SIL Open Font License 1.1](fonts/OFL.txt).

---

## Contact

For questions or collaboration: jikksun@korea.ac.kr
