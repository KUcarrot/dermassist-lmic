# DermAssist LMIC

> **Offline-first AI dermatology screening assistant for Sub-Saharan Africa and other low- and middle-income countries (LMICs).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemma 4](https://img.shields.io/badge/Gemma-4%20E4B-blue.svg)](https://ai.google.dev/gemma)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Hackathon](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-orange.svg)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/KUcarrot/dermassist-lmic-gemma4-lora)

In Sub-Saharan Africa, fewer than **1 dermatologist serves every 1,000,000 people**. DermAssist LMIC brings frontier dermatology AI to clinics 200 km from the nearest specialist, runs entirely offline on a single laptop, and is specifically fine-tuned for LMIC patient contexts including patients with albinism (1000x increased skin cancer risk).

---

## Demo

**Video demo:** [YouTube link](https://youtu.be/GnW1g7sm1WY)

---

## Key Results

The system was validated on two datasets to test cross-dataset robustness:

| Metric | HAM10000 (in-distribution, n=35) | BCN20000 (external, n=60) |
|---|---|---|
| Vision Classifier accuracy | 60.0% | 28.3% |
| Urgency-recommendation consistency | 100.0% | 100.0% |
| Hallucination-free output | 100.0% | 98.3% |
| Safety disclaimer inclusion | 100.0% | 100.0% |
| **Overall safety pass rate** | **100.0%** | **98.3%** |

**Key finding:** The system maintains safety guarantees under significant distribution shift, even when the upstream Vision Classifier accuracy drops by 32 percentage points. This is the result of deliberate "safety-by-design" through LMIC-specialized fine-tuning.

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
- **RAG Knowledge Base:** SQLite + BAAI/bge-m3 embeddings (1024-dim, 246 chunks), indexed from DermNet, BAD guidelines, and WHO LMIC dermatology protocols
- **LLM:** Gemma 4 E4B (4-bit quantized) + LoRA adapter (r=32, alpha=16) fine-tuned for LMIC clinical scenarios
- **UI:** Gradio with Noto Sans font (offline, multi-language ready)

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16 GB+ VRAM (tested on RTX 4080)
- CUDA 12.4
- 30 GB disk space for models and base Gemma 4

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/KUcarrot/dermassist-lmic.git
cd dermassist-lmic

# 2. Install PyTorch with CUDA 12.4 (must be done first)
pip install torch==2.6.0 torchvision==0.21.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Install the package and all dependencies
pip install -e .
```

### Download Pre-trained Models

All model artifacts (LoRA adapter, Vision Classifier, RAG knowledge base) are hosted on HuggingFace Hub. Download them with Python:

```python
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "KUcarrot/dermassist-lmic-gemma4-lora"

# Download LoRA adapter (full directory)
snapshot_download(
    repo_id=REPO_ID,
    local_dir="./models/gemma/lora_adapter/final_adapter",
    allow_patterns=["adapter_*", "tokenizer*", "chat_template*", "processor_config.json"],
)

# Download Vision Classifier
hf_hub_download(
    repo_id=REPO_ID,
    filename="vision_classifier.pth",
    local_dir="./models/vision",
)

# Download RAG knowledge base
hf_hub_download(
    repo_id=REPO_ID,
    filename="medical_knowledge.db",
    local_dir="./models/rag_db",
)
```

Or with the `hf` CLI:

```bash
hf auth login   # First-time login with your HuggingFace token

# Download everything to the correct local paths
hf download KUcarrot/dermassist-lmic-gemma4-lora \
    --local-dir models/gemma/lora_adapter/final_adapter \
    --include "adapter_*" "tokenizer*" "chat_template*" "processor_config.json"

hf download KUcarrot/dermassist-lmic-gemma4-lora vision_classifier.pth \
    --local-dir models/vision

hf download KUcarrot/dermassist-lmic-gemma4-lora medical_knowledge.db \
    --local-dir models/rag_db
```

### Run the Demo

```bash
python scripts/06_run_demo.py
```

Open `http://localhost:7860` in your browser. To verify offline operation, disconnect your network — the UI status should change to **OFFLINE MODE**.

Example images are provided in `examples/images/`:
- `nv_ISIC_0025610.jpg` — benign nevus (expected: routine triage)
- `bcc_ISIC_0031527.jpg` — basal cell carcinoma in albinism patient (expected: urgent triage)

See [`examples/README.md`](examples/README.md) for the corresponding patient information templates.

---

## Reproducing the Results

The Vision Classifier, LoRA adapter, and RAG database in this repository are all reproducible. Training code is provided locally but excluded from the public repository (one-time setup); the pre-trained artifacts are published on HuggingFace.

To reproduce the full pipeline from scratch:

```bash
# 1. Prepare HAM10000 data (requires Kaggle credentials)
python scripts/01_prepare_data.py

# 2. Train Vision Classifier (~2 hours on RTX 4080)
python scripts/02_train_vision.py

# 3. Build RAG knowledge base
python scripts/03_build_rag.py

# 4. Generate LMIC-specialized training data
python scripts/04_generate_training_data.py

# 5. Fine-tune Gemma 4 E4B with LoRA (~6 hours)
python scripts/05_train_gemma_lora.py

# 6. Evaluate on HAM10000 test split
python scripts/07_evaluate_ham10000.py

# 7. External validation on BCN20000
python scripts/08_evaluate_bcn20000.py
```

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
├── src/dermassist/         # Core inference package
│   ├── data/               # Data preprocessing, synthetic generation
│   ├── llm/                # Gemma 4 JSON parsing
│   ├── pipeline/           # End-to-end inference pipeline (includes RAGRetriever)
│   ├── ui/                 # Gradio UI with Noto Sans
│   └── evaluation/         # HAM10000 / BCN20000 evaluation
│
├── scripts/                # Executable entry points (01-08)
├── configs/                # Configuration (paths, hyperparameters)
├── docs/                   # Architecture and evaluation documentation
├── fonts/                  # Noto Sans (offline rendering)
└── examples/               # Sample dermoscopy images and scenarios
```

The following are excluded from this repository (available separately on HuggingFace Hub):
- **Training scripts** (Vision, RAG build, LoRA): one-time setup; published artifacts are on HuggingFace
- `data/` — datasets (download from Kaggle / ISIC Archive)
- `models/` — trained weights (download from HuggingFace Hub as shown above)
- `outputs/` — evaluation results (regenerated by scripts 07-08)

---

## Configuration

Key settings in `configs/config.py`:

| Parameter | Value | Description |
|---|---|---|
| `IMAGE_SIZE` | 224 | EfficientNet-B4 input size |
| `CLASS_NAMES` | 7 classes | mel, nv, bcc, akiec, bkl, df, vasc |
| `CONFIDENCE_THRESHOLD` | 0.70 | Below this, escalate to specialist |
| `GEMMA_CONFIG["base_model"]` | google/gemma-4-E4B-it | Base LLM |
| `GEMMA_CONFIG["lora_r"]` | 32 | LoRA rank |
| `GEMMA_CONFIG["lora_alpha"]` | 16 | LoRA alpha |
| `RAG_CONFIG["embedding_model"]` | BAAI/bge-m3 | Multilingual embeddings (1024-dim) |
| `RAG_CONFIG["top_k"]` | 5 | Documents retrieved per query |

---

## Why "Safety-by-Design"

Most medical AI systems rely on classifier confidence as the primary safety signal. DermAssist LMIC takes a different approach:

1. **Patient context awareness:** The LLM considers patient risk factors (albinism, HIV+) and resource constraints (specialist distance, biopsy availability) — not just classifier output.

2. **Conservative referral patterns:** Borderline confidence cases automatically escalate to specialist consultation through LMIC-specialized training corpus.

3. **Mandatory safety disclaimers:** Every output explicitly states AI limitations, even on high-confidence cases.

4. **Cross-dataset robustness:** External validation on BCN20000 demonstrated 98.3% safety pass rate despite Vision Classifier accuracy dropping to 28.3%.

This means the system maintains patient safety even when the Vision Classifier encounters unfamiliar populations (Sub-Saharan African skin types not represented in training data).

---

## Limitations

- **Geographic bias:** Trained on HAM10000 (Austria/USA) and validated on BCN20000 (Spain). True validation on Sub-Saharan African populations requires regional partnerships.
- **Hardware requirements:** RTX 4080-tier GPU needed. Future work: optimization for edge devices via Gemma 4 E2B.
- **Inference time:** Approximately 60 seconds per case. Acceptable for non-emergency screening but not real-time triage.
- **Vision domain adaptation:** External validation revealed substantial classifier degradation on out-of-distribution data, requiring domain-specific retraining before deployment in target populations.

---

## Pre-trained Models

All model artifacts are available on HuggingFace Hub at a single repository:

**[KUcarrot/dermassist-lmic-gemma4-lora](https://huggingface.co/KUcarrot/dermassist-lmic-gemma4-lora)**

| Artifact | File | Size |
|---|---|---|
| Gemma 4 LoRA adapter | `adapter_model.safetensors` + tokenizer | ~50 MB |
| Vision Classifier | `vision_classifier.pth` | 212 MB |
| RAG knowledge base | `medical_knowledge.db` | 1.35 MB |

---

## Citation

If you use this work, please cite:

```bibtex
@software{kim2026dermassist,
  author = {Kim, Donggeun},
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
- BCN20000 dataset (Hernandez-Perez et al., 2024)
- Gemma 4 model (Google DeepMind)
- DermNet, British Association of Dermatologists for medical knowledge sources
- African Teledermatology Project for LMIC clinical context
- Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Noto Sans fonts are licensed under [SIL Open Font License 1.1](fonts/OFL.txt).

The fine-tuned models on HuggingFace Hub are released under Apache 2.0. The base Gemma 4 model is subject to [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## Contact

For questions or collaboration: jikksun@korea.ac.kr
