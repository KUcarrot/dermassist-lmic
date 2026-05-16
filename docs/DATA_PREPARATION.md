# Data Preparation Guide

## HAM10000 Dataset (Training)

The HAM10000 dataset is used to train the Vision Classifier.

### Download

1. Visit [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Download with Kaggle CLI:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```
3. Extract to `data/raw/ham10000/`

### Preprocessing

```bash
python scripts/01_prepare_data.py
```

This will:
- Split into train/val/test (80/10/10)
- Apply standard image transformations
- Save processed images to `data/processed/ham10000/`

## BCN20000 Dataset (External Validation)

### Download from ISIC Archive

1. Register at [ISIC Archive](https://www.isic-archive.com/)
2. Navigate to BCN20000 collection (#249):
   https://api.isic-archive.com/collections/249/
3. Download the full collection (CSV metadata + images)

### Prepare for Evaluation

```bash
# Place files at:
#   data/external/bcn20000/bcn20000_metadata_*.csv
#   data/external/bcn20000/ISIC-images/

python scripts/08_evaluate_bcn20000.py
```

The evaluation script will:
- Sample 10 images per class (6 mappable classes)
- Run full pipeline (Vision + RAG + Gemma 4)
- Output stratified results to `outputs/external_validation/`

## LMIC Training Data (Pre-generated)

The 5,000 LMIC-specialized training samples used to fine-tune Gemma 4 are available as a pre-generated dataset:

**Hugging Face Hub:**
```
https://huggingface.co/datasets/YOUR_USERNAME/dermassist-lmic-training-data
```

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/dermassist-lmic-training-data")
```

### Re-generating from Scratch

```bash
python scripts/04_generate_training_data.py
```

This generates 5,000 samples across:
- 8 patient profiles (general LMIC, albinism, HIV+, etc.)
- 5 resource constraints (200km distance, no biopsy, etc.)
- 7 lesion classes
- Various symptom combinations

Estimated time: ~30 minutes on CPU.

## Pre-trained Models

### Gemma 4 LoRA Adapter

The fine-tuned LoRA adapter is available at:

**Hugging Face Hub:**
```
https://huggingface.co/YOUR_USERNAME/dermassist-lmic-gemma4-lora
```

Load with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/dermassist-lmic-gemma4-lora")
```

### Vision Classifier Weights

The trained EfficientNet-B4 weights are not currently hosted publicly. To reproduce:

```bash
python scripts/02_train_vision.py
```

Training time: ~2 hours on RTX 4080.
