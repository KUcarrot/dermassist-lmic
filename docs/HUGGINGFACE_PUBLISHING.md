# Publishing to Hugging Face Hub

This guide describes how to publish the dataset and model weights for reproducibility.

## Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login
```

## Publishing the Training Dataset

```python
from datasets import Dataset
import json

# Load 5,000 LMIC samples
samples = []
with open("data/training_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

dataset = Dataset.from_list(samples)

# Push to Hub
dataset.push_to_hub(
    "YOUR_USERNAME/dermassist-lmic-training-data",
    private=False,
)
```

### Dataset Card Template

Create `README.md` for the dataset:

```yaml
---
license: cc-by-nc-4.0
language:
  - en
tags:
  - medical
  - dermatology
  - LMIC
  - gemma
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
---

# DermAssist LMIC Training Dataset

5,000 synthesized LMIC-specialized samples for fine-tuning Gemma 4 on
dermatology screening tasks.

## Composition

- 8 patient risk profiles
- 5 healthcare resource constraints
- 7 lesion classes (HAM10000 taxonomy)

## Citation

See: https://github.com/YOUR_USERNAME/dermassist-lmic
```

## Publishing the LoRA Adapter

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models/gemma/lora_adapter_en/final_adapter",
    repo_id="YOUR_USERNAME/dermassist-lmic-gemma4-lora",
    repo_type="model",
)
```

### Model Card Template

```yaml
---
base_model: google/gemma-4-E4B-it
library_name: peft
license: cc-by-nc-4.0
language:
  - en
tags:
  - medical
  - dermatology
  - LMIC
  - lora
---

# DermAssist LMIC: Gemma 4 LoRA Adapter

LoRA adapter fine-tuned on 5,000 LMIC-specialized samples for
dermatology screening tasks.

## Performance

- 100% safety pass rate on HAM10000 (in-distribution)
- 98.3% safety pass rate on BCN20000 (external validation)

## Usage

See: https://github.com/YOUR_USERNAME/dermassist-lmic
```
