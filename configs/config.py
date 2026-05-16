"""
configs/config.py
=================
Global configuration for the DermAssist LMIC project.

All scripts import this file to ensure consistent paths and hyperparameters
across the pipeline (data preprocessing, vision training, RAG construction,
Gemma 4 fine-tuning, and inference).
"""

from pathlib import Path

# ============================================================
# 1. Path configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # Original raw datasets
PROCESSED_DIR = DATA_DIR / "processed"        # Preprocessed datasets
SYNTHETIC_DIR = DATA_DIR / "synthetic"        # Synthetic augmentation data
RAG_DIR = DATA_DIR / "rag_knowledge"          # Raw text sources for RAG
SPLIT_DIR = DATA_DIR / "splits"               # train/val/test split CSVs

MODEL_DIR = PROJECT_ROOT / "models"
VISION_MODEL_DIR = MODEL_DIR / "vision"       # EfficientNet checkpoints
GEMMA_MODEL_DIR = MODEL_DIR / "gemma"         # LoRA adapters
RAG_DB_DIR = MODEL_DIR / "rag_db"              # Vector database files

OUTPUT_DIR = PROJECT_ROOT / "outputs"          # Evaluation results, plots
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# 2. Dataset configuration
# ============================================================
# HAM10000 7-class taxonomy
CLASS_NAMES = [
    "akiec",   # Actinic Keratoses / Bowen's Disease
    "bcc",     # Basal Cell Carcinoma
    "bkl",     # Benign Keratosis
    "df",      # Dermatofibroma
    "mel",     # Melanoma (critical minority class)
    "nv",      # Melanocytic Nevi (majority class)
    "vasc",    # Vascular Lesions
]

# Binary classification mapping
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]  # Malignant / pre-malignant
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]  # Benign

# Minority classes for synthetic augmentation (target: 1,500 samples per class)
MINORITY_CLASSES = ["mel", "df", "vasc", "akiec"]
TARGET_SAMPLES_PER_CLASS = 1500

# ============================================================
# 3. Image preprocessing
# ============================================================
IMAGE_SIZE = 224                                  # EfficientNet-B4 input size
NORMALIZATION_MEAN = [0.7635, 0.5461, 0.5705]     # HAM10000 channel-wise mean
NORMALIZATION_STD = [0.1409, 0.1520, 0.1693]      # HAM10000 channel-wise std

# ============================================================
# 4. Vision Classifier configuration (Stage 1)
# ============================================================
VISION_CONFIG = {
    "model_name": "efficientnet_b4",      # timm model identifier
    "pretrained": True,
    "num_classes": len(CLASS_NAMES),       # 7
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "early_stopping_patience": 7,
    "grad_cam_enabled": True,              # Enable Grad-CAM visualization
    "seed": 42,
    # Loss function with class imbalance handling
    "loss_function": "focal",              # Options: "ce" | "focal" | "ce+focal"
    "focal_gamma": 2.0,                    # Focus parameter for hard samples
    "focal_alpha": None,                   # If None, use class weights
}

# ============================================================
# 5. Gemma 4 LoRA fine-tuning configuration (Stage 2)
# ============================================================
GEMMA_CONFIG = {
    "base_model": "google/gemma-4-E4B-it",
    "quantization": "bfloat16",            # bf16 fits 16GB VRAM (RTX 4080)
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj.linear",
        "k_proj.linear",
        "v_proj.linear",
        "o_proj.linear",
        "gate_proj.linear",
        "up_proj.linear",
        "down_proj.linear",
    ],
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "lr": 5e-5,
    "warmup_steps": 50,
    "epochs": 3,
    "max_seq_length": 2048,
    "freeze_vision_encoder": True,         # Freeze vision encoder for stability
}

# ============================================================
# 6. RAG configuration
# ============================================================
RAG_CONFIG = {
    "embedding_model": "BAAI/bge-m3",     # Multilingual embedding (1024-dim)
    "chunk_size": 512,                     # Token-level chunk size
    "chunk_overlap": 64,
    "top_k": 5,                            # Number of documents to retrieve
    "db_type": "sqlite_vec",               # Tablet-friendly (chromadb for dev)
}

# ============================================================
# 7. Confidence gate
# ============================================================
CONFIDENCE_THRESHOLD = 0.70                # Threshold for detailed analysis
ESCALATION_MESSAGE = (
    "Classifier confidence is below the threshold ({threshold:.0%}). "
    "Specialist dermatology consultation is recommended for accurate diagnosis."
)

# ============================================================
# 8. Auto-create directories
# ============================================================
for d in [
    RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, RAG_DIR, SPLIT_DIR,
    VISION_MODEL_DIR, GEMMA_MODEL_DIR, RAG_DB_DIR,
    OUTPUT_DIR, LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
