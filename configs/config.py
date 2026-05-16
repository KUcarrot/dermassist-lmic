"""
configs/config.py
=================
프로젝트 전역 설정값.
모든 스크립트가 이 파일을 import하여 경로·하이퍼파라미터를 일관되게 사용.
"""

from pathlib import Path

# ============================================================
# 1. 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # 원본 데이터셋
PROCESSED_DIR = DATA_DIR / "processed"        # 전처리 완료 데이터
SYNTHETIC_DIR = DATA_DIR / "synthetic"        # 합성 데이터
RAG_DIR = DATA_DIR / "rag_knowledge"          # RAG 지식베이스 원본 텍스트
SPLIT_DIR = DATA_DIR / "splits"              # train/val/test 분할 CSV

MODEL_DIR = PROJECT_ROOT / "models"
VISION_MODEL_DIR = MODEL_DIR / "vision"       # EfficientNet 체크포인트
GEMMA_MODEL_DIR = MODEL_DIR / "gemma"         # LoRA 어댑터
RAG_DB_DIR = MODEL_DIR / "rag_db"            # 벡터 DB 파일

OUTPUT_DIR = PROJECT_ROOT / "outputs"         # 평가 결과, 그래프 등
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# 2. 데이터셋 설정
# ============================================================
# HAM10000 클래스 정의 (7-class)
CLASS_NAMES = [
    "akiec",   # 광선각화증 / 보웬병 (Actinic Keratoses)
    "bcc",     # 기저세포암 (Basal Cell Carcinoma)
    "bkl",     # 양성 각화증 (Benign Keratosis)
    "df",      # 피부섬유종 (Dermatofibroma)
    "mel",     # 멜라노마 (Melanoma) ← 핵심 소수 클래스
    "nv",      # 멜라닌세포모반 (Melanocytic Nevi) ← 다수 클래스
    "vasc",    # 혈관병변 (Vascular Lesions)
]

# 악성 vs 양성 이진 분류 매핑
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]  # 악성/전암성
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]  # 양성

# 합성데이터로 보강할 소수 클래스 (목표: 클래스당 1,500장)
MINORITY_CLASSES = ["mel", "df", "vasc", "akiec"]
TARGET_SAMPLES_PER_CLASS = 1500

# ============================================================
# 3. 전처리 설정
# ============================================================
IMAGE_SIZE = 224                # EfficientNet-B4 입력 크기
NORMALIZATION_MEAN = [0.7635, 0.5461, 0.5705]  # HAM10000 통계값
NORMALIZATION_STD = [0.1409, 0.1520, 0.1693]

# ============================================================
# 4. Vision Classifier 설정 (Stage 1)
# ============================================================
VISION_CONFIG = {
    "model_name": "efficientnet_b4",   # timm 모델명
    "pretrained": True,
    "num_classes": len(CLASS_NAMES),    # 7
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "early_stopping_patience": 7,
    "grad_cam_enabled": True,          # Grad-CAM 시각화 활성화
    "seed": 42,
}

VISION_CONFIG.update({
    "loss_function": "focal",   # "ce" | "focal" | "ce+focal"
    "focal_gamma": 2.0,         # 어려운 샘플에 더 집중
    "focal_alpha": None,        # None이면 class_weights 사용
})
# ============================================================
# 5. Gemma LoRA 설정 (Stage 2)
# ============================================================


GEMMA_CONFIG = {
    "base_model": "google/gemma-4-E4B-it",
    "quantization": "bfloat16",            # 4080 16GB 대응 (bf16도 가능)
    "lora_r": 32, # 32
    "lora_alpha": 16, # 64
    "lora_dropout": 0.05,
    "lora_target_modules": [
    "q_proj.linear", 
    "k_proj.linear", 
    "v_proj.linear", 
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear"
],
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "lr": 5e-5, #2e-4,
    #"warmup_ratio": 0.03,
    "warmup_steps": 50,
    "epochs": 3, # 3
    "max_seq_length": 2048, # 2048
    "freeze_vision_encoder": True,     # Vision encoder 고정 (안정성)
}

# ============================================================
# 6. RAG 설정
# ============================================================
RAG_CONFIG = {
    "embedding_model": "BAAI/bge-m3", #"all-MiniLM-L6-v2"(영어전용),  # 경량 임베딩 (22MB)
    "chunk_size": 512,             # 토큰 단위 청크 크기
    "chunk_overlap": 64,
    "top_k": 5,                    # 검색 시 반환할 문서 수
    "db_type": "sqlite_vec",       # 태블릿 호환 (chromadb는 개발용)
}

# ============================================================
# 7. Confidence Gate 설정
# ============================================================
CONFIDENCE_THRESHOLD = 0.70        # 이 이상일 때만 상세 분석 진행
ESCALATION_MESSAGE = (
    "분류 신뢰도가 기준({threshold:.0%}) 미만입니다. "
    "정확한 판단을 위해 피부과 전문의 상담을 권고합니다."
)

# ============================================================
# 8. 디렉터리 자동 생성
# ============================================================
for d in [
    RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, RAG_DIR, SPLIT_DIR,
    VISION_MODEL_DIR, GEMMA_MODEL_DIR, RAG_DB_DIR,
    OUTPUT_DIR, LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
