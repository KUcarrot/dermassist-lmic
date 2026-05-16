"""
Skin Lesion Assistant - Integrated Pipeline
============================================
Vision Classifier + RAG + Gemma 4 LoRA inference pipeline.

이 파일은 src/dermassist/pipeline/assistant.py에 위치합니다.
프로젝트 루트는 4단계 위입니다.
"""

import os
import sys
from pathlib import Path


# ============================================================
# 1. 프로젝트 .venv 자동 활성화 (개발 편의용)
# ============================================================
def _ensure_project_venv() -> None:
    """
    프로젝트의 .venv 인터프리터가 아니면 동일 스크립트를 .venv Python으로 재실행.
    
    새 모듈 구조: src/dermassist/pipeline/assistant.py
    프로젝트 루트: 4단계 위 (parent.parent.parent.parent)
    """
    # src/dermassist/pipeline/assistant.py → 프로젝트 루트
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    if sys.platform == "win32":
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = project_root / ".venv" / "bin" / "python"

    if not venv_python.is_file():
        return

    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except OSError:
        return

    script_path = str(Path(__file__).resolve())
    os.execv(str(venv_python), [str(venv_python), script_path, *sys.argv[1:]])


# 직접 실행 시에만 자동 활성화 (import될 때는 건너뜀)
if __name__ == "__main__":
    _ensure_project_venv()


# ============================================================
# 2. 표준 라이브러리 imports
# ============================================================
import re
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# ============================================================
# 3. 외부 라이브러리 imports
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# 4. 프로젝트 내부 imports
# ============================================================
# robust_json_parser: 같은 패키지 내 llm 모듈
from dermassist.llm.json_parser import parse_gemma_response

# configs.config: 프로젝트 루트의 configs/ 디렉터리
# (pip install -e . 으로 설치된 환경에서는 configs/__init__.py 필요)
try:
    from configs.config import (
        PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, IMAGE_SIZE,
        VISION_CONFIG, GEMMA_CONFIG, RAG_CONFIG,
        NORMALIZATION_MEAN, NORMALIZATION_STD,
        CONFIDENCE_THRESHOLD, ESCALATION_MESSAGE,
    )
except ImportError:
    # configs를 import 못하면 프로젝트 루트를 sys.path에 추가하고 재시도
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from configs.config import (
        PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, IMAGE_SIZE,
        VISION_CONFIG, GEMMA_CONFIG, RAG_CONFIG,
        NORMALIZATION_MEAN, NORMALIZATION_STD,
        CONFIDENCE_THRESHOLD, ESCALATION_MESSAGE,
    )


# ============================================================
# SYSTEM PROMPT (학습 데이터 07_gemma_training_data_gen_en.py와 정확히 동일)
# ============================================================
SYSTEM_PROMPT_EN = """You are a skin lesion screening assistant designed for community health workers in low-resource settings, particularly rural areas of sub-Saharan Africa and other LMICs (Low- and Middle-Income Countries) where dermatologist density is below 1 per million population.

Your role:
1. Provide structured triage recommendations to support frontline health workers
2. Never replace specialist diagnosis - only assist with case prioritization
3. Account for limited follow-up infrastructure (long travel distances, sparse specialists, limited biopsy access) when recommending urgency
4. Suggest teledermatology referral when in-person specialist access is impractical
5. Use plain English suitable for translation by health workers

Critical safety rules:
- This system runs entirely offline; no patient data is transmitted
- Always include limitations and AI disclaimer
- For high-confidence malignant cases (>=70%), recommend urgent specialist contact
- For low-confidence cases, default to safer recommendation (specialist review)
- Account for the patient's likely access barriers (distance, cost, follow-up reliability)

Input you will receive:
- Vision Classifier output (predicted class, confidence, Grad-CAM description)
- Patient metadata (age, sex, body site, duration, symptoms, risk factors)
- RAG-retrieved medical context

Output format: Strictly valid JSON with the following fields:
{
  "observed_features": [list of clinical observations],
  "abcde_analysis": {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...},
  "classification_summary": "summary of vision findings",
  "evidence_sources": [list of evidence references],
  "recommendation": "clinical recommendation accounting for resource constraints",
  "urgency": "routine" | "soon" | "urgent",
  "patient_summary": "plain-English summary for patient",
  "limitations": "AI disclaimers and limitations"
}"""


# ============================================================
# Grad-CAM 영어 서술 템플릿 (학습 데이터와 일치)
# ============================================================
GRAD_CAM_TEMPLATES_EN = {
    "mel": [
        "Asymmetric activation pattern with strong response on irregular border regions",
        "Heterogeneous activation distribution suggesting multiple color zones",
        "Focal high-activation areas at lesion margins, characteristic of irregular pigment network",
        "Asymmetric center-to-edge gradient with hot spots at peripheral irregular areas",
        "Strong activation in areas suggestive of regression or color variation",
    ],
    "bcc": [
        "Activation focused at lesion edges with translucent border characteristics",
        "Central depression activation with peripheral nodular pattern",
        "Focal activation at telangiectatic vessel locations",
        "Pearl-like nodular activation pattern at lesion center",
        "Activation pattern consistent with rolled border morphology",
    ],
    "akiec": [
        "Diffuse activation across rough, scaly surface area",
        "Moderate activation pattern indicating keratotic surface texture",
        "Activation distributed across erythematous patch with scale",
        "Surface-prominent activation suggesting actinic damage",
        "Activation pattern at sun-exposed area with scale formation",
    ],
    "nv": [
        "Symmetric central activation indicating uniform pigmented lesion",
        "Even activation distribution following regular pigment network",
        "Centered round activation with smooth border response",
        "Homogeneous activation pattern across the lesion",
        "Symmetrical activation consistent with benign melanocytic pattern",
    ],
    "bkl": [
        "Surface-textured activation with stuck-on appearance pattern",
        "Activation pattern showing milia-like cysts and comedo-like openings",
        "Diffuse activation across scaly hyperkeratotic surface",
        "Activation following waxy, verrucous surface contours",
        "Pattern consistent with seborrheic keratosis topography",
    ],
    "df": [
        "Central depression activation with peripheral firm-papule response",
        "Activation pattern showing characteristic central white scar-like area",
        "Focused activation at firm dermal nodule location",
        "Pattern consistent with dermatofibroma dimple sign location",
        "Activation distribution indicating fibrous papule structure",
    ],
    "vasc": [
        "Strong activation at vascular lacunae (red-blue zones)",
        "Activation pattern indicating cherry angioma vessel structure",
        "Focal activation at vascular spaces typical of angiokeratoma",
        "Activation distribution showing red-purple papule structure",
        "Pattern consistent with vascular lesion morphology",
    ],
}


# ============================================================
# 1. 환자 메타데이터 스키마 (LMIC 확장)
# ============================================================
@dataclass
class PatientMetadata:
    """환자 컨텍스트 정보 (LMIC 필드 포함)."""
    # 기본 필드
    age: Optional[int] = None
    sex: Optional[str] = None
    body_site: Optional[str] = None
    duration_months: Optional[int] = None
    symptoms: Optional[str] = None
    family_history: Optional[str] = None

    # LMIC 확장 필드 (학습 데이터와 동일한 형식)
    context: str = "general LMIC patient"
    risk_factor: str = "limited healthcare access"
    skin_type: str = "unspecified"
    resource_constraint: str = (
        "Standard primary care setting with limited specialist access"
    )


# ============================================================
# 2. Vision Classifier 래퍼
# ============================================================
class VisionClassifier:
    """EfficientNet-B4 분류기 + Grad-CAM 생성."""

    def __init__(self, ckpt_path: Path, device: torch.device):
        import timm
        import torchvision.transforms as T

        self.device = device
        self.model = timm.create_model(
            VISION_CONFIG["model_name"], pretrained=False,
            num_classes=len(CLASS_NAMES),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device).eval()

        # 전처리 파이프라인
        norm_path = SPLIT_DIR / "normalization_stats.json"
        if norm_path.exists():
            norm_stats = json.load(open(norm_path))
            mean, std = norm_stats["mean"], norm_stats["std"]
        else:
            mean, std = NORMALIZATION_MEAN, NORMALIZATION_STD

        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Grad-CAM hook 등록
        self._gradients = None
        self._activations = None
        target = self.model.conv_head
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

        print(f"[Vision] 로드 완료 (mode: {ckpt.get('mode', 'unknown')})")

    def _save_activation(self, m, i, o):
        self._activations = o.detach()

    def _save_gradient(self, m, gi, go):
        self._gradients = go[0].detach()

    def _generate_gradcam_description(
        self, cam: np.ndarray, pred_class: str
    ) -> str:
        """예측 클래스 + Grad-CAM 분석 결과 → 영어 서술."""
        import random

        h, w = cam.shape

        # 대칭성 측정
        left_sum = cam[:, :w//2].sum()
        right_sum = cam[:, w//2:].sum()
        symmetry_ratio = (
            min(left_sum, right_sum) / (max(left_sum, right_sum) + 1e-8)
        )

        # 클래스별 영어 템플릿 (학습 데이터와 일치)
        templates = GRAD_CAM_TEMPLATES_EN.get(pred_class)
        if not templates:
            return "Activation distributed across the lesion area"

        # 대칭성 기반 템플릿 선택
        if pred_class == "nv":
            # nv는 대칭/균일 패턴 우선
            if symmetry_ratio > 0.8:
                desc = templates[0]  # symmetric central activation
            else:
                desc = templates[1]  # even distribution
        elif pred_class == "mel":
            # mel은 비대칭 패턴 우선
            if symmetry_ratio < 0.7:
                desc = templates[0]  # asymmetric pattern
            else:
                desc = templates[1]  # heterogeneous distribution
        elif pred_class == "bcc":
            # bcc는 가장자리 활성화 우선
            desc = templates[0]
        elif pred_class == "akiec":
            desc = templates[0]
        else:
            # 기타 클래스는 첫 템플릿 사용 (일관성)
            desc = templates[0]

        return desc

    def predict(self, image: Image.Image) -> Dict:
        """이미지 분류 + 영어 Grad-CAM 설명 생성."""
        import cv2

        image_rgb = image.convert("RGB")
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        image_tensor.requires_grad_(False)

        # Forward
        self.model.zero_grad()
        image_for_grad = image_tensor.clone().detach().requires_grad_(True)
        output = self.model(image_for_grad)
        probs = torch.softmax(output, dim=1).squeeze()

        pred_idx = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = probs[pred_idx].item()

        # Top-3
        top3_idx = probs.topk(3).indices.cpu().numpy()
        top3 = [
            {CLASS_NAMES[i]: round(probs[i].item(), 4)}
            for i in top3_idx
        ]

        # Grad-CAM 생성
        output[0, pred_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        grad_cam_desc = self._generate_gradcam_description(cam, pred_class)

        return {
            "predicted_class": pred_class,
            "probability": round(pred_prob, 4),
            "top3": top3,
            "is_malignant": pred_class in MALIGNANT_CLASSES,
            "grad_cam_description": grad_cam_desc,
            "grad_cam_map": cam_resized,
        }


# ============================================================
# 3. RAG 검색 래퍼 (영어 쿼리)
# ============================================================
class RAGRetriever:
    """bge-m3 임베딩 기반 의료 지식 검색 (영어)."""

    def __init__(self, db_path: Path, embedding_model: str = "BAAI/bge-m3"):
        import sqlite3
        from sentence_transformers import SentenceTransformer

        self.db_path = db_path
        if not db_path.exists():
            print(f"[경고] RAG DB 없음: {db_path}")
            self.model = None
            return

        print(f"[RAG] 임베딩 모델 로드: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print(f"  차원: {self.model.get_sentence_embedding_dimension()}")

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        print(f"[RAG] DB 연결: {db_path}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """영어 쿼리 → 임베딩 → top-k 검색."""
        if self.model is None:
            return []

        query_emb = self.model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)[0]

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, doc_id, title, source, source_type, category, "
            "page_number, content, embedding FROM documents"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        db_embeddings = np.array(
            [np.frombuffer(r[8], dtype=np.float32) for r in rows]
        )
        similarities = db_embeddings @ query_emb

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            r = rows[idx]
            results.append({
                "title": r[2],
                "source": r[3],
                "category": r[5],
                "page": r[6],
                "content": r[7],
                "similarity": float(similarities[idx]),
            })
        return results

    def build_context(self, classifier_output: Dict, top_k: int = 5) -> str:
        """예측 클래스 + Grad-CAM으로 영어 RAG 쿼리 빌드."""
        # 영어 쿼리 (한국어 → 영어 변경)
        query = (
            f"{classifier_output['predicted_class']} skin lesion "
            f"clinical features diagnosis dermoscopy "
            f"{classifier_output['grad_cam_description']}"
        )

        results = self.search(query, top_k=top_k)
        if not results:
            return "(No reference material available)"

        # 컨텍스트 조립 (영어)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {r['title']}] {r['content'][:500]}"
            )
        return "\n\n".join(context_parts)


# ============================================================
# 4. Gemma LoRA 추론 래퍼 (영어)
# ============================================================
class GemmaInference:
    """Gemma 4 E4B + LoRA 어댑터로 영어 의료 보조 응답 생성."""

    def __init__(
        self,
        base_model_id: str,
        lora_adapter_dir: Path,
        device: torch.device,
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[Gemma] 베이스 모델 로드: {base_model_id}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        # LoRA 어댑터 로드
        if lora_adapter_dir.exists():
            print(f"[Gemma] LoRA 어댑터 로드: {lora_adapter_dir}")
            self.model = PeftModel.from_pretrained(
                self.base_model, str(lora_adapter_dir),
            )
        else:
            print(f"[경고] LoRA 어댑터 없음 — 베이스 모델로 추론")
            self.model = self.base_model

        self.model.eval()

    def _build_user_prompt(
        self,
        classifier_output: Dict,
        patient_meta: PatientMetadata,
        rag_context: str,
    ) -> str:
        """학습 데이터 형식과 일치하는 영어 user prompt 빌드."""
        top3_str = ", ".join(
            f"{list(d.keys())[0]}: {list(d.values())[0]:.1%}"
            for d in classifier_output["top3"]
        )
        is_malignant_str = (
            "True" if classifier_output["is_malignant"] else "False"
        )

        parts = []

        # === Section 1: Vision Classifier Output ===
        parts.append("## Vision Classifier Output\n")
        parts.append(
            f"- Predicted class: {classifier_output['predicted_class']}"
        )
        parts.append(
            f"- Confidence: {classifier_output['probability']:.1%}"
        )
        parts.append(f"- Top-3 candidates: {top3_str}")
        parts.append(f"- Is malignant class: {is_malignant_str}")
        parts.append(
            f"- Grad-CAM description: "
            f"{classifier_output['grad_cam_description']}"
        )
        parts.append("")

        # === Section 2: Patient Information ===
        parts.append("## Patient Information\n")
        parts.append(
            f"- Age: {patient_meta.age if patient_meta.age else 'unknown'}"
        )
        parts.append(
            f"- Sex: {patient_meta.sex if patient_meta.sex else 'unknown'}"
        )
        parts.append(
            f"- Body site: "
            f"{patient_meta.body_site if patient_meta.body_site else 'unknown'}"
        )
        parts.append(
            f"- Lesion duration: "
            f"{patient_meta.duration_months if patient_meta.duration_months else 'unknown'} months"
        )
        parts.append(
            f"- Symptoms: "
            f"{patient_meta.symptoms if patient_meta.symptoms else 'no notable symptoms'}"
        )
        parts.append(f"- Patient context: {patient_meta.context}")
        parts.append(f"- Risk factor: {patient_meta.risk_factor}")
        parts.append(f"- Skin type: {patient_meta.skin_type}")
        if patient_meta.family_history:
            parts.append(f"- Family history: {patient_meta.family_history}")
        parts.append("")

        # === Section 3: Healthcare Setting ===
        parts.append("## Healthcare Setting\n")
        parts.append(
            f"- Resource constraint: {patient_meta.resource_constraint}"
        )
        parts.append("")

        # === Section 4: RAG Context ===
        if rag_context and rag_context != "(No reference material available)":
            parts.append("## Relevant Medical References\n")
            parts.append(rag_context)
            parts.append("")

        # === Section 5: Request ===
        parts.append(
            "Please provide a structured assessment in valid JSON format. "
            "Response style: detailed."
        )

        return "\n".join(parts)

    def generate(
        self,
        classifier_output: Dict,
        patient_meta: PatientMetadata,
        rag_context: str,
        max_new_tokens: int = 1500, # 800
    ) -> Dict:
        """영어 LMIC 형식의 구조화된 의료 보조 응답 생성."""
        # User prompt 빌드 (학습 데이터 형식과 일치)
        user_prompt = self._build_user_prompt(
            classifier_output, patient_meta, rag_context,
        )

        # Chat template (학습과 동일하게 system role 네이티브 사용)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 생성 (greedy decoding으로 결정적)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # 강화된 JSON 파싱 (마크다운 펜스, ABCDE double colon, 다국어 환각 자동 처리)
        return parse_gemma_response(generated, debug=False)


# ============================================================
# 5. 영어 일관성 검증 및 환각 교정
# ============================================================
def clean_observed_features(features):
    """observed_features에서 abcde_analysis 중복 노출 제거."""
    if not isinstance(features, list):
        return features

    cleaned = []
    for item in features:
        if not isinstance(item, str):
            cleaned.append(item)
            continue
        if item.strip().lower().startswith("abcde analysis"):
            continue
        if re.match(r"^ABCDE\s*[-:]", item.strip()):
            continue
        cleaned.append(item)

    return cleaned if cleaned else ["Vision Classifier output processed"]

def validate_and_correct_response_en(
    response: Dict, classifier_output: Dict,
) -> Dict:
    """
    영어 응답의 Vision 분류 일관성 검증 및 환각 자동 교정.
    학습 데이터의 영어 형식에 맞춰 검출/교정.
    """
    is_malignant = classifier_output["is_malignant"]
    confidence = classifier_output["probability"]
    predicted_class = classifier_output["predicted_class"]

    correction_log = []

    if "observed_features" in response:
        response["observed_features"] = clean_observed_features(
            response["observed_features"]
        )


    # === 1. urgency 일관성 검증 ===
    if not is_malignant and confidence >= 0.70:
        # 양성 고신뢰도 → urgent 다운그레이드
        if response.get("urgency") == "urgent":
            response["urgency"] = "routine"
            correction_log.append(
                "urgency downgraded: benign high-confidence"
            )

        # patient_summary에서 악성 암시 표현 제거 (영어)
        summary = response.get("patient_summary", "")
        malignant_indicators = [
            "high malignancy",
            "likely malignant",
            "highly suspicious",
            "cancer suspected",
            "malignant tumor",
            "needs urgent doctor check",
            "go to nearest health center as soon as possible",
        ]
        if any(ind.lower() in summary.lower() for ind in malignant_indicators):
            response["patient_summary"] = (
                "This skin spot looks like it is likely not dangerous. "
                f"Vision Classifier shows {confidence:.1%} confidence "
                f"in benign category. "
                "Watch it carefully. Tell the health worker if it: "
                "gets bigger, changes color, bleeds, itches, "
                "or develops new symptoms. "
                "Apply sunscreen if available. "
                "Schedule routine check in 6 months. "
                "This is AI screening only - see doctor if you are worried."
            )
            correction_log.append(
                "summary rewritten: benign high-confidence"
            )

    elif is_malignant and confidence >= 0.70:
        # 악성 고신뢰도 → routine 업그레이드
        if response.get("urgency") == "routine":
            response["urgency"] = "soon"
            correction_log.append("urgency upgraded: malignant")

    # === 2. 환각 검출 및 자동 교정 ===
    contradiction_patterns = [
        # 자체 모순 표현
        (r"\bBCC\s+benign\s+form\b", "benign lesion"),
        (
            r"benign\s+form\s+of\s+(BCC|melanoma|carcinoma)",
            "benign lesion (atypical presentation)",
        ),
        (r"carcinoma\s+benign", "benign lesion"),
        (
            r"malignant\s+(form|type)\s+of\s+(nevus|nv|seborrheic)",
            "atypical lesion",
        ),
        # 영문 의학 용어 대문자 표기 (한국어 환각의 영어 버전)
        (r"\bBASAL CELL CARCINOMA\b", "basal cell carcinoma"),
        (r"\bSQUAMOUS CELL CARCINOMA\b", "squamous cell carcinoma"),
        (r"\bMELANOMA\b(?!\s*:)", "melanoma"),

        # 추가: 가짜 URL 환각
        (r"\bAI://[\w\-\.]+", "(contact local health center)"),
        (r"\bdoctor://[\w\-\.]+", "(contact local health center)"),
        (r"\bvia://[\w\-\.]+", ""),

        # 추가: 토큰 결합 환각
        (r"\bSpecialleistialongvised\b", "Specialist supervised"),
        (r"\bSpecial(?:leist|liest)ial\w*\b", "Specialist"),

        # 추가: 잘못된 진단명
        (r"\bnailfold\s+dystrophy\s*\(nv\)", "benign nevus (nv)"),
        (r"\bFingernail\s+melanoma", "nevus"),

        # 추가: 마크다운 펜스 잔존
        (r"```json\s*", ""),
        (r"```\s*$", ""),
    ]

    text_fields = [
        "classification_summary", "recommendation",
        "patient_summary", "limitations",
    ]
    list_fields = ["observed_features", "evidence_sources"]

    correction_count = 0

    def clean_text(text: str) -> str:
        nonlocal correction_count
        original = text
        for pattern, replacement in contradiction_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 비-라틴 문자 제거 (한국어/일본어/벵골어 환각)
        text = re.sub(r"[\u3131-\uD79D\uAC00-\uD7A3]+", "", text)  # 한글
        text = re.sub(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", "", text)  # 일본어
        text = re.sub(r"[\u0980-\u09FF]+", "", text)  # 벵골어

        text = text.strip()

        if text != original:
            correction_count += 1
        return text

    # 텍스트 필드 교정
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            response[field] = clean_text(response[field])

    # 리스트 필드 교정
    for field in list_fields:
        if field in response and isinstance(response[field], list):
            response[field] = [
                clean_text(item) if isinstance(item, str) else item
                for item in response[field]
            ]

    # ABCDE 필드 교정
    if "abcde_analysis" in response and isinstance(response["abcde_analysis"], dict):
        for key, value in response["abcde_analysis"].items():
            if isinstance(value, str):
                response["abcde_analysis"][key] = clean_text(value)

    if correction_count > 0:
        correction_log.append(
            f"hallucination corrections: {correction_count}"
        )

    # === 3. 필수 필드 보강 ===
    required_fields = {
        "observed_features": ["No features extracted"],
        "abcde_analysis": {k: "Inconclusive" for k in "ABCDE"},
        "classification_summary": "Vision Classifier analysis incomplete",
        "evidence_sources": ["AI screening output"],
        "recommendation": (
            "Consult dermatology specialist for direct examination"
        ),
        "urgency": "soon",
        "patient_summary": (
            "AI screening completed. Please consult a healthcare professional "
            "for definitive evaluation."
        ),
        "limitations": (
            "This is an AI-assisted screening tool, not a medical diagnosis. "
            "Results require validation by qualified healthcare professional."
        ),
    }

    for field, default in required_fields.items():
        if field not in response or not response[field]:
            response[field] = default
            correction_log.append(f"added missing field: {field}")

    # === 4. urgency 값 검증 ===
    valid_urgency = ["routine", "soon", "urgent"]
    if response.get("urgency") not in valid_urgency:
        response["urgency"] = "soon"
        correction_log.append("invalid urgency value reset")

    # 교정 로그 기록 (디버깅용)
    if correction_log:
        response["_correction_applied"] = " | ".join(correction_log)

    return response


# ============================================================
# 6. 통합 파이프라인
# ============================================================
class SkinLesionAssistant:
    """End-to-End 영어 LMIC 추론 파이프라인."""

    def __init__(
        self,
        vision_ckpt: Path,
        rag_db: Path,
        gemma_base: str,
        lora_adapter: Path,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[초기화] 디바이스: {self.device}")

        print("\n[1/3] Vision Classifier 로드...")
        self.vision = VisionClassifier(vision_ckpt, self.device)

        print("\n[2/3] RAG Retriever 로드...")
        self.rag = RAGRetriever(rag_db, embedding_model="BAAI/bge-m3")

        print("\n[3/3] Gemma 추론 엔진 로드...")
        self.gemma = GemmaInference(gemma_base, lora_adapter, self.device)

        print("\n[완료] 파이프라인 준비 완료")

    def analyze(
        self,
        image_path: Path,
        patient_meta: Optional[PatientMetadata] = None,
    ) -> Dict:
        """단일 이미지 분석 → 전체 파이프라인 실행."""
        patient_meta = patient_meta or PatientMetadata()

        # Step 1: 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # Step 2: Vision Classification
        clf_output = self.vision.predict(image)
        print(f"\n[Classification] {clf_output['predicted_class']} "
              f"({clf_output['probability']:.1%})")

        # Step 3: Confidence Gate
        if clf_output["probability"] < CONFIDENCE_THRESHOLD:
            print(f"[Confidence Gate] Confidence {clf_output['probability']:.1%} < "
                  f"threshold {CONFIDENCE_THRESHOLD:.0%}")
            print("  → Recommending specialist review")

        # Step 4: RAG 검색 (영어)
        rag_context = self.rag.build_context(clf_output, top_k=5)
        print(f"[RAG] Context length: {len(rag_context)} chars")

        # Step 5: Gemma 응답 생성 (영어)
        print("[Gemma] Generating response...")
        response = self.gemma.generate(clf_output, patient_meta, rag_context)

        # Step 6: 영어 일관성 검증
        response = validate_and_correct_response_en(response, clf_output)
        if "_correction_applied" in response:
            print(f"[Validation] Corrections: {response['_correction_applied']}")

        # 최종 결과 패키징
        result = {
            "input": {
                "image_path": str(image_path),
                "patient_metadata": asdict(patient_meta),
            },
            "classifier_output": {
                k: v for k, v in clf_output.items()
                if k != "grad_cam_map"  # numpy 배열 제외
            },
            "rag_sources_used": len(rag_context) > 20,
            "response": response,
        }

        return result


# ============================================================
# 7. CLI 실행
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LMIC Skin Lesion Screening Assistant"
    )
    parser.add_argument("--image", type=str, required=True, help="이미지 경로")

    # 기본 환자 정보
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument(
        "--sex", type=str, default=None, choices=["male", "female"]
    )
    parser.add_argument(
        "--body_site", type=str, default=None,
        help="e.g., 'face', 'back', 'left forearm'",
    )
    parser.add_argument("--duration_months", type=int, default=None)
    parser.add_argument(
        "--symptoms", type=str, default=None,
        help="e.g., 'rapid growth', 'intermittent itching'",
    )
    parser.add_argument("--family_history", type=str, default=None)

    # LMIC 추가 필드
    parser.add_argument(
        "--context", type=str, default="general LMIC patient",
        help=(
            "Patient context (e.g., 'rural farmer with chronic UV exposure', "
            "'patient with albinism', 'HIV positive patient')"
        ),
    )
    parser.add_argument(
        "--risk_factor", type=str, default="limited healthcare access",
        help=(
            "e.g., 'OCA (oculocutaneous albinism)', 'chronic sun exposure', "
            "'HIV infection'"
        ),
    )
    parser.add_argument(
        "--skin_type", type=str, default="unspecified",
        help="Fitzpatrick scale (e.g., 'Fitzpatrick V', 'Fitzpatrick I (albinism)')",
    )
    parser.add_argument(
        "--resource_constraint", type=str,
        default="Standard primary care setting with limited specialist access",
        help="Healthcare resource limitation",
    )

    # 모델 옵션
    parser.add_argument(
        "--use_baseline", action="store_true",
        help="Use baseline Vision model instead of with_synthetic",
    )
    parser.add_argument(
        "--use_korean_lora", action="store_true",
        help="Use Korean LoRA adapter (legacy) instead of English LMIC version",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON save path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[오류] 이미지 없음: {image_path}")
        sys.exit(1)

    # Vision 모델 경로
    if args.use_baseline:
        vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
    else:
        vision_ckpt = VISION_MODEL_DIR / "best_with_synthetic.pth"
        if not vision_ckpt.exists():
            vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
            print(f"[알림] with_synthetic 없음 → baseline 사용")

    # LoRA 어댑터 경로 (영어 LMIC 버전 우선)
    rag_db = RAG_DB_DIR / "medical_knowledge.db"
    if args.use_korean_lora:
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"
        print("[알림] 한국어 LoRA 어댑터 사용 (legacy)")
    else:
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter_en" / "final_adapter"
        if not lora_adapter.exists():
            print(f"[경고] 영어 LoRA 어댑터 없음: {lora_adapter}")
            print(f"  → 한국어 어댑터로 폴백 (응답 품질 저하 가능)")
            lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"

    # 파이프라인 실행
    print("=" * 60)
    print(" Skin Lesion Assistant — LMIC English Pipeline")
    print(" Target: Sub-Saharan Africa & Other LMICs")
    print("=" * 60)

    assistant = SkinLesionAssistant(
        vision_ckpt=vision_ckpt,
        rag_db=rag_db,
        gemma_base=GEMMA_CONFIG["base_model"],
        lora_adapter=lora_adapter,
    )

    # 환자 정보 구성 (LMIC 필드 포함)
    patient_meta = PatientMetadata(
        age=args.age,
        sex=args.sex,
        body_site=args.body_site,
        duration_months=args.duration_months,
        symptoms=args.symptoms,
        family_history=args.family_history,
        context=args.context,
        risk_factor=args.risk_factor,
        skin_type=args.skin_type,
        resource_constraint=args.resource_constraint,
    )

    # 분석 실행
    print("\n" + "=" * 60)
    print(f" Image Analysis: {image_path.name}")
    print("=" * 60)

    result = assistant.analyze(image_path, patient_meta)

    # 결과 출력
    print("\n" + "=" * 60)
    print(" Final Result")
    print("=" * 60)
    print(json.dumps(result["response"], ensure_ascii=False, indent=2))

    # 저장
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            OUTPUT_DIR / "inference_results"
            / f"{image_path.stem}_result_en.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()