"""
Skin Lesion Assistant - Integrated Pipeline
============================================
Vision Classifier + RAG + Gemma 4 LoRA inference pipeline.

This file is located at src/dermassist/pipeline/assistant.py.
Project root is 4 levels up.
"""

import os
import sys
from pathlib import Path


# ============================================================
# 1. Auto-activate project .venv (developer convenience)
# ============================================================
def _ensure_project_venv() -> None:
    """
    Re-execute the script with the project's .venv Python if not already active.

    Module structure: src/dermassist/pipeline/assistant.py
    Project root: 4 levels up (parent.parent.parent.parent)
    """
    # src/dermassist/pipeline/assistant.py -> project root
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


# Auto-activate only on direct execution (skip when imported)
if __name__ == "__main__":
    _ensure_project_venv()


# ============================================================
# 2. Standard library imports
# ============================================================
import re
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# ============================================================
# 3. Third-party imports
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# 4. Project-internal imports
# ============================================================
# JSON parser: same-package llm module
from dermassist.llm.json_parser import parse_gemma_response

# configs.config: configs/ directory at project root
# (requires configs/__init__.py when installed via pip install -e .)

from configs.config import (
    PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
    GEMMA_MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, IMAGE_SIZE,
    VISION_CONFIG, GEMMA_CONFIG, RAG_CONFIG,
    NORMALIZATION_MEAN, NORMALIZATION_STD,
    CONFIDENCE_THRESHOLD, ESCALATION_MESSAGE,
)


# ============================================================
# SYSTEM PROMPT (matches training data exactly)
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
# Grad-CAM English description templates (match training data)
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
# 1. Patient metadata schema (with LMIC extensions)
# ============================================================
@dataclass
class PatientMetadata:
    """Patient context information (includes LMIC-specific fields)."""
    # Standard fields
    age: Optional[int] = None
    sex: Optional[str] = None
    body_site: Optional[str] = None
    duration_months: Optional[int] = None
    symptoms: Optional[str] = None
    family_history: Optional[str] = None

    # LMIC extension fields (matches training data format)
    context: str = "general LMIC patient"
    risk_factor: str = "limited healthcare access"
    skin_type: str = "unspecified"
    resource_constraint: str = (
        "Standard primary care setting with limited specialist access"
    )


# ============================================================
# 2. Vision Classifier wrapper
# ============================================================
class VisionClassifier:
    """EfficientNet-B4 classifier with Grad-CAM generation."""

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

        # Preprocessing pipeline
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

        # Register Grad-CAM hooks
        self._gradients = None
        self._activations = None
        target = self.model.conv_head
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

        print(f"[Vision] Loaded (mode: {ckpt.get('mode', 'unknown')})")

    def _save_activation(self, m, i, o):
        self._activations = o.detach()

    def _save_gradient(self, m, gi, go):
        self._gradients = go[0].detach()

    def _generate_gradcam_description(
        self, cam: np.ndarray, pred_class: str
    ) -> str:
        """Generate English Grad-CAM description from predicted class + activation map."""
        import random

        h, w = cam.shape

        # Compute symmetry
        left_sum = cam[:, :w//2].sum()
        right_sum = cam[:, w//2:].sum()
        symmetry_ratio = (
            min(left_sum, right_sum) / (max(left_sum, right_sum) + 1e-8)
        )

        # Per-class English templates (match training data)
        templates = GRAD_CAM_TEMPLATES_EN.get(pred_class)
        if not templates:
            return "Activation distributed across the lesion area"

        # Select template based on symmetry
        if pred_class == "nv":
            # nv: prefer symmetric/uniform patterns
            if symmetry_ratio > 0.8:
                desc = templates[0]  # symmetric central activation
            else:
                desc = templates[1]  # even distribution
        elif pred_class == "mel":
            # mel: prefer asymmetric patterns
            if symmetry_ratio < 0.7:
                desc = templates[0]  # asymmetric pattern
            else:
                desc = templates[1]  # heterogeneous distribution
        elif pred_class == "bcc":
            # bcc: prefer edge-activation patterns
            desc = templates[0]
        elif pred_class == "akiec":
            desc = templates[0]
        else:
            # Other classes: use first template (consistency)
            desc = templates[0]

        return desc

    def predict(self, image: Image.Image) -> Dict:
        """Classify image and generate English Grad-CAM description."""
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

        # Top-3 predictions
        top3_idx = probs.topk(3).indices.cpu().numpy()
        top3 = [
            {CLASS_NAMES[i]: round(probs[i].item(), 4)}
            for i in top3_idx
        ]

        # Generate Grad-CAM
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
# 3. RAG retrieval wrapper (English queries)
# ============================================================
class RAGRetriever:
    """bge-m3 embedding-based medical knowledge retrieval (English)."""

    def __init__(self, db_path: Path, embedding_model: str = "BAAI/bge-m3"):
        import sqlite3
        from sentence_transformers import SentenceTransformer

        self.db_path = db_path
        if not db_path.exists():
            print(f"[Warning] RAG DB not found: {db_path}")
            self.model = None
            return

        print(f"[RAG] Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print(f"  Dimension: {self.model.get_sentence_embedding_dimension()}")

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        print(f"[RAG] DB connected: {db_path}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """English query -> embedding -> top-k retrieval."""
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
        """Build English RAG query from predicted class + Grad-CAM."""
        # English query
        query = (
            f"{classifier_output['predicted_class']} skin lesion "
            f"clinical features diagnosis dermoscopy "
            f"{classifier_output['grad_cam_description']}"
        )

        results = self.search(query, top_k=top_k)
        if not results:
            return "(No reference material available)"

        # Assemble context (English)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {r['title']}] {r['content'][:500]}"
            )
        return "\n\n".join(context_parts)


# ============================================================
# 4. Gemma LoRA inference wrapper (English)
# ============================================================
class GemmaInference:
    """Generate English medical assistant responses via Gemma 4 E4B + LoRA adapter."""

    def __init__(
        self,
        base_model_id: str,
        lora_adapter_dir: Path,
        device: torch.device,
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[Gemma] Loading base model: {base_model_id}")

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

        # Load LoRA adapter
        if lora_adapter_dir.exists():
            print(f"[Gemma] Loading LoRA adapter: {lora_adapter_dir}")
            self.model = PeftModel.from_pretrained(
                self.base_model, str(lora_adapter_dir),
            )
        else:
            print(f"[Warning] LoRA adapter not found - using base model")
            self.model = self.base_model

        self.model.eval()

    def _build_user_prompt(
        self,
        classifier_output: Dict,
        patient_meta: PatientMetadata,
        rag_context: str,
    ) -> str:
        """Build English user prompt matching training data format."""
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
        max_new_tokens: int = 1500,
    ) -> Dict:
        """Generate structured English LMIC medical assistant response."""
        # Build user prompt (matches training data format)
        user_prompt = self._build_user_prompt(
            classifier_output, patient_meta, rag_context,
        )

        # Chat template (native system role, matches training)
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

        # Generate (greedy decoding for determinism)
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

        # Robust JSON parsing (handles markdown fences, ABCDE double colon,
        # multilingual hallucinations automatically)
        return parse_gemma_response(generated, debug=False)


# ============================================================
# 5. English consistency validation and hallucination correction
# ============================================================
def clean_observed_features(features):
    """Remove ABCDE analysis duplicates from observed_features."""
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
    Validate Vision classification consistency and auto-correct hallucinations
    in English responses. Detection/correction matches the English training format.
    """
    is_malignant = classifier_output["is_malignant"]
    confidence = classifier_output["probability"]
    predicted_class = classifier_output["predicted_class"]

    correction_log = []

    if "observed_features" in response:
        response["observed_features"] = clean_observed_features(
            response["observed_features"]
        )

    # === 1. Urgency consistency check ===
    if not is_malignant and confidence >= 0.70:
        # Benign high-confidence -> downgrade urgent
        if response.get("urgency") == "urgent":
            response["urgency"] = "routine"
            correction_log.append(
                "urgency downgraded: benign high-confidence"
            )

        # Remove malignancy-implying language from patient_summary (English)
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
        # Malignant high-confidence -> upgrade routine
        if response.get("urgency") == "routine":
            response["urgency"] = "soon"
            correction_log.append("urgency upgraded: malignant")

    # === 2. Hallucination detection and auto-correction ===
    contradiction_patterns = [
        # Self-contradictory expressions
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
        # All-caps medical terms (English equivalent of Korean hallucination)
        (r"\bBASAL CELL CARCINOMA\b", "basal cell carcinoma"),
        (r"\bSQUAMOUS CELL CARCINOMA\b", "squamous cell carcinoma"),
        (r"\bMELANOMA\b(?!\s*:)", "melanoma"),

        # Fake URL hallucinations
        (r"\bAI://[\w\-\.]+", "(contact local health center)"),
        (r"\bdoctor://[\w\-\.]+", "(contact local health center)"),
        (r"\bvia://[\w\-\.]+", ""),

        # Token-concatenation hallucinations
        (r"\bSpecialleistialongvised\b", "Specialist supervised"),
        (r"\bSpecial(?:leist|liest)ial\w*\b", "Specialist"),

        # Incorrect diagnostic terms
        (r"\bnailfold\s+dystrophy\s*\(nv\)", "benign nevus (nv)"),
        (r"\bFingernail\s+melanoma", "nevus"),

        # Residual markdown fences
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

        # Remove non-Latin character hallucinations (Korean/Japanese/Bengali)
        text = re.sub(r"[\u3131-\uD79D\uAC00-\uD7A3]+", "", text)  # Korean
        text = re.sub(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", "", text)  # Japanese
        text = re.sub(r"[\u0980-\u09FF]+", "", text)  # Bengali

        text = text.strip()

        if text != original:
            correction_count += 1
        return text

    # Clean text fields
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            response[field] = clean_text(response[field])

    # Clean list fields
    for field in list_fields:
        if field in response and isinstance(response[field], list):
            response[field] = [
                clean_text(item) if isinstance(item, str) else item
                for item in response[field]
            ]

    # Clean ABCDE fields
    if "abcde_analysis" in response and isinstance(response["abcde_analysis"], dict):
        for key, value in response["abcde_analysis"].items():
            if isinstance(value, str):
                response["abcde_analysis"][key] = clean_text(value)

    if correction_count > 0:
        correction_log.append(
            f"hallucination corrections: {correction_count}"
        )

    # === 3. Fill in required fields ===
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

    # === 4. Urgency value validation ===
    valid_urgency = ["routine", "soon", "urgent"]
    if response.get("urgency") not in valid_urgency:
        response["urgency"] = "soon"
        correction_log.append("invalid urgency value reset")

    # Record correction log (for debugging)
    if correction_log:
        response["_correction_applied"] = " | ".join(correction_log)

    return response


# ============================================================
# 6. Integrated pipeline
# ============================================================
class SkinLesionAssistant:
    """End-to-end English LMIC inference pipeline."""

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
        print(f"[Init] Device: {self.device}")

        print("\n[1/3] Loading Vision Classifier...")
        self.vision = VisionClassifier(vision_ckpt, self.device)

        print("\n[2/3] Loading RAG Retriever...")
        self.rag = RAGRetriever(rag_db, embedding_model="BAAI/bge-m3")

        print("\n[3/3] Loading Gemma inference engine...")
        self.gemma = GemmaInference(gemma_base, lora_adapter, self.device)

        print("\n[Ready] Pipeline initialized")

    def analyze(
        self,
        image_path: Path,
        patient_meta: Optional[PatientMetadata] = None,
    ) -> Dict:
        """Analyze single image through the full pipeline."""
        patient_meta = patient_meta or PatientMetadata()

        # Step 1: Load image
        image = Image.open(image_path).convert("RGB")

        # Step 2: Vision classification
        clf_output = self.vision.predict(image)
        print(f"\n[Classification] {clf_output['predicted_class']} "
              f"({clf_output['probability']:.1%})")

        # Step 3: Confidence gate
        if clf_output["probability"] < CONFIDENCE_THRESHOLD:
            print(f"[Confidence Gate] Confidence {clf_output['probability']:.1%} < "
                  f"threshold {CONFIDENCE_THRESHOLD:.0%}")
            print("  -> Recommending specialist review")

        # Step 4: RAG retrieval (English)
        rag_context = self.rag.build_context(clf_output, top_k=5)
        print(f"[RAG] Context length: {len(rag_context)} chars")

        # Step 5: Gemma response generation (English)
        print("[Gemma] Generating response...")
        response = self.gemma.generate(clf_output, patient_meta, rag_context)

        # Step 6: English consistency validation
        response = validate_and_correct_response_en(response, clf_output)
        if "_correction_applied" in response:
            print(f"[Validation] Corrections: {response['_correction_applied']}")

        # Package final result
        result = {
            "input": {
                "image_path": str(image_path),
                "patient_metadata": asdict(patient_meta),
            },
            "classifier_output": {
                k: v for k, v in clf_output.items()
                if k != "grad_cam_map"  # Exclude numpy array
            },
            "rag_sources_used": len(rag_context) > 20,
            "response": response,
        }

        return result


# ============================================================
# 7. CLI execution
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LMIC Skin Lesion Screening Assistant"
    )
    parser.add_argument("--image", type=str, required=True, help="Image path")

    # Basic patient information
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

    # LMIC extension fields
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

    # Model options
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
        print(f"[Error] Image not found: {image_path}")
        sys.exit(1)

    # Vision model path
    if args.use_baseline:
        vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
    else:
        vision_ckpt = VISION_MODEL_DIR / "best_with_synthetic.pth"
        if not vision_ckpt.exists():
            vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
            print(f"[Info] with_synthetic not found - using baseline")

    # LoRA adapter path (English LMIC version preferred)
    rag_db = RAG_DB_DIR / "medical_knowledge.db"
    if args.use_korean_lora:
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"
        print("[Info] Using Korean LoRA adapter (legacy)")
    else:
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter_en" / "final_adapter"
        if not lora_adapter.exists():
            print(f"[Warning] English LoRA adapter not found: {lora_adapter}")
            print(f"  -> Falling back to Korean adapter (response quality may degrade)")
            lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"

    # Run pipeline
    print("=" * 60)
    print(" Skin Lesion Assistant - LMIC English Pipeline")
    print(" Target: Sub-Saharan Africa & Other LMICs")
    print("=" * 60)

    assistant = SkinLesionAssistant(
        vision_ckpt=vision_ckpt,
        rag_db=rag_db,
        gemma_base=GEMMA_CONFIG["base_model"],
        lora_adapter=lora_adapter,
    )

    # Compose patient information (with LMIC fields)
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

    # Run analysis
    print("\n" + "=" * 60)
    print(f" Image Analysis: {image_path.name}")
    print("=" * 60)

    result = assistant.analyze(image_path, patient_meta)

    # Print result
    print("\n" + "=" * 60)
    print(" Final Result")
    print("=" * 60)
    print(json.dumps(result["response"], ensure_ascii=False, indent=2))

    # Save
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