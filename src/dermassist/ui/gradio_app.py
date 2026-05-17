"""
gradio_app.py
=============
DermAssist LMIC Gradio UI.

Location: src/dermassist/ui/gradio_app.py

Features:
1. DullRazor hair removal preprocessing
2. UI layout: Original -> Preprocessed -> Grad-CAM (3 columns)
3. Noto Sans font applied via base64 CSS embedding

Requirements:
  - fonts/ directory at project root
  - opencv-python (cv2)
  - pip install -e . completed

Run:
  python scripts/06_run_demo.py
"""

import os
import sys
import json
import socket
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ============================================================
# Project-internal imports (new module structure)
# ============================================================
# configs.config: configs/ directory at project root
try:
    from configs.config import (
        PROCESSED_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR, GEMMA_CONFIG,
        CLASS_NAMES, MALIGNANT_CLASSES, IMAGE_SIZE,
        CONFIDENCE_THRESHOLD,
    )
except ImportError:
    # If configs not found, add project root to sys.path
    # src/dermassist/ui/gradio_app.py -> project root (4 levels up)
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from configs.config import (
        PROCESSED_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR, GEMMA_CONFIG,
        CLASS_NAMES, MALIGNANT_CLASSES, IMAGE_SIZE,
        CONFIDENCE_THRESHOLD,
    )

# Pipeline (same package)
from dermassist.pipeline.assistant import (
    SkinLesionAssistant,
    PatientMetadata,
    validate_and_correct_response_en,
)

# Font loader (same ui package)
from dermassist.ui.font_loader import build_font_css


# ============================================================
# Font family constants
# ============================================================
FONT_FAMILY_SANS = "'Noto Sans', system-ui, -apple-system, sans-serif"
FONT_FAMILY_MONO = "'Consolas', 'Monaco', 'Courier New', monospace"


# ============================================================
# 1. Offline Detection
# ============================================================
def is_truly_offline() -> bool:
    """Check if the system has internet connectivity (DNS test)."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return False
    except OSError:
        return True


def get_network_status_html() -> str:
    """Render network status badge (offline/online)."""
    if is_truly_offline():
        return f"""
        <div style="
            background: #16a34a; color: white;
            padding: 8px 16px; border-radius: 8px;
            display: inline-block; font-weight: 600;
            font-size: 14px;
            font-family: {FONT_FAMILY_SANS};">
            🔒 OFFLINE MODE — No data transmitted
        </div>
        """
    else:
        return f"""
        <div style="
            background: #ea580c; color: white;
            padding: 8px 16px; border-radius: 8px;
            display: inline-block; font-weight: 600;
            font-size: 14px;
            font-family: {FONT_FAMILY_SANS};">
            ⚠ ONLINE — For demo: disconnect WiFi to verify offline operation
        </div>
        """


# ============================================================
# 2. Patient Profile Mapping
# ============================================================
PROFILE_MAPPING = {
    "General LMIC patient": {
        "context": "general LMIC patient",
        "risk_factor": "limited healthcare access",
        "skin_type": "Fitzpatrick IV",
    },
    "Rural farmer (chronic UV exposure)": {
        "context": "rural farmer with chronic UV exposure",
        "risk_factor": "chronic UV exposure",
        "skin_type": "Fitzpatrick V",
    },
    "Patient with albinism (high cancer risk)": {
        "context": "patient with albinism, multiple sun-damaged areas",
        "risk_factor": "OCA (oculocutaneous albinism)",
        "skin_type": "Fitzpatrick I (albinism)",
    },
    "Child with albinism (early sun damage)": {
        "context": "pediatric albinism patient with early sun damage",
        "risk_factor": "OCA, limited sun protection access",
        "skin_type": "Fitzpatrick I (albinism)",
    },
    "HIV positive (immunocompromised)": {
        "context": "immunocompromised patient on ART",
        "risk_factor": "HIV infection",
        "skin_type": "Fitzpatrick IV",
    },
    "Outdoor laborer": {
        "context": "manual laborer with prolonged sun exposure",
        "risk_factor": "occupational sun exposure",
        "skin_type": "Fitzpatrick IV",
    },
    "Elderly rural": {
        "context": "elderly patient with cumulative sun damage",
        "risk_factor": "cumulative lifetime UV exposure",
        "skin_type": "Fitzpatrick IV",
    },
    "Remote area (no specialist access)": {
        "context": "patient in remote area with no specialist access",
        "risk_factor": "limited healthcare access",
        "skin_type": "Fitzpatrick V",
    },
}

RESOURCE_CONSTRAINT_OPTIONS = {
    "Standard primary care (specialist available)":
        "Standard primary care setting with limited specialist access",
    "Long distance (200+ km to specialist)":
        "Patient travel to nearest dermatologist requires 200+ km journey",
    "No biopsy available locally":
        "No biopsy facilities in this primary care setting",
    "Long wait time (4-8 weeks)":
        "Specialist referral wait time typically 4-8 weeks",
    "Teledermatology available":
        "Teledermatology service available via African Teledermatology Project",
}

BODY_SITES = [
    "face", "scalp", "neck", "chest", "back", "abdomen",
    "left forearm (sun-exposed)", "right forearm (sun-exposed)",
    "left lower leg", "right lower leg",
    "hand dorsum", "foot sole", "ear", "lip",
]

SYMPTOMS_OPTIONS = [
    "asymptomatic (no notable symptoms)",
    "intermittent itching",
    "occasional bleeding when scratched",
    "rapid growth over past 3 months",
    "color change reported by patient",
    "tender to palpation",
    "non-healing ulceration",
    "scaly, rough surface",
    "raised firm nodule",
    "size doubled in 6 months",
]


# ============================================================
# 3. DullRazor Hair Removal Preprocessing
# ============================================================
def dullrazor_preprocess(image_pil: Image.Image) -> Image.Image:
    """
    DullRazor hair-removal preprocessing for dermoscopy images.

    Algorithm:
      1. Grayscale conversion
      2. Black-hat morphology to detect dark linear structures (hair)
      3. Threshold to create hair mask
      4. cv2.inpaint to interpolate from surrounding pixels

    Reference:
        Lee et al. (1997) "DullRazor: A software approach to hair
        removal from images" - standard medical imaging preprocessing.
    """
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


# ============================================================
# 4. Grad-CAM Overlay Generator
# ============================================================
def create_gradcam_overlay(
    original_image: Image.Image,
    cam_map: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Create Grad-CAM heatmap overlay on the original image."""
    img_array = np.array(original_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))

    cam_min, cam_max = cam_map.min(), cam_map.max()
    if cam_max > cam_min:
        cam_normalized = (cam_map - cam_min) / (cam_max - cam_min)
    else:
        cam_normalized = np.zeros_like(cam_map)
    cam_uint8 = (cam_normalized * 255).astype(np.uint8)

    if cam_uint8.shape != (IMAGE_SIZE, IMAGE_SIZE):
        cam_uint8 = cv2.resize(cam_uint8, (IMAGE_SIZE, IMAGE_SIZE))

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (heatmap * alpha + img_array * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


# ============================================================
# 5. Result Formatting (Noto Sans applied)
# ============================================================
URGENCY_BADGES = {
    "routine": ("🟢 Routine", "#16a34a", "Routine follow-up — no urgent action needed"),
    "soon": ("🟡 Soon", "#ea580c", "Schedule specialist consultation within 4-8 weeks"),
    "urgent": ("🔴 Urgent", "#dc2626", "Urgent specialist referral needed"),
}


def format_recommendation_html(response: Dict, classifier_output: Dict) -> str:
    """Render urgency badge + clinical recommendation as HTML."""
    urgency = response.get("urgency", "soon")
    badge_text, badge_color, badge_subtitle = URGENCY_BADGES.get(urgency, URGENCY_BADGES["soon"])

    recommendation = response.get("recommendation", "No recommendation available")
    predicted_class = classifier_output.get("predicted_class", "unknown")
    confidence = classifier_output.get("probability", 0.0)
    is_malignant = classifier_output.get("is_malignant", False)

    cls_label = "Malignant category" if is_malignant else "Benign category"
    cls_color = "#dc2626" if is_malignant else "#16a34a"

    return f"""
    <div style="font-family: {FONT_FAMILY_SANS};">
      <div style="
          background: {badge_color};
          color: white;
          padding: 24px;
          border-radius: 12px;
          margin-bottom: 16px;
          text-align: center;">
        <div style="font-size: 36px; font-weight: 700; margin-bottom: 8px;">
          {badge_text}
        </div>
        <div style="font-size: 16px; opacity: 0.95; font-weight: 400;">
          {badge_subtitle}
        </div>
      </div>

      <div style="
          background: #f8fafc;
          border-left: 4px solid {badge_color};
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 16px;">
        <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;
                    text-transform: uppercase; letter-spacing: 0.5px;
                    font-weight: 500;">
          Clinical Recommendation
        </div>
        <div style="font-size: 16px; line-height: 1.6; color: #1e293b;
                    font-weight: 400;">
          {recommendation}
        </div>
      </div>

      <div style="
          font-size: 12px;
          color: #94a3b8;
          padding: 12px;
          background: #f1f5f9;
          border-radius: 6px;
          font-family: {FONT_FAMILY_MONO};">
        <span style="color: {cls_color}; font-weight: 700;">{cls_label}</span>
        &nbsp;•&nbsp; Vision Classifier: {predicted_class} ({confidence:.1%})
        &nbsp;•&nbsp; AI screening — specialist confirmation required
      </div>
    </div>
    """


def format_patient_summary_html(response: Dict) -> str:
    """Render plain-language patient summary as HTML."""
    summary = response.get("patient_summary", "No patient summary available")
    return f"""
    <div style="
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 20px;
        border-radius: 8px;
        font-family: {FONT_FAMILY_SANS};
        font-size: 15px;
        line-height: 1.7;
        color: #422006;
        font-weight: 400;">
      <div style="font-size: 13px; color: #92400e; margin-bottom: 12px;
                  text-transform: uppercase; letter-spacing: 0.5px;
                  font-weight: 700;">
        For Patient
      </div>
      {summary}
    </div>
    """


def format_visual_analysis_html(classifier_output: Dict, response: Dict) -> str:
    """Render Grad-CAM description, ABCDE table, and Top-3 candidates as HTML."""
    abcde = response.get("abcde_analysis", {})
    grad_cam = classifier_output.get("grad_cam_description", "")
    top3 = classifier_output.get("top3", [])

    abcde_rows = "".join([
        f"<tr><td style='padding:8px;font-weight:700;width:80px;'>{k}</td>"
        f"<td style='padding:8px;color:#475569;font-weight:400;'>{v}</td></tr>"
        for k, v in abcde.items()
    ])

    top3_text = ", ".join(
        f"{list(d.keys())[0]}: {list(d.values())[0]:.1%}"
        for d in top3
    )

    return f"""
    <div style="font-family: {FONT_FAMILY_SANS};">
      <div style="
          background: #f0f9ff;
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 16px;">
        <div style="font-size: 13px; color: #0369a1; font-weight: 700;
                    text-transform: uppercase; margin-bottom: 8px;">
          Grad-CAM Visual Analysis
        </div>
        <div style="font-size: 14px; color: #1e293b; font-weight: 400;">
          {grad_cam}
        </div>
      </div>

      <table style="width:100%; border-collapse:collapse; margin-bottom: 16px;">
        <thead>
          <tr style="background:#f1f5f9;">
            <th colspan="2" style="padding:12px; text-align:left; font-size:13px;
                color:#475569; text-transform:uppercase; font-weight: 700;">
              ABCDE Analysis
            </th>
          </tr>
        </thead>
        <tbody>
          {abcde_rows}
        </tbody>
      </table>

      <div style="font-size: 12px; color: #94a3b8; padding: 12px;
                  background: #f8fafc; border-radius: 6px;
                  font-family: {FONT_FAMILY_MONO};">
        Top-3 candidates: {top3_text}
      </div>
    </div>
    """


def format_limitations_html(response: Dict) -> str:
    """Render AI limitations and evidence sources as HTML."""
    limitations = response.get("limitations", "")
    sources = response.get("evidence_sources", [])

    sources_html = "".join([
        f"<li style='margin-bottom:6px; color:#1e293b; font-weight:400;'>{s}</li>"
        for s in sources
    ])

    return f"""
    <div style="font-family: {FONT_FAMILY_SANS};">
      <div style="
          background: #fef2f2;
          border-left: 4px solid #dc2626;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 16px;">
        <div style="font-size: 13px; color: #991b1b; margin-bottom: 12px;
                    text-transform: uppercase; letter-spacing: 0.5px;
                    font-weight: 700;">
          ⚠ AI Limitations & Disclaimers
        </div>
        <div style="font-size: 14px; line-height: 1.7; color: #450a0a;
                    font-weight: 400;">
          {limitations}
        </div>
      </div>

      <div style="background:#ffffff; padding:16px; border-radius:8px;
                  border:1px solid #e2e8f0;">
        <div style="font-size:13px; color:#1e293b; margin-bottom:12px;
                    font-weight:700; text-transform:uppercase;">
          Evidence Sources
        </div>
        <ul style="margin:0; padding-left:20px; font-size:13px;
                   color:#1e293b; line-height:1.6;">
          {sources_html}
        </ul>
      </div>
    </div>
    """


# ============================================================
# 6. Pipeline Manager
# ============================================================
class PipelineManager:
    """Pre-warmed pipeline manager loaded in background thread."""

    def __init__(self):
        self.assistant: Optional[SkinLesionAssistant] = None
        self.is_loading = False
        self.load_progress = 0
        self.load_message = "Not loaded"

    def prewarm(self):
        """Pre-load all pipeline components in the background."""
        if self.assistant is not None or self.is_loading:
            return

        self.is_loading = True
        try:
            print("[Pre-warm] Loading Vision Classifier...")
            self.load_message = "Loading Vision Classifier..."
            self.load_progress = 10

            vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"

            print("[Pre-warm] Loading RAG embeddings...")
            self.load_message = "Loading RAG embeddings (BAAI/bge-m3)..."
            self.load_progress = 30

            rag_db = RAG_DB_DIR / "medical_knowledge.db"

            print("[Pre-warm] Loading Gemma 4 E4B + LoRA...")
            self.load_message = "Loading Gemma 4 E4B (4-bit quantization) + LoRA..."
            self.load_progress = 60

            lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"

            self.assistant = SkinLesionAssistant(
                vision_ckpt=vision_ckpt,
                rag_db=rag_db,
                gemma_base=GEMMA_CONFIG["base_model"],
                lora_adapter=lora_adapter,
            )

            self.load_progress = 100
            self.load_message = "Pipeline ready"
            print("[Pre-warm] Complete")

        except Exception as e:
            self.load_message = f"Load failed: {e}"
            print(f"[Pre-warm] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_loading = False


pipeline_mgr = PipelineManager()


def get_pipeline_status_html() -> str:
    """Render pipeline loading status badge."""
    if pipeline_mgr.assistant is not None:
        return f"""
        <div style="background:#16a34a; color:white; padding:8px 16px;
                    border-radius:6px; display:inline-block; font-size:13px;
                    font-weight:600; font-family:{FONT_FAMILY_SANS};">
          ✓ Pipeline Ready
        </div>
        """
    elif pipeline_mgr.is_loading:
        return f"""
        <div style="background:#ea580c; color:white; padding:8px 16px;
                    border-radius:6px; display:inline-block; font-size:13px;
                    font-weight:600; font-family:{FONT_FAMILY_SANS};">
          ⏳ {pipeline_mgr.load_message} ({pipeline_mgr.load_progress}%)
        </div>
        """
    else:
        return f"""
        <div style="background:#94a3b8; color:white; padding:8px 16px;
                    border-radius:6px; display:inline-block; font-size:13px;
                    font-weight:600; font-family:{FONT_FAMILY_SANS};">
          ⚪ Pipeline Not Loaded
        </div>
        """


# ============================================================
# 7. Main Inference Function (includes DullRazor preprocessing)
# ============================================================
def run_analysis(
    image: Image.Image,
    age: int,
    sex: str,
    body_site: str,
    duration_months: int,
    symptoms: list,
    has_size_change: bool,
    has_itching_pain: bool,
    patient_profile: str,
    resource_constraint: str,
    progress=gr.Progress(),
):
    """Run the full analysis pipeline (with DullRazor preprocessing display)."""

    if image is None:
        empty_html = (
            f"<div style='color:#dc2626; padding:20px; "
            f"font-family:{FONT_FAMILY_SANS};'>"
            f"Please upload an image first.</div>"
        )
        return None, None, None, empty_html, "", "", "", "00:00"

    if pipeline_mgr.assistant is None:
        if pipeline_mgr.is_loading:
            wait_html = (
                f"<div style='color:#ea580c; padding:20px; "
                f"font-family:{FONT_FAMILY_SANS};'>"
                f"⏳ Pipeline still loading. Please wait and try again."
                f"</div>"
            )
        else:
            wait_html = (
                f"<div style='color:#dc2626; padding:20px; "
                f"font-family:{FONT_FAMILY_SANS};'>"
                f"Pipeline not loaded. Please restart the application."
                f"</div>"
            )
        return image, None, None, wait_html, "", "", "", "00:00"

    t_total = time.time()

    # DullRazor preprocessing (visual display)
    progress(0.05, desc="Preprocessing: hair removal (DullRazor)...")
    t_dullrazor = time.time()
    try:
        preprocessed_img = dullrazor_preprocess(image)
        elapsed_dullrazor = time.time() - t_dullrazor
        print(f"[DullRazor] Preprocessing complete: {elapsed_dullrazor:.2f}s")
    except Exception as e:
        print(f"[DullRazor] Preprocessing failed: {e}")
        preprocessed_img = image
        elapsed_dullrazor = 0.0

    # Save temp file (classification uses original image)
    progress(0.1, desc="Processing image...")
    temp_dir = Path("/tmp") if os.name != "nt" else Path(os.environ.get("TEMP", "."))
    temp_path = temp_dir / f"gradio_temp_{int(time.time())}.png"
    image.save(temp_path)

    # Build patient context
    progress(0.2, desc="Building patient context...")
    profile_data = PROFILE_MAPPING.get(
        patient_profile, PROFILE_MAPPING["General LMIC patient"]
    )

    symptoms_str = ", ".join(symptoms) if symptoms else "no notable symptoms"
    if has_size_change:
        symptoms_str += "; recent size increase reported"
    if has_itching_pain:
        symptoms_str += "; itching or pain reported"

    patient_meta = PatientMetadata(
        age=int(age),
        sex=sex,
        body_site=body_site,
        duration_months=int(duration_months),
        symptoms=symptoms_str,
        context=profile_data["context"],
        risk_factor=profile_data["risk_factor"],
        skin_type=profile_data["skin_type"],
        resource_constraint=RESOURCE_CONSTRAINT_OPTIONS.get(
            resource_constraint,
            "Standard primary care setting with limited specialist access",
        ),
    )

    # Vision + Grad-CAM
    progress(0.3, desc="Running Vision Classifier + Grad-CAM...")
    t_pipeline = time.time()

    image_pil = Image.open(temp_path).convert("RGB")
    clf_output = pipeline_mgr.assistant.vision.predict(image_pil)
    cam_map = clf_output.get("grad_cam_map")

    progress(0.5, desc="RAG retrieval...")
    rag_context = pipeline_mgr.assistant.rag.build_context(clf_output, top_k=5)

    progress(0.6, desc="Gemma 4 inference...")
    response = pipeline_mgr.assistant.gemma.generate(
        clf_output, patient_meta, rag_context,
    )

    # validate_and_correct_response_en already imported above
    response = validate_and_correct_response_en(response, clf_output)

    elapsed_pipeline = time.time() - t_pipeline

    progress(0.95, desc="Generating visualizations...")

    # Grad-CAM Overlay
    if cam_map is not None and isinstance(cam_map, np.ndarray):
        try:
            overlay_img = create_gradcam_overlay(image, cam_map)
            print(f"[Gradio] Grad-CAM overlay generated: {overlay_img.size}")
        except Exception as e:
            print(f"[Gradio] Overlay generation error: {e}")
            overlay_img = None
    else:
        print(f"[Gradio] No cam_map available")
        overlay_img = None

    # Format outputs
    rec_html = format_recommendation_html(response, clf_output)
    visual_html = format_visual_analysis_html(clf_output, response)
    patient_html = format_patient_summary_html(response)
    limit_html = format_limitations_html(response)

    elapsed_total = time.time() - t_total
    timing_str = (
        f"<div style='font-family:{FONT_FAMILY_MONO}; font-size:13px; color:#475569; "
        f"padding:8px 16px; background:#f1f5f9; border-radius:6px; "
        f"text-align:center;'>"
        f"Total: {elapsed_total:.1f}s | Preprocess: {elapsed_dullrazor:.2f}s | "
        f"Pipeline: {elapsed_pipeline:.1f}s (Vision + RAG + Gemma 4)"
        f"</div>"
    )

    try:
        os.remove(temp_path)
    except OSError:
        pass

    progress(1.0, desc="Complete")

    return (
        image,              # Original
        preprocessed_img,   # DullRazor preprocessed
        overlay_img,        # Grad-CAM
        rec_html,
        visual_html,
        patient_html,
        limit_html,
        timing_str,
    )


# ============================================================
# 8. Gradio UI Layout
# ============================================================
def build_ui() -> tuple[gr.Blocks, str]:
    """Build the Gradio Blocks UI and return (demo, custom_css)."""
    font_css = build_font_css()

    custom_css = font_css + """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }

    .gradio-container,
    .gradio-container button,
    .gradio-container input,
    .gradio-container select,
    .gradio-container textarea,
    .gradio-container label,
    .gradio-container .label-wrap,
    .gradio-container [data-testid="block-label"] {
        font-family: 'Noto Sans', system-ui, -apple-system, sans-serif !important;
    }

    .header-section {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        font-family: 'Noto Sans', system-ui, sans-serif;
    }
    """

    with gr.Blocks(
        title="Skin Lesion Assistant — LMIC",
    ) as demo:

        # Header
        gr.HTML(
            f"""
            <div class='header-section'>
              <div style='font-size: 28px; font-weight: 700; margin-bottom: 8px;
                          font-family: {FONT_FAMILY_SANS};'>
                Skin Lesion Screening Assistant
              </div>
              <div style='font-size: 14px; opacity: 0.9; margin-bottom: 12px;
                          font-weight: 400;
                          font-family: {FONT_FAMILY_SANS};'>
                Offline-first AI for Sub-Saharan Africa & Other LMICs
                &nbsp;•&nbsp; Vision Classifier + RAG + Gemma 4 E4B
              </div>
            </div>
            """
        )

        # Status row
        with gr.Row():
            with gr.Column(scale=2):
                network_status = gr.HTML(get_network_status_html())
            with gr.Column(scale=2):
                pipeline_status = gr.HTML(get_pipeline_status_html())
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Status", size="sm")

        refresh_btn.click(
            fn=lambda: (get_network_status_html(), get_pipeline_status_html()),
            outputs=[network_status, pipeline_status],
        )

        with gr.Row():
            # LEFT: INPUT
            with gr.Column(scale=1):
                gr.Markdown("### 1. Image Upload")
                image_input = gr.Image(type="pil", label="Skin Lesion Image", height=280)

                gr.Markdown("### 2. Patient Information")

                with gr.Row():
                    age_input = gr.Slider(5, 90, value=50, step=1, label="Age")
                    sex_input = gr.Radio(["male", "female"], value="male", label="Sex")

                body_site_input = gr.Dropdown(
                    choices=BODY_SITES, value="back", label="Body Site",
                )

                duration_input = gr.Slider(
                    1, 60, value=6, step=1, label="Lesion Duration (months)",
                )

                gr.Markdown("**Symptoms (multi-select):**")
                symptoms_input = gr.CheckboxGroup(
                    choices=SYMPTOMS_OPTIONS,
                    value=["asymptomatic (no notable symptoms)"],
                    label="Symptom Selection",
                    show_label=False,
                )

                gr.Markdown("**Key Indicators (highlighted):**")
                with gr.Row():
                    size_change_input = gr.Checkbox(
                        label="Recent size increase", value=False,
                    )
                    itching_pain_input = gr.Checkbox(
                        label="Itching / pain", value=False,
                    )

                gr.Markdown("### 3. Patient Risk Profile")
                profile_input = gr.Dropdown(
                    choices=list(PROFILE_MAPPING.keys()),
                    value="General LMIC patient",
                    label="Patient Context",
                )

                resource_input = gr.Dropdown(
                    choices=list(RESOURCE_CONSTRAINT_OPTIONS.keys()),
                    value="Standard primary care (specialist available)",
                    label="Healthcare Resource Setting",
                )

                analyze_btn = gr.Button(
                    "⚡ Analyze Offline", variant="primary", size="lg",
                )

                timing_display = gr.HTML(
                    f"<div style='font-family:{FONT_FAMILY_MONO}; font-size:13px; "
                    f"color:#475569; padding:8px 16px; background:#f1f5f9; "
                    f"border-radius:6px; text-align:center;'>"
                    f"Awaiting analysis..."
                    f"</div>"
                )

            # RIGHT: OUTPUT
            with gr.Column(scale=2):
                gr.Markdown("### Analysis Results")

                # 3-column image display
                with gr.Row():
                    original_display = gr.Image(
                        label="Original",
                        interactive=False,
                        height=240,
                    )
                    preprocessed_display = gr.Image(
                        label="Preprocessed (Hair Removed)",
                        interactive=False,
                        height=240,
                    )
                    overlay_display = gr.Image(
                        label="Grad-CAM Overlay (Model Attention)",
                        interactive=False,
                        height=240,
                    )

                with gr.Tabs():
                    with gr.Tab("📋 Recommendation"):
                        rec_output = gr.HTML(
                            f"<div style='padding:40px; text-align:center; "
                            f"color:#94a3b8; font-family:{FONT_FAMILY_SANS};'>"
                            f"Upload an image and click 'Analyze Offline' to begin."
                            f"</div>"
                        )

                    with gr.Tab("🔬 Visual Analysis"):
                        visual_output = gr.HTML("")

                    with gr.Tab("🧑 For Patient"):
                        patient_output = gr.HTML("")

                    with gr.Tab("⚠ Limitations"):
                        limit_output = gr.HTML("")

        gr.Markdown(
            """
            ---

            **System Information**
            - **Image Preprocessing:** DullRazor hair removal algorithm (Lee et al., 1997)
            - **Vision Classifier:** EfficientNet-B4 fine-tuned on HAM10000
            - **RAG Knowledge Base:** DermNet + BAD guidelines (BAAI/bge-m3 embeddings)
            - **Gemma 4 E4B:** Fine-tuned with 5,000 LMIC-specialized English samples (LoRA, 9.1M params)
            - **All inference runs locally; no patient data is transmitted externally**

            *This tool is for clinical decision support only and does not replace specialist diagnosis.*
            """
        )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                image_input, age_input, sex_input, body_site_input,
                duration_input, symptoms_input,
                size_change_input, itching_pain_input,
                profile_input, resource_input,
            ],
            outputs=[
                original_display,
                preprocessed_display,
                overlay_display,
                rec_output,
                visual_output,
                patient_output,
                limit_output,
                timing_display,
            ],
        )

        timer = gr.Timer(3.0)
        timer.tick(
            fn=lambda: (get_network_status_html(), get_pipeline_status_html()),
            outputs=[network_status, pipeline_status],
        )

    return demo, custom_css


# ============================================================
# 9. Main
# ============================================================
def main():
    print("=" * 60)
    print(" Skin Lesion Assistant - Gradio UI")
    print("=" * 60)
    print(f" Network status: {'OFFLINE' if is_truly_offline() else 'ONLINE'}")
    print(" Starting pre-warm in background thread...")
    print("=" * 60)

    prewarm_thread = threading.Thread(target=pipeline_mgr.prewarm, daemon=True)
    prewarm_thread.start()

    demo, custom_css = build_ui()
    demo.queue(max_size=5)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )


if __name__ == "__main__":
    main()
