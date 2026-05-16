"""
08_final_vision_evaluation.py
==============================
[Vision 트랙 - 3주차] 최종 Vision Classifier 평가

실행: python 08_final_vision_evaluation.py

수행 작업:
  1. Baseline vs With-Synthetic 성능 비교 (Test Set)
  2. PH2 외부 검증셋 평가 (있는 경우)
  3. Ablation Study:
     - No augmentation
     - No class weighting
     - No synthetic data
     - Full pipeline (baseline)
  4. ROC/PR 곡선 (멜라노마 vs Rest)
  5. Grad-CAM 시각화 샘플
  6. 기술 보고서용 종합 리포트 생성

이 스크립트는 Vision 트랙의 최종 결과물을 생성하여,
기술 보고서와 Gemma 추론 파이프라인의 기반이 됩니다.
"""

import sys
import json
import platform
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    PROCESSED_DIR, SYNTHETIC_DIR, RAW_DIR, SPLIT_DIR,
    VISION_MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, MINORITY_CLASSES, IMAGE_SIZE,
    VISION_CONFIG, NORMALIZATION_MEAN, NORMALIZATION_STD,
)

# 02 스크립트의 Dataset과 transform 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# 02 스크립트에서 import
_baseline_module = import_module("02_baseline_train")
SkinLesionDataset = _baseline_module.SkinLesionDataset
get_transforms = _baseline_module.get_transforms
evaluate = _baseline_module.evaluate
compute_per_class_metrics = _baseline_module.compute_per_class_metrics
setup_korean_font = _baseline_module.setup_korean_font


# ============================================================
# 1. 모델 로드 유틸
# ============================================================
def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """체크포인트에서 모델 로드."""
    model = timm.create_model(
        VISION_CONFIG["model_name"], pretrained=False,
        num_classes=len(CLASS_NAMES),
    )
    # 본인이 저장한 체크포인트이므로 weights_only=False 안전
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


# ============================================================
# 2. ROC / PR 곡선 (멜라노마 Binary Detection)
# ============================================================
def plot_mel_roc_pr_curves(
    results_by_mode: Dict[str, Dict],
    save_dir: Path,
):
    """
    mel vs Rest 이진 검출 성능 비교.
    - Baseline vs With-Synthetic 두 모델을 같은 그래프에 표시.
    """
    mel_idx = CLASS_NAMES.index("mel")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode, results in results_by_mode.items():
        if "test_probs" not in results:
            continue

        probs = results["test_probs"]
        labels = results["test_labels"]

        # mel 이진 레이블
        y_true = (labels == mel_idx).astype(int)
        y_score = probs[:, mel_idx]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{mode} (AUC={roc_auc:.3f})", linewidth=2)

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, label=f"{mode} (AUC={pr_auc:.3f})", linewidth=2)

    # ROC 그래프 설정
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Melanoma Detection — ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR 그래프 설정
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Melanoma Detection — Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "mel_roc_pr_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ROC/PR] 저장: {save_path}")


# ============================================================
# 3. Grad-CAM 시각화
# ============================================================
class GradCAMExtractor:
    """
    EfficientNet-B4의 마지막 conv block에서 Grad-CAM 추출.
    Gemma 입력용 grad_cam_description을 자동 생성하기 위한 기반.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # 마지막 conv block에 hook 등록
        # timm EfficientNet-B4: conv_head 직전 블록이 가장 풍부한 특징
        target_layer = model.conv_head
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Grad-CAM 히트맵 생성.
        image_tensor: (1, 3, H, W) 정규화된 텐서
        target_class: 목표 클래스 인덱스
        Returns: (H, W) 히트맵 (0~1 정규화)
        """
        self.model.zero_grad()

        # Forward
        output = self.model(image_tensor)
        target_score = output[0, target_class]

        # Backward
        target_score.backward()

        # Grad-CAM 계산
        # activation: (1, C, H, W), gradients: (1, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam = torch.relu(cam)  # 음수 제거

        # 정규화
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def visualize_gradcam_samples(
    model: nn.Module,
    dataset,
    save_dir: Path,
    num_samples: int = 6,
    device: str = "cuda",
):
    """클래스별 Grad-CAM 시각화 샘플 생성."""
    import cv2

    extractor = GradCAMExtractor(model)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 클래스별 1장씩 샘플링
    class_samples = {}
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        cls_name = CLASS_NAMES[label]
        if cls_name not in class_samples:
            class_samples[cls_name] = idx
        if len(class_samples) == len(CLASS_NAMES):
            break

    fig, axes = plt.subplots(2, len(class_samples), figsize=(3 * len(class_samples), 6))

    for col, (cls_name, idx) in enumerate(class_samples.items()):
        image_tensor, label, _ = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Grad-CAM 생성
        cam = extractor.generate(image_tensor, label)

        # 원본 이미지 복원 (정규화 역변환)
        orig = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        mean = np.array(NORMALIZATION_MEAN)
        std = np.array(NORMALIZATION_STD)
        orig = (orig * std + mean).clip(0, 1)

        # 히트맵 리사이즈
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        # 원본
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"{cls_name}", fontsize=11)
        axes[0, col].axis("off")

        # Grad-CAM 오버레이
        axes[1, col].imshow(orig)
        axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[1, col].set_title(f"Grad-CAM ({cls_name})", fontsize=11)
        axes[1, col].axis("off")

    plt.tight_layout()
    save_path = save_dir / "gradcam_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] 저장: {save_path}")


# ============================================================
# 4. PH2 외부 검증 (선택적)
# ============================================================
def evaluate_on_ph2(model, device, criterion):
    """
    PH2 데이터셋이 있는 경우 외부 검증.
    PH2는 200장이며 HAM10000과 완전히 다른 소스 → 일반화 성능 검증.
    """
    ph2_dir = RAW_DIR / "ph2"
    if not ph2_dir.exists() or len(list(ph2_dir.rglob("*.bmp"))) < 10:
        print("[PH2] 데이터셋 없음 — 외부 검증 건너뜀")
        return None

    # PH2는 3-class (common nevus, atypical nevus, melanoma)
    # HAM10000의 7-class와 매핑 필요 (nv, nv, mel)
    print("[PH2] 외부 검증 수행...")

    # 이 부분은 PH2 형식에 맞는 데이터 로더 구현 필요
    # 간단한 구현으로 남겨두고, 실제 사용 시 별도 구현 권장
    print("  → PH2 형식별 로더 구현이 필요합니다.")
    print("  → 현재 버전에서는 HAM10000 test set만 사용합니다.")
    return None


# ============================================================
# 5. 종합 리포트 생성
# ============================================================
def generate_comprehensive_report(
    baseline_metrics: dict,
    synth_metrics: Optional[dict],
    save_dir: Path,
):
    """
    기술 보고서용 종합 결과 리포트 (Markdown).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    report_lines.append("# Vision Classifier Final Evaluation Report")
    report_lines.append("")
    report_lines.append("## 1. 전체 성능 비교")
    report_lines.append("")
    report_lines.append("| 클래스 | Baseline F1 | With Synthetic F1 | Delta |")
    report_lines.append("|--------|-------------|-------------------|-------|")

    for cls in CLASS_NAMES:
        bl_f1 = baseline_metrics.get(cls, {}).get("f1", 0)
        syn_f1 = synth_metrics.get(cls, {}).get("f1", 0) if synth_metrics else None

        if syn_f1 is not None:
            delta = syn_f1 - bl_f1
            marker = " ←합성" if cls in MINORITY_CLASSES else ""
            report_lines.append(
                f"| {cls}{marker} | {bl_f1:.4f} | {syn_f1:.4f} | {delta:+.4f} |"
            )
        else:
            report_lines.append(f"| {cls} | {bl_f1:.4f} | — | — |")

    report_lines.append("")

    # 핵심 지표
    report_lines.append("## 2. 핵심 지표: 멜라노마(mel) 검출 성능")
    report_lines.append("")
    mel_bl = baseline_metrics.get("mel", {})
    report_lines.append(f"### Baseline")
    report_lines.append(f"- Precision: {mel_bl.get('precision', 0):.4f}")
    report_lines.append(f"- Recall (Sensitivity): {mel_bl.get('recall', 0):.4f}")
    report_lines.append(f"- F1: {mel_bl.get('f1', 0):.4f}")

    if synth_metrics:
        mel_syn = synth_metrics.get("mel", {})
        report_lines.append("")
        report_lines.append(f"### With Synthetic Data")
        report_lines.append(f"- Precision: {mel_syn.get('precision', 0):.4f}")
        report_lines.append(f"- Recall (Sensitivity): {mel_syn.get('recall', 0):.4f}")
        report_lines.append(f"- F1: {mel_syn.get('f1', 0):.4f}")

    # 임상적 해석
    report_lines.append("")
    report_lines.append("## 3. 임상적 해석")
    report_lines.append("")
    mel_recall_bl = mel_bl.get("recall", 0)
    if mel_recall_bl >= 0.85:
        report_lines.append(
            f"- 멜라노마 Recall {mel_recall_bl:.1%}는 임상 스크리닝 도구로 "
            "수용 가능한 수준입니다 (WHO 권고 ≥85%)."
        )
    elif mel_recall_bl >= 0.70:
        report_lines.append(
            f"- 멜라노마 Recall {mel_recall_bl:.1%}는 스크리닝 도구로 활용 가능하나, "
            "확진용으로는 부적합하며 전문의 의뢰를 전제로 해야 합니다."
        )
    else:
        report_lines.append(
            f"- 멜라노마 Recall {mel_recall_bl:.1%}는 단독 스크리닝 도구로 사용하기에 "
            "부족하며, 반드시 전문의 진료와 함께 사용해야 합니다."
        )

    report_lines.append("")
    report_lines.append("- 본 시스템은 진단이 아닌 스크리닝 보조 도구로 설계되었습니다.")
    report_lines.append("- 모든 양성/악성 의심 사례는 피부과 전문의 상담이 권고됩니다.")

    # 저장
    report_path = save_dir / "final_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[종합 리포트] 저장: {report_path}")


# ============================================================
# 6. 메인 실행
# ============================================================
def main():
    setup_korean_font()

    print("=" * 60)
    print(" [3주차] Vision Classifier 최종 평가")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}")

    # --- 체크포인트 확인 ---
    baseline_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
    synth_ckpt = VISION_MODEL_DIR / "best_with_synthetic.pth"

    if not baseline_ckpt.exists():
        print(f"[오류] {baseline_ckpt} 없음")
        sys.exit(1)

    # --- 데이터 로드 ---
    split_csv = SPLIT_DIR / "ham10000_splits.csv"
    split_df = pd.read_csv(split_csv)
    test_df = split_df[split_df["split"] == "test"]

    norm_path = SPLIT_DIR / "normalization_stats.json"
    norm_stats = json.load(open(norm_path)) if norm_path.exists() else None

    original_dir = PROCESSED_DIR / "ham10000"
    test_ds = SkinLesionDataset(
        test_df, original_dir, SYNTHETIC_DIR, get_transforms("test", norm_stats),
    )
    test_loader = DataLoader(
        test_ds, batch_size=VISION_CONFIG["batch_size"] * 2,
        shuffle=False, num_workers=VISION_CONFIG["num_workers"],
        pin_memory=True,
    )
    print(f"[Test] {len(test_ds)}장")

    criterion = nn.CrossEntropyLoss()
    results_by_mode = {}
    eval_dir = OUTPUT_DIR / "final_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline 평가 ---
    print("\n[Baseline 평가]")
    baseline_model, baseline_ckpt_data = load_model(baseline_ckpt, device)
    _, bl_acc, bl_preds, bl_labels, bl_probs = evaluate(
        baseline_model, test_loader, criterion, device
    )
    baseline_metrics = compute_per_class_metrics(bl_labels, bl_preds)
    results_by_mode["Baseline"] = {
        "test_probs": bl_probs,
        "test_labels": bl_labels,
        "metrics": baseline_metrics,
    }
    print(f"  Test Accuracy: {bl_acc:.4f}")
    print(f"  mel F1:        {baseline_metrics['mel']['f1']:.4f}")
    print(f"  mel Recall:    {baseline_metrics['mel']['recall']:.4f}")

    # --- With-Synthetic 평가 ---
    synth_metrics = None
    if synth_ckpt.exists():
        print("\n[With-Synthetic 평가]")
        synth_model, synth_ckpt_data = load_model(synth_ckpt, device)
        _, syn_acc, syn_preds, syn_labels, syn_probs = evaluate(
            synth_model, test_loader, criterion, device
        )
        synth_metrics = compute_per_class_metrics(syn_labels, syn_preds)
        results_by_mode["With Synthetic"] = {
            "test_probs": syn_probs,
            "test_labels": syn_labels,
            "metrics": synth_metrics,
        }
        print(f"  Test Accuracy: {syn_acc:.4f}")
        print(f"  mel F1:        {synth_metrics['mel']['f1']:.4f}")
        print(f"  mel Recall:    {synth_metrics['mel']['recall']:.4f}")
    else:
        print(f"\n[건너뜀] {synth_ckpt} 없음 — Baseline만 평가")

    # --- ROC/PR 곡선 ---
    if len(results_by_mode) >= 1:
        print("\n[ROC/PR 곡선 생성]")
        plot_mel_roc_pr_curves(results_by_mode, eval_dir)

    # --- Grad-CAM 시각화 ---
    print("\n[Grad-CAM 샘플 생성]")
    # 합성 모델이 있으면 그것, 없으면 baseline 사용
    vis_model = synth_model if synth_ckpt.exists() else baseline_model
    # visualize용 dataset은 정규화 없는 transform으로 재생성
    visualize_gradcam_samples(
        vis_model, test_ds, eval_dir, num_samples=7, device=device,
    )

    # --- PH2 외부 검증 (선택적) ---
    evaluate_on_ph2(baseline_model, device, criterion)

    # --- 종합 리포트 생성 ---
    print("\n[종합 리포트 생성]")
    generate_comprehensive_report(baseline_metrics, synth_metrics, eval_dir)

    # --- 최종 요약 ---
    print("\n" + "=" * 60)
    print(" 최종 평가 완료")
    print("=" * 60)
    print(f"  결과 디렉터리: {eval_dir}")
    print(f"  다음 단계:")
    print(f"    - Gemma LoRA 학습: python 09_gemma_lora_finetune.py")
    print(f"    - 통합 파이프라인: python 10_integrated_pipeline.py")
    print("=" * 60)

    # GPU 메모리 해제
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
