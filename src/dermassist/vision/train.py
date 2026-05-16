"""
02_baseline_train.py (v2 — 합성 데이터 지원)
==============================================
[Vision 트랙] EfficientNet-B4 학습

실행 모드:
  # Mode 1: 원본만 (1주차 baseline)
  python 02_baseline_train.py --mode baseline

  # Mode 2: 원본 + 합성 데이터 (2주차 재학습)
  python 02_baseline_train.py --mode with_synthetic

주요 변경사항 (v1 → v2):
  - CLI 모드 인자 추가 (--mode baseline / with_synthetic)
  - 합성 이미지 경로 자동 분기 (syn_* 접두사로 구분)
  - 매 에폭 mel F1 별도 추적
  - 체크포인트 파일명 모드별 분리 (덮어쓰기 방지)
  - 학습 종료 후 이전 모드 결과와 자동 비교
  - 한글 폰트 경고 해결
"""

import sys
import os
import json
import time
import random
import argparse
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    PROCESSED_DIR, SYNTHETIC_DIR, SPLIT_DIR, VISION_MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, MINORITY_CLASSES, IMAGE_SIZE,
    VISION_CONFIG, NORMALIZATION_MEAN, NORMALIZATION_STD,
)


# ============================================================
# 0. 한글 폰트 설정 (그래프용)
# ============================================================
def setup_korean_font():
    """matplotlib 한글 폰트 설정. 시스템별 기본 폰트 자동 선택."""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        try:
            import matplotlib.font_manager as fm
            fonts = [f.name for f in fm.fontManager.ttflist]
            if "NanumGothic" in fonts:
                plt.rcParams["font.family"] = "NanumGothic"
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 재현성(Seed) 설정
# ============================================================
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ============================================================
# 2. Dataset 클래스 — 원본/합성 이미지 경로 자동 분기
# ============================================================
class SkinLesionDataset(Dataset):
    """
    피부 병변 이미지 Dataset.
    image_id가 'syn_'으로 시작하면 합성 이미지 디렉터리에서 로드.
    그 외에는 원본 전처리 디렉터리에서 로드.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        original_image_dir: Path,
        synthetic_root: Path,
        transform=None,
    ):
        """
        Args:
            df: image_id, dx 컬럼이 포함된 DataFrame
            original_image_dir: 원본 전처리 이미지 디렉터리
                (예: data/processed/ham10000)
            synthetic_root: 합성 이미지 루트 디렉터리
                (예: data/synthetic — 하위에 {class}/filtered/)
            transform: torchvision transform 파이프라인
        """
        self.df = df.reset_index(drop=True)
        self.original_dir = original_image_dir
        self.synthetic_root = synthetic_root
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
        self._validate_paths()

    def _resolve_path(self, image_id: str, dx: str) -> Path:
        """image_id에 따라 원본/합성 경로 결정."""
        if image_id.startswith("syn_"):
            # 합성 이미지: data/synthetic/{class}/filtered/{image_id}.png
            return self.synthetic_root / dx / "filtered" / f"{image_id}.png"
        else:
            # 원본 이미지
            return self.original_dir / f"{image_id}.png"

    def _validate_paths(self):
        """파일이 실제 존재하는지 검사하고, 없는 행은 제거."""
        valid_indices = []
        missing_count = 0
        for i, row in self.df.iterrows():
            path = self._resolve_path(row["image_id"], row["dx"])
            if path.exists():
                valid_indices.append(i)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"  [경고] 누락된 이미지 {missing_count}장 제거")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["image_id"], row["dx"])

        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["dx"]]
        binary_label = 1 if row["dx"] in MALIGNANT_CLASSES else 0

        return image, label, binary_label


# ============================================================
# 3. Augmentation
# ============================================================
def get_transforms(split: str, norm_stats: Optional[dict] = None):
    mean = norm_stats["mean"] if norm_stats else NORMALIZATION_MEAN
    std = norm_stats["std"] if norm_stats else NORMALIZATION_STD

    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(30),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ============================================================
# 4. 모델 & 학습 유틸
# ============================================================
def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        VISION_CONFIG["model_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )
    print(f"[모델] {VISION_CONFIG['model_name']}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    return model


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["dx"].value_counts().reindex(CLASS_NAMES).values.astype(float)
    weights = len(df) / (len(CLASS_NAMES) * counts)
    weights = weights / weights.min()
    print(f"[클래스 가중치] {dict(zip(CLASS_NAMES, weights.round(2)))}")
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels, _ in tqdm(loader, desc="  Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = probs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (
        running_loss / total, correct / total,
        np.array(all_preds), np.array(all_labels), np.array(all_probs),
    )


# ============================================================
# 5. 클래스별 메트릭 계산 (mel F1 별도 추적용)
# ============================================================
def compute_per_class_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """클래스별 precision/recall/F1 반환. mel은 의료 안전성상 별도 추적 필수."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    return {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i, cls in enumerate(CLASS_NAMES)
    }


# ============================================================
# 6. 평가 보고서 생성
# ============================================================
def save_evaluation_report(
    labels, preds, probs, split_name: str, save_dir: Path, mode: str
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Classification Report
    report_text = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    )
    report_path = save_dir / f"{split_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {mode.upper()} MODE ===\n\n")
        f.write(report_text)
    print(f"\n[{split_name} 분류 보고서]\n{report_text}")

    # 이진 분류 (악성/양성)
    binary_labels = np.array([1 if CLASS_NAMES[l] in MALIGNANT_CLASSES else 0 for l in labels])
    binary_preds = np.array([1 if CLASS_NAMES[p] in MALIGNANT_CLASSES else 0 for p in preds])
    binary_report = classification_report(
        binary_labels, binary_preds,
        target_names=["Benign", "Malignant"], digits=4
    )
    print(f"[{split_name} 이진 분류 (악성/양성)]\n{binary_report}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_name} ({mode})")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = save_dir / f"{split_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Confusion Matrix] {cm_path}")

    # 클래스별 메트릭 JSON 저장 (비교용)
    per_class = compute_per_class_metrics(labels, preds)
    metrics_path = save_dir / f"{split_name}_per_class_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(per_class, f, indent=2)

    return per_class


# ============================================================
# 7. Baseline vs With-Synthetic 자동 비교
# ============================================================
def compare_with_baseline(current_metrics: dict, current_mode: str):
    """이전 baseline 결과와 현재 결과를 자동 비교."""
    if current_mode != "with_synthetic":
        return

    baseline_metrics_path = OUTPUT_DIR / "baseline_evaluation" / "test_baseline_per_class_metrics.json"
    if not baseline_metrics_path.exists():
        print(f"\n[비교 불가] {baseline_metrics_path} 없음")
        print("  → 먼저 python 02_baseline_train.py --mode baseline 를 실행하세요.")
        return

    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)

    print("\n" + "=" * 70)
    print(" Baseline vs With-Synthetic 비교 (Test Set)")
    print("=" * 70)
    print(f"  {'클래스':<8} {'Baseline F1':>12} {'New F1':>12} {'Delta':>10} {'판정':<14}")
    print("  " + "-" * 60)

    for cls in CLASS_NAMES:
        bl_f1 = baseline_metrics.get(cls, {}).get("f1", 0)
        new_f1 = current_metrics.get(cls, {}).get("f1", 0)
        delta = new_f1 - bl_f1

        if delta > 0.02:
            judgment = "개선"
        elif delta > 0:
            judgment = "소폭 상승"
        elif delta > -0.02:
            judgment = "변동 없음"
        else:
            judgment = "악화"

        marker = " ←합성대상" if cls in MINORITY_CLASSES else ""
        print(f"  {cls:<8} {bl_f1:>12.4f} {new_f1:>12.4f} {delta:>+10.4f} {judgment:<14}{marker}")

    # 핵심: mel F1 강조
    mel_bl = baseline_metrics.get("mel", {}).get("f1", 0)
    mel_new = current_metrics.get("mel", {}).get("f1", 0)
    mel_delta = mel_new - mel_bl

    print("\n" + "=" * 70)
    print(" 핵심 지표: 멜라노마(mel) F1 변화")
    print("=" * 70)
    print(f"  Baseline:        {mel_bl:.4f}")
    print(f"  With Synthetic:  {mel_new:.4f}")
    rel_pct = mel_delta / max(mel_bl, 0.001) * 100
    print(f"  변화폭:          {mel_delta:+.4f} ({rel_pct:+.1f}%)")

    # 판정 가이드
    print()
    if mel_delta > 0.03:
        print("  [판정] 유의미한 개선. 합성 데이터가 멜라노마 검출에 효과적.")
        print("         → Gemma 트랙으로 진행 권장.")
    elif mel_delta > 0.01:
        print("  [판정] 소폭 개선. 기술 보고서에 수치 명시 가능.")
        print("         → Gemma 트랙 진행 가능, 추가 개선 여지 있음.")
    elif mel_delta > -0.01:
        print("  [판정] 변동 없음. 합성 데이터의 효과가 미미함.")
        print("         → mel만 strength=0.40으로 재생성 권장.")
    else:
        print("  [판정] 악화. 합성 데이터가 노이즈로 작용.")
        print("         → mel 재생성 또는 합성 미포함 Baseline이 더 나을 수 있음.")

    # 악성 클래스 평균
    print("\n" + "=" * 70)
    print(" 악성 클래스(mel/bcc/akiec) 평균 F1 변화")
    print("=" * 70)
    mal_bl_avg = np.mean([baseline_metrics.get(c, {}).get("f1", 0) for c in MALIGNANT_CLASSES])
    mal_new_avg = np.mean([current_metrics.get(c, {}).get("f1", 0) for c in MALIGNANT_CLASSES])
    print(f"  Baseline:        {mal_bl_avg:.4f}")
    print(f"  With Synthetic:  {mal_new_avg:.4f}")
    print(f"  변화폭:          {mal_new_avg - mal_bl_avg:+.4f}")

    # 비교 결과 저장
    comparison = {
        "baseline": baseline_metrics,
        "with_synthetic": current_metrics,
        "mel_f1_delta": mel_delta,
        "malignant_f1_delta": float(mal_new_avg - mal_bl_avg),
    }
    comparison_path = OUTPUT_DIR / "synthetic_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[비교 결과 저장] {comparison_path}")


# ============================================================
# 8. 메인 실행
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "with_synthetic"],
        help="baseline: 원본만 / with_synthetic: 원본+합성 데이터",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    setup_korean_font()

    print("=" * 60)
    print(f" EfficientNet-B4 학습 — Mode: {mode.upper()}")
    print("=" * 60)

    seed = VISION_CONFIG.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device} | [Seed] {seed}")

    # --- 모드별 split CSV 선택 ---
    if mode == "baseline":
        split_csv = SPLIT_DIR / "ham10000_splits.csv"
        ckpt_name = "best_baseline.pth"
        eval_prefix = "test_baseline"
    else:  # with_synthetic
        split_csv = SPLIT_DIR / "ham10000_splits_with_synthetic.csv"
        ckpt_name = "best_with_synthetic.pth"
        eval_prefix = "test_with_synthetic"

    if not split_csv.exists():
        print(f"[오류] {split_csv} 없음")
        if mode == "with_synthetic":
            print("  → 먼저 06_synthetic_data_filtering.py를 실행하세요.")
        else:
            print("  → 먼저 01_download_and_preprocess.py를 실행하세요.")
        sys.exit(1)

    print(f"[데이터] {split_csv.name}")

    split_df = pd.read_csv(split_csv)
    norm_path = SPLIT_DIR / "normalization_stats.json"
    norm_stats = json.load(open(norm_path)) if norm_path.exists() else None

    original_dir = PROCESSED_DIR / "ham10000"

    # --- 데이터 분할 ---
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    # 학습 세트 클래스 분포 출력 (원본/합성 구분)
    print("\n[학습 세트 클래스 분포]")
    for cls in CLASS_NAMES:
        cls_df = train_df[train_df["dx"] == cls]
        orig = (~cls_df["image_id"].str.startswith("syn_")).sum()
        syn = cls_df["image_id"].str.startswith("syn_").sum()
        total = orig + syn
        if syn > 0:
            print(f"  {cls:<8} 원본: {orig:>5} + 합성: {syn:>4} = {total:>5}")
        else:
            print(f"  {cls:<8} 원본: {orig:>5}{'':>19} = {total:>5}")

    # --- Dataset & DataLoader ---
    train_ds = SkinLesionDataset(
        train_df, original_dir, SYNTHETIC_DIR,
        get_transforms("train", norm_stats),
    )
    val_ds = SkinLesionDataset(
        val_df, original_dir, SYNTHETIC_DIR,
        get_transforms("val", norm_stats),
    )
    test_ds = SkinLesionDataset(
        test_df, original_dir, SYNTHETIC_DIR,
        get_transforms("test", norm_stats),
    )

    train_loader = DataLoader(
        train_ds, batch_size=VISION_CONFIG["batch_size"],
        shuffle=True, num_workers=VISION_CONFIG["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VISION_CONFIG["batch_size"] * 2,
        shuffle=False, num_workers=VISION_CONFIG["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=VISION_CONFIG["batch_size"] * 2,
        shuffle=False, num_workers=VISION_CONFIG["num_workers"], pin_memory=True,
    )

    print(f"\n[데이터] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # --- 모델 & 옵티마이저 ---
    model = create_model(num_classes=len(CLASS_NAMES)).to(device)
    class_weights = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=VISION_CONFIG["lr"],
        weight_decay=VISION_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=VISION_CONFIG["epochs"]
    )

    # --- 학습 ---
    best_val_f1 = 0.0
    best_val_mel_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1_macro": [], "val_f1_mel": [],
    }

    print(f"\n[학습 시작] Epochs: {VISION_CONFIG['epochs']}, "
          f"Early stopping: {VISION_CONFIG['early_stopping_patience']}")

    mel_idx = CLASS_NAMES.index("mel")

    for epoch in range(1, VISION_CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels, _ = evaluate(
            model, val_loader, criterion, device
        )
        val_f1_macro = f1_score(val_labels, val_preds, average="macro")

        # mel F1 별도 계산
        val_f1_per_class = f1_score(
            val_labels, val_preds,
            labels=list(range(len(CLASS_NAMES))),
            average=None, zero_division=0,
        )
        val_f1_mel = val_f1_per_class[mel_idx]

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(float(val_f1_macro))
        history["val_f1_mel"].append(float(val_f1_mel))

        print(
            f"  Epoch {epoch:3d}/{VISION_CONFIG['epochs']} | "
            f"Train: L={train_loss:.4f} A={train_acc:.4f} | "
            f"Val: L={val_loss:.4f} A={val_acc:.4f} "
            f"F1(macro)={val_f1_macro:.4f} F1(mel)={val_f1_mel:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Best 모델 저장 (macro F1 기준)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_val_mel_f1 = val_f1_mel
            patience_counter = 0
            ckpt_path = VISION_MODEL_DIR / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1_macro": float(val_f1_macro),
                "val_f1_mel": float(val_f1_mel),
                "val_acc": val_acc,
                "mode": mode,
            }, ckpt_path)
            print(f"    ★ Best 저장 (F1 macro: {val_f1_macro:.4f}, F1 mel: {val_f1_mel:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= VISION_CONFIG["early_stopping_patience"]:
                print(f"  [Early Stopping] {VISION_CONFIG['early_stopping_patience']} epoch 개선 없음")
                break

    # --- 테스트 평가 ---
    print("\n[테스트 평가]")
    ckpt = torch.load(
        VISION_MODEL_DIR / ckpt_name,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    eval_dir = OUTPUT_DIR / "baseline_evaluation"
    per_class_metrics = save_evaluation_report(
        test_labels, test_preds, test_probs, eval_prefix, eval_dir, mode,
    )

    # --- 학습 곡선 (mel F1 포함) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].set_title("Accuracy")

    axes[2].plot(history["val_f1_macro"], label="Val F1 (macro)", linewidth=2)
    axes[2].plot(history["val_f1_mel"], label="Val F1 (mel)", linewidth=2, color="red")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].legend(); axes[2].set_title("F1 Curves")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(eval_dir / f"training_curves_{mode}.png", dpi=150)
    plt.close()

    with open(eval_dir / f"training_history_{mode}.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- 최종 요약 ---
    print("\n" + "=" * 60)
    print(f" [{mode.upper()}] 학습 완료")
    print("=" * 60)
    print(f"  Best Val F1 (macro): {best_val_f1:.4f}")
    print(f"  Best Val F1 (mel):   {best_val_mel_f1:.4f}")
    print(f"  Test mel F1:         {per_class_metrics['mel']['f1']:.4f}")
    print(f"  Test mel Recall:     {per_class_metrics['mel']['recall']:.4f}")

    # Baseline과 자동 비교
    compare_with_baseline(per_class_metrics, mode)


if __name__ == "__main__":
    main()
