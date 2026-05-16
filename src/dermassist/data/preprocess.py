"""
preprocess.py
=============
Data download → preprocessing → train/val/test split pipeline.

Run via:
    python scripts/01_prepare_data.py

Processing flow:
  1. Download HAM10000 + PH2 (Kaggle API or manual)
  2. DullRazor hair removal + CLAHE contrast enhancement
  3. Resize to 224x224 + compute normalization statistics
  4. Stratified 7:1:2 split (PH2 as external holdout)
  5. Save class distribution visualization
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Headless backend for environments without GUI

# Use DejaVu Sans (default matplotlib font, available on all platforms)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Add project root to sys.path for configs import
try:
    from configs.config import (
        RAW_DIR, PROCESSED_DIR, SPLIT_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
        IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    )
except ImportError:
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from configs.config import (
        RAW_DIR, PROCESSED_DIR, SPLIT_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
        IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    )


# ============================================================
# 1. Data download
# ============================================================
def download_ham10000():
    """
    Download the HAM10000 dataset.

    Requires Kaggle API credentials to be configured.
    Manual download:
        https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    """
    ham_dir = RAW_DIR / "ham10000"
    if ham_dir.exists() and len(list(ham_dir.glob("*.jpg"))) > 9000:
        print(f"[Skip] HAM10000 already exists: {ham_dir}")
        return ham_dir

    ham_dir.mkdir(parents=True, exist_ok=True)
    print("[Download] HAM10000...")

    try:
        os.system(
            f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 "
            f"-p {RAW_DIR} --unzip"
        )
        # Consolidate images into a single folder
        for part_dir in RAW_DIR.glob("HAM10000_images_part*"):
            for img in part_dir.glob("*.jpg"):
                shutil.move(str(img), str(ham_dir / img.name))
        print(f"[Done] HAM10000: {len(list(ham_dir.glob('*.jpg')))} images")
    except Exception as e:
        print(f"[Error] Kaggle download failed: {e}")
        print("  Please download manually and place images at: data/raw/ham10000/")
        print("  Place metadata CSV at: data/raw/HAM10000_metadata.csv")

    return ham_dir


def download_ph2():
    """
    Download the PH2 dataset (200 images, used as external holdout validation).

    Official source: https://www.fc.up.pt/addi/ph2%20database.html
    Note: Automated download is unreliable; manual setup is required.
    """
    ph2_dir = RAW_DIR / "ph2"
    if ph2_dir.exists() and len(list(ph2_dir.rglob("*.bmp"))) > 100:
        print(f"[Skip] PH2 already exists: {ph2_dir}")
        return ph2_dir

    ph2_dir.mkdir(parents=True, exist_ok=True)
    print("[Download] PH2...")
    print("  Note: PH2 automated download is unreliable.")
    print("  Please download manually and place at: data/raw/ph2/")
    print("  Alternative: ISIC Archive has equivalent images at")
    print("    https://www.isic-archive.com/")

    return ph2_dir


# ============================================================
# 2. Preprocessing pipeline
# ============================================================
def dullrazor_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    DullRazor algorithm for hair removal from skin lesion images.

    Principle: Detect hair via blackhat morphological operation,
    then inpaint to restore the underlying skin pixels.

    Reference: Lee et al. (1997) "DullRazor: A software approach
    to hair removal from images."

    Args:
        image: BGR-format OpenCV image (H, W, 3)
    Returns:
        Hair-removed image (BGR)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blackhat transform highlights dark linear structures (hair)
    # Kernel size 17 is well-suited for typical hair thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create binary hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint hair regions using surrounding skin pixels
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances subtle color and texture variations in skin lesions.
    Applied only to the L channel in LAB color space to preserve hue.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # clipLimit=2.0: prevent over-amplification of contrast
    # tileGridSize=(8,8): adaptive local region size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_single_image(
    image_path: Path,
    output_path: Path,
    target_size: int = IMAGE_SIZE,
) -> bool:
    """
    Single-image preprocessing pipeline:
      1. Load
      2. DullRazor hair removal
      3. CLAHE contrast enhancement
      4. Center-crop to square then resize
      5. Save as lossless PNG
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # Step 1: Hair removal
        img = dullrazor_hair_removal(img)

        # Step 2: CLAHE
        img = apply_clahe(img)

        # Step 3: Center-crop to square, then resize
        h, w = img.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img = img[top : top + min_dim, left : left + min_dim]
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Step 4: Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"  [Preprocess failed] {image_path.name}: {e}")
        return False


# ============================================================
# 3. Metadata loading and split
# ============================================================
def load_ham10000_metadata() -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV.

    Required columns: image_id, dx (diagnosis), dx_type, age, sex, localization.
    """
    # Candidate metadata paths
    candidates = [
        RAW_DIR / "HAM10000_metadata.csv",
        RAW_DIR / "HAM10000_metadata",
        RAW_DIR / "hmnist_28_28_RGB.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            print(f"[Metadata] Loaded: {path.name} ({len(df)} rows)")
            return df

    raise FileNotFoundError(
        f"HAM10000 metadata not found. "
        f"Please place it at one of: {[str(c) for c in candidates]}"
    )


def create_stratified_split(df: pd.DataFrame) -> dict:
    """
    Stratified 7:1:2 split (train:val:test).

    - Images with the same lesion_id are placed in the same split
      to prevent data leakage.
    - Class proportions are preserved across splits.

    Returns:
        {"train": [image_ids], "val": [...], "test": [...]}
    """
    # Group by lesion_id to prevent duplicate images across splits
    if "lesion_id" in df.columns:
        lesion_df = df.groupby("lesion_id").first().reset_index()
        unique_col = "lesion_id"
    else:
        lesion_df = df.copy()
        unique_col = "image_id"

    # First split: train (70%) + temp (30%)
    train_ids, temp_ids = train_test_split(
        lesion_df[unique_col],
        test_size=0.30,
        random_state=42,
        stratify=lesion_df["dx"],
    )

    # Second split: val (10%) + test (20%) - split temp at 1:2 ratio
    temp_df = lesion_df[lesion_df[unique_col].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df[unique_col],
        test_size=0.667,  # 2/3 of temp = ~20% of total
        random_state=42,
        stratify=temp_df["dx"],
    )

    # Map lesion_id back to image_id if using lesion-based split
    splits = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if unique_col == "lesion_id":
            image_ids = df[df["lesion_id"].isin(ids)]["image_id"].tolist()
        else:
            image_ids = ids.tolist()
        splits[name] = image_ids

    return splits


# ============================================================
# 4. Normalization statistics
# ============================================================
def compute_normalization_stats(image_dir: Path, image_ids: list) -> dict:
    """
    Compute per-channel mean and std from the training set.

    These statistics are used for both the Vision Classifier and
    Gemma vision encoder input normalization.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0

    for img_id in tqdm(image_ids, desc="Computing normalization stats"):
        img_path = image_dir / f"{img_id}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.0

        pixel_sum += img.mean(axis=(0, 1))
        pixel_sq_sum += (img ** 2).mean(axis=(0, 1))
        count += 1

    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)

    return {"mean": mean.tolist(), "std": std.tolist(), "num_images": count}


# ============================================================
# 5. Class distribution visualization
# ============================================================
def plot_class_distribution(df: pd.DataFrame, splits: dict, save_path: Path):
    """
    Visualize class distribution per split.

    Highlights minority class imbalance for the report (Figure 1).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (split_name, image_ids) in zip(axes, splits.items()):
        split_df = df[df["image_id"].isin(image_ids)]
        counts = split_df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0)

        # Red for malignant classes, blue for benign
        colors = [
            "#e74c3c" if cls in MALIGNANT_CLASSES else "#3498db"
            for cls in CLASS_NAMES
        ]
        bars = ax.bar(CLASS_NAMES, counts.values, color=colors)
        ax.set_title(f"{split_name.upper()} (n={len(image_ids)})", fontsize=14)
        ax.set_ylabel("Number of images")
        ax.tick_params(axis="x", rotation=45)

        # Annotate bar heights
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "HAM10000 Class Distribution (red: malignant/pre-malignant, blue: benign)",
        fontsize=15, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved to: {save_path}")


# ============================================================
# 6. Main entry point
# ============================================================
def main():
    print("=" * 60)
    print(" Data Download and Preprocessing Pipeline")
    print("=" * 60)

    # --- Step 1: Download ---
    ham_dir = download_ham10000()
    ph2_dir = download_ph2()

    # --- Step 2: Load metadata ---
    df = load_ham10000_metadata()
    print(f"\n[Original class distribution]")
    print(df["dx"].value_counts().to_string())

    # Add binary label
    df["binary_label"] = df["dx"].apply(
        lambda x: "malignant" if x in MALIGNANT_CLASSES else "benign"
    )

    # --- Step 3: Preprocessing ---
    print(f"\n[Preprocessing] Processing {len(df)} images...")
    processed_dir = PROCESSED_DIR / "ham10000"
    success, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_id = row["image_id"]
        # Search for source image
        src = ham_dir / f"{img_id}.jpg"
        if not src.exists():
            # Try alternate paths
            for alt in RAW_DIR.rglob(f"{img_id}.jpg"):
                src = alt
                break
        dst = processed_dir / f"{img_id}.png"

        if dst.exists():
            success += 1
            continue

        if preprocess_single_image(src, dst):
            success += 1
        else:
            fail += 1

    print(f"[Preprocessing done] Success: {success}, Failed: {fail}")

    # --- Step 4: Train/val/test split ---
    splits = create_stratified_split(df)
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} images")

    # Save split CSV
    split_records = []
    for name, ids in splits.items():
        for img_id in ids:
            row = df[df["image_id"] == img_id].iloc[0]
            split_records.append({
                "image_id": img_id,
                "split": name,
                "dx": row["dx"],
                "binary_label": row["binary_label"],
            })
    split_df = pd.DataFrame(split_records)
    split_csv_path = SPLIT_DIR / "ham10000_splits.csv"
    split_df.to_csv(split_csv_path, index=False)
    print(f"[Split saved] {split_csv_path}")

    # --- Step 5: Normalization statistics ---
    print("\n[Normalization] Computing training-set statistics...")
    norm_stats = compute_normalization_stats(processed_dir, splits["train"])
    norm_path = SPLIT_DIR / "normalization_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Mean: {norm_stats['mean']}")
    print(f"  Std:  {norm_stats['std']}")
    print(f"  Saved to: {norm_path}")

    # --- Step 6: Visualization ---
    plot_class_distribution(df, splits, OUTPUT_DIR / "class_distribution.png")

    print("\n" + "=" * 60)
    print(" Preprocessing complete. Next: python scripts/02_train_vision.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
