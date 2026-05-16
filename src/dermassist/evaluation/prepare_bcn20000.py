"""
prepare_bcn20000.py
===================
Prepare BCN20000 evaluation samples by extracting class-balanced images
from the raw ISIC Archive metadata.

This script applies actual BCN20000 distribution analysis:
  - vasc class: not present in BCN20000 (excluded)
  - SCC: treated separately from akiec (excluded, conservative approach)
  - 6 evaluation classes: nv, mel, bcc, akiec, bkl, df

Outputs (relative to project root):
  data/external/bcn20000_eval/
  ├── metadata.csv                          # Evaluation metadata
  └── nv/, mel/, bcc/, akiec/, bkl/, df/    # Per-class image folders

Run via:
    python -m dermassist.evaluation.prepare_bcn20000
"""

import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


# ============================================================
# Path configuration (relative to project root)
# ============================================================
# This file is at: src/dermassist/evaluation/prepare_bcn20000.py
# Project root is 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SOURCE_METADATA_CSV = (
    PROJECT_ROOT / "data" / "external" / "bcn20000" /
    "bcn20000_metadata_2026-05-07.csv"
)

# Downloaded ISIC-images folder
SOURCE_IMAGES_DIR = (
    PROJECT_ROOT / "data" / "external" / "bcn20000" / "ISIC-images"
)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "bcn20000_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters
SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42


# ============================================================
# HAM10000 to BCN20000 diagnosis mapping
# ============================================================
# Notes on class exclusions:
# - vasc: BCN20000 has insufficient vascular lesion samples
# - SCC: clinically distinct from akiec, excluded for conservative evaluation
HAM_TO_BCN_DIAGNOSIS = {
    "nv": [
        "Nevus",
    ],
    "mel": [
        "Melanoma, NOS",
        "Melanoma metastasis",
    ],
    "bcc": [
        "Basal cell carcinoma",
    ],
    "akiec": [
        "Solar or actinic keratosis",
    ],
    "bkl": [
        "Seborrheic keratosis",
        "Solar lentigo",
    ],
    "df": [
        "Dermatofibroma",
    ],
    # vasc excluded: insufficient samples in BCN20000
}


def find_image_path(isic_id: str, source_dir: Path) -> Path:
    """Find the image file path for a given ISIC ID."""
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png"]:
        path = source_dir / f"{isic_id}{ext}"
        if path.exists():
            return path
    return None


def main():
    print("=" * 60)
    print(" BCN20000 External Validation Sample Preparation")
    print("=" * 60)
    print(f" Metadata: {SOURCE_METADATA_CSV}")
    print(f" Image folder: {SOURCE_IMAGES_DIR}")
    print(f" Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Path validation
    if not SOURCE_METADATA_CSV.exists():
        print(f"\n[Error] Metadata file not found: {SOURCE_METADATA_CSV}")
        print("        Please verify the file is at the expected location.")
        sys.exit(1)

    if not SOURCE_IMAGES_DIR.exists():
        print(f"\n[Error] Image folder not found: {SOURCE_IMAGES_DIR}")
        sys.exit(1)

    # Load metadata
    print("\n[1/4] Loading metadata...")
    df = pd.read_csv(SOURCE_METADATA_CSV)
    print(f"  Total rows: {len(df):,}")
    print(f"  Rows with diagnosis_3: {df['diagnosis_3'].notna().sum():,}")

    # Scan image folder
    print("\n[2/4] Scanning image folder...")
    sample_images = list(SOURCE_IMAGES_DIR.glob("*.jpg"))[:5]
    if not sample_images:
        sample_images = list(SOURCE_IMAGES_DIR.glob("*.JPG"))[:5]
    print(f"  Sample images (first 5):")
    for img in sample_images:
        print(f"    {img.name}")

    total_images = (
        len(list(SOURCE_IMAGES_DIR.glob("*.jpg"))) +
        len(list(SOURCE_IMAGES_DIR.glob("*.JPG")))
    )
    print(f"  Total images: {total_images:,}")

    # Per-class sampling
    print(f"\n[3/4] Per-class sampling (target: {SAMPLES_PER_CLASS} per class)")
    random.seed(RANDOM_SEED)

    selected_rows = []
    sampling_summary = {}

    for ham_class, bcn_diagnoses in HAM_TO_BCN_DIAGNOSIS.items():
        mask = df["diagnosis_3"].isin(bcn_diagnoses)
        candidates = df[mask].copy()

        candidate_count = len(candidates)
        print(f"\n  {ham_class} <- {', '.join(bcn_diagnoses)}")
        print(f"    Candidates after mapping: {candidate_count}")

        if candidate_count > 0:
            unique_diagnoses = candidates["diagnosis_3"].value_counts()
            for diag, cnt in unique_diagnoses.items():
                print(f"      {diag}: {cnt}")

        if candidate_count == 0:
            print(f"    [Warning] No samples available")
            sampling_summary[ham_class] = 0
            continue

        sample_n = min(SAMPLES_PER_CLASS, candidate_count)
        sampled = candidates.sample(n=sample_n, random_state=RANDOM_SEED)
        sampled["ham_class"] = ham_class
        selected_rows.append(sampled)
        sampling_summary[ham_class] = sample_n
        print(f"    Sampled: {sample_n}")

    if not selected_rows:
        print("\n[Error] No samples could be selected")
        sys.exit(1)

    selected_df = pd.concat(selected_rows, ignore_index=True)
    print(f"\n  Total samples: {len(selected_df)}")

    # Copy images
    print(f"\n[4/4] Copying images")

    eval_records = []
    missing_images = []

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Copying"):
        ham_class = row["ham_class"]
        isic_id = row["isic_id"]

        src_path = find_image_path(isic_id, SOURCE_IMAGES_DIR)
        if src_path is None:
            missing_images.append(isic_id)
            continue

        class_dir = OUTPUT_DIR / ham_class
        class_dir.mkdir(exist_ok=True)
        dst_path = class_dir / f"{isic_id}.jpg"

        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        eval_records.append({
            "isic_id": isic_id,
            "ham_class": ham_class,
            "bcn_diagnosis_3": row["diagnosis_3"],
            "bcn_diagnosis_1": row["diagnosis_1"],
            "age_approx": row["age_approx"] if pd.notna(row["age_approx"]) else "",
            "sex": row["sex"] if pd.notna(row["sex"]) else "",
            "anatom_site_general": (
                row["anatom_site_general"]
                if pd.notna(row["anatom_site_general"]) else ""
            ),
            "image_path": str(dst_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        })

    eval_df = pd.DataFrame(eval_records)
    metadata_path = OUTPUT_DIR / "metadata.csv"
    eval_df.to_csv(metadata_path, index=False, encoding="utf-8")

    # Summary
    print("\n" + "=" * 60)
    print(" Sample preparation complete")
    print("=" * 60)
    print(f"  Metadata: {metadata_path}")
    print(f"  Image folder: {OUTPUT_DIR}")
    if missing_images:
        print(f"  [Warning] {len(missing_images)} images not found:")
        for mid in missing_images[:5]:
            print(f"    - {mid}")
        if len(missing_images) > 5:
            print(f"    ... and {len(missing_images) - 5} more")
    print()

    print(f"  Per-class samples:")
    for ham_class in HAM_TO_BCN_DIAGNOSIS:
        actual_count = len(eval_df[eval_df["ham_class"] == ham_class])
        target = SAMPLES_PER_CLASS
        status = "OK" if actual_count == target else "SHORT"
        print(f"    [{status}] {ham_class}: {actual_count}/{target}")

    print(f"\n  Total evaluation samples: {len(eval_df)}")
    print(f"  Estimated evaluation time: ~{len(eval_df) * 65 / 60:.0f} minutes")
    print()
    print(f"  Next step: python scripts/08_evaluate_bcn20000.py")
    print(f"\n[Notes]")
    print(f"  vasc class excluded: insufficient samples in BCN20000.")
    print(f"  SCC excluded: clinically distinct from akiec (conservative evaluation).")


if __name__ == "__main__":
    main()
