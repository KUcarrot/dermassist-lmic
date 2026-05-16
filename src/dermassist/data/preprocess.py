"""
01_download_and_preprocess.py
=============================
[Vision 트랙 - 1주차] 데이터 다운로드 → 전처리 → 학습/검증/테스트 분할

실행: python3 01_download_and_preprocess.py

처리 흐름:
  1. HAM10000 + PH2 다운로드 (Kaggle API / 수동)
  2. DullRazor 모발 제거 + CLAHE 대비 보정
  3. 224x224 리사이즈 + 정규화 통계 계산
  4. Stratified 7:1:2 분할 (PH2는 완전 외부 holdout)
  5. 클래스 분포 시각화 저장
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
matplotlib.use("Agg")  # GUI 없는 환경 대응 Anti-Grain Geometry
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

# --- 프로젝트 루트를 sys.path에 추가 ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    RAW_DIR, PROCESSED_DIR, SPLIT_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
    IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
)


# ============================================================
# 1. 데이터 다운로드
# ============================================================
def download_ham10000():
    """
    HAM10000 데이터셋 다운로드.
    Kaggle API 토큰이 설정되어 있어야 합니다.
    수동 다운로드: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    """
    ham_dir = RAW_DIR / "ham10000"
    if ham_dir.exists() and len(list(ham_dir.glob("*.jpg"))) > 9000:
        print(f"[건너뜀] HAM10000 이미 존재: {ham_dir}")
        return ham_dir

    ham_dir.mkdir(parents=True, exist_ok=True)
    print("[다운로드] HAM10000...")

    try:
        os.system(
            f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 "
            f"-p {RAW_DIR} --unzip"
        )
        # 다운로드 후 이미지를 한 폴더로 통합
        for part_dir in RAW_DIR.glob("HAM10000_images_part*"):
            for img in part_dir.glob("*.jpg"):
                shutil.move(str(img), str(ham_dir / img.name))
        print(f"[완료] HAM10000: {len(list(ham_dir.glob('*.jpg')))}장")
    except Exception as e:
        print(f"[오류] Kaggle 다운로드 실패: {e}")
        print("  → 수동으로 다운로드 후 data/raw/ham10000/ 에 이미지를 배치하세요.")
        print("  → 메타데이터 CSV는 data/raw/HAM10000_metadata.csv 에 배치하세요.")

    return ham_dir


def download_ph2(): # 자동 다운안됨. 내가 수동으로 넣었음
    """
    PH2 데이터셋 다운로드 (200장, 외부 holdout 검증용).
    공식: https://www.fc.up.pt/addi/ph2%20database.html
    """
    ph2_dir = RAW_DIR / "ph2"
    if ph2_dir.exists() and len(list(ph2_dir.rglob("*.bmp"))) > 100:
        print(f"[건너뜀] PH2 이미 존재: {ph2_dir}")
        return ph2_dir

    ph2_dir.mkdir(parents=True, exist_ok=True)
    print("[다운로드] PH2...")
    print("  → PH2는 자동 다운로드가 불안정합니다.")
    print("  → 수동 다운로드 후 data/raw/ph2/ 에 배치하세요.")
    print("  → 또는 ISIC Archive에서 동일 이미지 확보 가능:")
    print("    https://www.isic-archive.com/")

    return ph2_dir


# ============================================================
# 2. 전처리 파이프라인
# ============================================================
def dullrazor_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    DullRazor 알고리즘: 피부 영상에서 모발을 제거.
    원리: blackhat 형태학 변환으로 모발 검출 → inpainting으로 제거.

    Args:
        image: BGR 형식 OpenCV 이미지 (H, W, 3)
    Returns:
        모발 제거된 이미지 (BGR)
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blackhat 변환: 어두운 선형 구조(모발) 강조
    # 커널 크기 17은 일반적인 모발 두께에 적합
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 이진화: 모발 영역 마스크 생성
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpainting: 모발 영역을 주변 피부색으로 복원
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용.
    피부 병변의 미세한 색상·질감 차이를 강조.
    LAB 색공간의 L 채널에만 적용하여 색상 왜곡 방지.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # clipLimit=2.0: 과도한 대비 증가 방지
    # tileGridSize=(8,8): 적응적 영역 크기
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
    단일 이미지 전처리 파이프라인:
      1. 로드
      2. DullRazor 모발 제거
      3. CLAHE 대비 보정
      4. 중앙 크롭 → 리사이즈 (비율 유지)
      5. 저장 (PNG, 무손실)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # Step 1: 모발 제거
        img = dullrazor_hair_removal(img)

        # Step 2: CLAHE
        img = apply_clahe(img)

        # Step 3: 중앙 크롭 (정사각형) → 리사이즈
        h, w = img.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img = img[top : top + min_dim, left : left + min_dim]
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Step 4: 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"  [전처리 실패] {image_path.name}: {e}")
        return False


# ============================================================
# 3. 메타데이터 로드 및 분할
# ============================================================
def load_ham10000_metadata() -> pd.DataFrame:
    """
    HAM10000 메타데이터 CSV 로드.
    필수 컬럼: image_id, dx (진단명), dx_type, age, sex, localization
    """
    # 가능한 메타데이터 경로들
    candidates = [
        RAW_DIR / "HAM10000_metadata.csv",
        RAW_DIR / "HAM10000_metadata",
        RAW_DIR / "hmnist_28_28_RGB.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            print(f"[메타데이터] 로드 완료: {path.name} ({len(df)}행)")
            return df

    raise FileNotFoundError(
        f"HAM10000 메타데이터를 찾을 수 없습니다. "
        f"다음 경로 중 하나에 배치하세요: {[str(c) for c in candidates]}"
    )


def create_stratified_split(df: pd.DataFrame) -> dict:
    """
    Stratified 7:1:2 분할.
    - 동일 병변(lesion_id)은 같은 분할에 배치 (데이터 누수 방지).
    - 클래스 비율 유지.

    Returns:
        {"train": [...image_ids], "val": [...], "test": [...]}
    """
    # 동일 lesion_id 그룹화: 중복 이미지가 서로 다른 분할에 들어가는 것을 방지
    if "lesion_id" in df.columns:
        lesion_df = df.groupby("lesion_id").first().reset_index()
        unique_col = "lesion_id"
    else:
        lesion_df = df.copy()
        unique_col = "image_id"

    # 1차 분할: train(70%) + temp(30%)
    train_ids, temp_ids = train_test_split(
        lesion_df[unique_col],
        test_size=0.30,
        random_state=42,
        stratify=lesion_df["dx"],
    )

    # 2차 분할: val(10%) + test(20%) → temp를 1:2로 분할
    temp_df = lesion_df[lesion_df[unique_col].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df[unique_col],
        test_size=0.667,  # temp의 2/3 → 전체의 ~20%
        random_state=42,
        stratify=temp_df["dx"],
    )

    # lesion_id → image_id 매핑 (lesion_id 기반 분할이면)
    splits = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if unique_col == "lesion_id":
            image_ids = df[df["lesion_id"].isin(ids)]["image_id"].tolist()
        else:
            image_ids = ids.tolist()
        splits[name] = image_ids

    return splits


# ============================================================
# 4. 정규화 통계 계산
# ============================================================
def compute_normalization_stats(image_dir: Path, image_ids: list) -> dict:
    """
    학습 세트 이미지로부터 채널별 mean, std 계산.
    이 값은 Vision Classifier와 Gemma Vision 입력 모두에 사용.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0

    for img_id in tqdm(image_ids, desc="정규화 통계 계산"):
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
# 5. 클래스 분포 시각화
# ============================================================
def plot_class_distribution(df: pd.DataFrame, splits: dict, save_path: Path):
    """
    분할별 클래스 분포 시각화.
    소수 클래스 불균형을 한눈에 파악 → 보고서 Figure 1 용.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (split_name, image_ids) in zip(axes, splits.items()):
        split_df = df[df["image_id"].isin(image_ids)]
        counts = split_df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0)

        # 악성 클래스는 빨간색, 양성은 파란색
        colors = [
            "#e74c3c" if cls in MALIGNANT_CLASSES else "#3498db"
            for cls in CLASS_NAMES
        ]
        bars = ax.bar(CLASS_NAMES, counts.values, color=colors)
        ax.set_title(f"{split_name.upper()} (n={len(image_ids)})", fontsize=14)
        ax.set_ylabel("이미지 수")
        ax.tick_params(axis="x", rotation=45)

        # 막대 위에 수치 표시
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "HAM10000 클래스 분포 (빨강: 악성/전암, 파랑: 양성)", fontsize=15, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] 저장: {save_path}")


# ============================================================
# 6. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print(" [1주차] 데이터 다운로드 & 전처리 파이프라인")
    print("=" * 60)

    # --- Step 1: 다운로드 ---
    ham_dir = download_ham10000()
    ph2_dir = download_ph2()

    # --- Step 2: 메타데이터 로드 ---
    df = load_ham10000_metadata()
    print(f"\n[클래스 분포 (원본)]")
    print(df["dx"].value_counts().to_string())

    # 이진 레이블 추가
    df["binary_label"] = df["dx"].apply(
        lambda x: "malignant" if x in MALIGNANT_CLASSES else "benign"
    )

    # --- Step 3: 전처리 ---
    print(f"\n[전처리] {len(df)}장 처리 중...")
    processed_dir = PROCESSED_DIR / "ham10000"
    success, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="전처리"):
        img_id = row["image_id"]
        # 원본 이미지 탐색 (ham10000 폴더 내)
        src = ham_dir / f"{img_id}.jpg"
        if not src.exists():
            # 대체 경로 시도
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

    print(f"[전처리 완료] 성공: {success}, 실패: {fail}")

    # --- Step 4: 학습/검증/테스트 분할 ---
    splits = create_stratified_split(df)
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)}장")

    # 분할 결과 CSV 저장
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
    print(f"[분할 저장] {split_csv_path}")

    # --- Step 5: 정규화 통계 ---
    print("\n[정규화] 학습 세트 통계 계산...")
    norm_stats = compute_normalization_stats(processed_dir, splits["train"])
    norm_path = SPLIT_DIR / "normalization_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Mean: {norm_stats['mean']}")
    print(f"  Std:  {norm_stats['std']}")
    print(f"  저장: {norm_path}")

    # --- Step 6: 시각화 ---
    plot_class_distribution(df, splits, OUTPUT_DIR / "class_distribution.png")

    print("\n" + "=" * 60)
    print(" 전처리 완료. 다음 단계: python3 02_baseline_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
