"""
prepare_bcn20000_samples.py (v2 - 실제 분포 반영)
==================================================
BCN20000 메타데이터에서 클래스별 샘플 추출.

실제 분포 분석 결과 반영:
  - vasc 클래스: BCN20000에 없음 (제외)
  - SCC: akiec과 별개로 처리 (제외)
  - 6개 클래스 평가: nv, mel, bcc, akiec, bkl, df

출력:
  C:/donggeun/Gemma4/data/external/bcn20000_eval/
  ├── metadata.csv                        ← 평가용 메타데이터
  ├── nv/, mel/, bcc/, akiec/, bkl/, df/  ← 클래스별 폴더
"""

import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


# ============================================================
# 경로 설정 (실제 폴더 구조에 맞춤)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_METADATA_CSV = PROJECT_ROOT / "data" / "external" / "bcn20000" / "bcn20000_metadata_2026-05-07.csv"
 
# 다운받은 ISIC-images 폴더 경로
SOURCE_IMAGES_DIR = PROJECT_ROOT / "data" / "external" / "bcn20000" / "ISIC-images"

# 출력 경로
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "bcn20000_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 클래스별 샘플 수
SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42


# ============================================================
# 실제 BCN20000 분포 기반 매핑
# ============================================================
# vasc 클래스는 BCN20000에 없으므로 제외
# SCC도 akiec과 임상적으로 다르므로 제외 (보수적 접근)
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
    # vasc는 BCN20000에 충분한 샘플이 없어 제외
}


def find_image_path(isic_id: str, source_dir: Path) -> Path:
    """ISIC ID에 해당하는 이미지 파일 찾기."""
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png"]:
        path = source_dir / f"{isic_id}{ext}"
        if path.exists():
            return path
    return None


def main():
    print("=" * 60)
    print(" BCN20000 외부 검증 샘플 준비 (v2)")
    print("=" * 60)
    print(f" 메타데이터: {SOURCE_METADATA_CSV}")
    print(f" 이미지 폴더: {SOURCE_IMAGES_DIR}")
    print(f" 출력: {OUTPUT_DIR}")
    print("=" * 60)

    # 경로 검증
    if not SOURCE_METADATA_CSV.exists():
        print(f"\n[오류] 메타데이터 없음: {SOURCE_METADATA_CSV}")
        print("       파일이 Gemma4 루트에 있는지 확인하세요.")
        sys.exit(1)

    if not SOURCE_IMAGES_DIR.exists():
        print(f"\n[오류] 이미지 폴더 없음: {SOURCE_IMAGES_DIR}")
        sys.exit(1)

    # 메타데이터 읽기
    print("\n[1/4] 메타데이터 로드...")
    df = pd.read_csv(SOURCE_METADATA_CSV)
    print(f"  총 행: {len(df):,}")
    print(f"  diagnosis_3 있음: {df['diagnosis_3'].notna().sum():,}")

    # 이미지 폴더 검증
    print("\n[2/4] 이미지 폴더 스캔...")
    sample_images = list(SOURCE_IMAGES_DIR.glob("*.jpg"))[:5]
    if not sample_images:
        sample_images = list(SOURCE_IMAGES_DIR.glob("*.JPG"))[:5]
    print(f"  샘플 이미지 (첫 5개):")
    for img in sample_images:
        print(f"    {img.name}")

    total_images = len(list(SOURCE_IMAGES_DIR.glob("*.jpg"))) + \
                   len(list(SOURCE_IMAGES_DIR.glob("*.JPG")))
    print(f"  총 이미지: {total_images:,}장")

    # 클래스별 샘플링
    print("\n[3/4] 클래스별 샘플링 (목표: 클래스당 10장)")
    random.seed(RANDOM_SEED)

    selected_rows = []
    sampling_summary = {}

    for ham_class, bcn_diagnoses in HAM_TO_BCN_DIAGNOSIS.items():
        mask = df["diagnosis_3"].isin(bcn_diagnoses)
        candidates = df[mask].copy()

        candidate_count = len(candidates)
        print(f"\n  {ham_class} ← {', '.join(bcn_diagnoses)}")
        print(f"    매핑 후보: {candidate_count}장")

        if candidate_count > 0:
            unique_diagnoses = candidates["diagnosis_3"].value_counts()
            for diag, cnt in unique_diagnoses.items():
                print(f"      {diag}: {cnt}장")

        if candidate_count == 0:
            print(f"    [경고] 샘플 0장")
            sampling_summary[ham_class] = 0
            continue

        sample_n = min(SAMPLES_PER_CLASS, candidate_count)
        sampled = candidates.sample(n=sample_n, random_state=RANDOM_SEED)
        sampled["ham_class"] = ham_class
        selected_rows.append(sampled)
        sampling_summary[ham_class] = sample_n
        print(f"    샘플링: {sample_n}장")

    if not selected_rows:
        print("\n[오류] 샘플링 결과 없음")
        sys.exit(1)

    selected_df = pd.concat(selected_rows, ignore_index=True)
    print(f"\n  총 샘플: {len(selected_df)}장")

    # 이미지 복사
    print(f"\n[4/4] 이미지 복사")

    eval_records = []
    missing_images = []

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="복사"):
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

    # 요약
    print("\n" + "=" * 60)
    print(" 샘플 준비 완료")
    print("=" * 60)
    print(f"  메타데이터: {metadata_path}")
    print(f"  이미지 폴더: {OUTPUT_DIR}")
    if missing_images:
        print(f"  [경고] 누락 이미지 {len(missing_images)}장:")
        for mid in missing_images[:5]:
            print(f"    - {mid}")
        if len(missing_images) > 5:
            print(f"    ... 외 {len(missing_images) - 5}장")
    print()

    print(f"  클래스별 샘플:")
    for ham_class in HAM_TO_BCN_DIAGNOSIS:
        actual_count = len(eval_df[eval_df["ham_class"] == ham_class])
        target = SAMPLES_PER_CLASS
        status = "OK" if actual_count == target else "부족"
        print(f"    [{status}] {ham_class}: {actual_count}/{target}장")

    print(f"\n  총 평가 대상: {len(eval_df)}장")
    print(f"  예상 평가 시간: 약 {len(eval_df) * 65 / 60:.0f}분")
    print()
    print(f"  다음 단계: python test_batch_external_bcn20000.py")
    print(f"\n[참고]")
    print(f"  vasc 클래스는 BCN20000에 충분한 샘플이 없어 제외했습니다.")
    print(f"  SCC도 akiec과 임상적 차이가 있어 제외했습니다 (보수적 평가).")


if __name__ == "__main__":
    main()
