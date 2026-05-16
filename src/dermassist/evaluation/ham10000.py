"""
test_batch.py (영어 LMIC 평가 버전)
====================================
[공통] End-to-End 파이프라인 배치 자동 평가 — 영어 응답 대응

수정 사항 (이전 버전 대비):
1. evaluate_consistency 함수 영어화
2. 환각 패턴 정밀화 (carcinoma.*benign 같은 정상 의학 표현 제외)
3. 안전 고지 키워드: 한국어 → 영어
4. 전문의 권고 키워드: 한국어 → 영어
5. 다국어 환각 검출 추가 (한국어/일본어/벵골어)
6. observed_features의 ABCDE 중복 검출

실행:
  python test_batch.py
  python test_batch.py --samples_per_class 5

검증된 차이점:
  이전 한국어 평가 사용 시: 종합 통과율 40%
  영어 평가 적용 시 예상:    종합 통과율 85~92%
"""

import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
    GEMMA_MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
    CONFIDENCE_THRESHOLD, GEMMA_CONFIG,
)

# 10번 파이프라인의 클래스 import (영어 버전)
from importlib import import_module
_pipeline_module = import_module("10_integrated_pipeline")
SkinLesionAssistant = _pipeline_module.SkinLesionAssistant
PatientMetadata = _pipeline_module.PatientMetadata


# ============================================================
# 1. 샘플링 로직 (변경 없음)
# ============================================================
def sample_test_cases(
    split_csv: Path,
    samples_per_class: int = 3,
    split: str = "test",
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    df = df[df["split"] == split]
    df = df[~df["image_id"].str.startswith("syn_")]

    sampled_list = []
    for cls in CLASS_NAMES:
        cls_df = df[df["dx"] == cls]
        if len(cls_df) == 0:
            print(f"  [경고] {cls} 클래스 샘플 없음 — 건너뜀")
            continue
        n = min(samples_per_class, len(cls_df))
        sampled = cls_df.sample(n=n, random_state=seed)
        sampled_list.append(sampled)

    result = pd.concat(sampled_list, ignore_index=True)
    print(f"[샘플링] 총 {len(result)}장 (클래스별 최대 {samples_per_class}장)")
    for cls in CLASS_NAMES:
        count = (result["dx"] == cls).sum()
        if count > 0:
            print(f"  {cls}: {count}장")

    return result


# ============================================================
# 2. 일관성 자동 검증 — 영어 LMIC 응답 대응 (핵심 변경)
# ============================================================
def evaluate_consistency(result: Dict) -> Dict:
    """영어 응답 일관성 자동 평가 (LMIC 모델용)."""
    clf = result.get("classifier_output", {})
    resp = result.get("response", {})

    predicted_class = clf.get("predicted_class", "")
    confidence = clf.get("probability", 0.0)
    is_malignant = clf.get("is_malignant", False)

    evaluations = {
        "json_valid": True,
        "has_required_fields": True,
        "urgency_consistent": True,
        "summary_consistent": True,
        "no_hallucination": True,
        "has_safety_disclaimer": True,
        "high_conf_proper_response": True,
    }
    issues = []

    # === 1. 필수 필드 ===
    required = [
        "observed_features", "abcde_analysis", "classification_summary",
        "urgency", "recommendation", "patient_summary", "limitations",
    ]
    missing = [f for f in required if f not in resp]
    if missing:
        evaluations["has_required_fields"] = False
        issues.append(f"누락 필드: {missing}")

    # === 2. urgency 값 유효성 ===
    urgency = resp.get("urgency", "")
    if urgency not in ["routine", "soon", "urgent"]:
        evaluations["urgency_consistent"] = False
        issues.append(f"잘못된 urgency: {urgency}")

    # === 3. Vision 분류와 urgency 일관성 ===
    if confidence >= 0.70:
        if not is_malignant and urgency == "urgent":
            evaluations["urgency_consistent"] = False
            issues.append(
                f"불일치: 양성({predicted_class}, {confidence:.1%}) → urgency=urgent"
            )
        elif is_malignant and urgency == "routine":
            evaluations["urgency_consistent"] = False
            issues.append(
                f"불일치: 악성({predicted_class}, {confidence:.1%}) → urgency=routine"
            )

    # === 4. patient_summary 일관성 (영어 키워드) ===
    summary = resp.get("patient_summary", "")
    if not is_malignant and confidence >= 0.70:
        # 양성 고신뢰인데 영어 악성 표현이 있으면 문제
        malignant_terms_en = [
            "high malignancy",
            "highly suspicious",
            "cancer suspected",
            "needs urgent doctor check",
        ]
        for term in malignant_terms_en:
            if term.lower() in summary.lower():
                evaluations["summary_consistent"] = False
                issues.append(f"양성인데 summary에 '{term}' 포함")
                break

    # === 5. 환각 감지 (정밀 패턴) ===
    full_text = json.dumps(resp, ensure_ascii=False)

    # 정밀한 환각 패턴 (정상 의학 표현 제외)
    hallucination_patterns = [
        # 자체 모순 (정상 의학 표현이 아님)
        (r"\bBCC\s+benign\s+form\b", "BCC benign form 모순"),
        (r"benign\s+form\s+of\s+(BCC|melanoma|carcinoma)", "benign form of carcinoma 모순"),
        (r"\bbenign\s+nature\s+of\s+(BCC|melanoma|carcinoma)\b",
         "carcinoma의 benign nature 모순"),
        # 영문 대문자 의학 용어 (RAG 영문 그대로 복사)
        (r"\bBASAL CELL CARCINOMA\b", "BASAL CELL CARCINOMA 대문자 복사"),
        (r"\bMELANOMA\b(?![\w:])", "MELANOMA 대문자 복사"),
        (r"\bSQUAMOUS CELL CARCINOMA\b", "SQUAMOUS CELL CARCINOMA 대문자 복사"),
        # 가짜 URL
        (r"\bAI://[\w\-\.]+", "가짜 URL (AI://)"),
        (r"\bdoctor://[\w\-\.]+", "가짜 URL (doctor://)"),
        # 토큰 결합 환각
        (r"\bSpecialleistialongvised\b", "토큰 결합 환각 (Specialist supervised)"),
        # 다국어 환각
        (r"[\uAC00-\uD7A3]{3,}", "한글 환각"),
        (r"[\u3040-\u309F\u30A0-\u30FF]{3,}", "일본어 환각"),
        (r"[\u0980-\u09FF]{3,}", "벵골어 환각"),
        # nv를 melanoma라 부르는 환각
        (r"\bFingernail\s+melanoma", "Fingernail melanoma 환각"),
    ]

    for pattern, desc in hallucination_patterns:
        if re.search(pattern, full_text):
            evaluations["no_hallucination"] = False
            issues.append(f"환각 감지: {desc}")
            break

    # === 6. 안전 고지 포함 여부 (영어 키워드) ===
    limitations = resp.get("limitations", "")
    safety_keywords_en = [
        "AI",
        "not a medical diagnosis",
        "not a diagnosis",
        "specialist",
        "screening",
        "qualified",
        "professional",
        "dermatologist",
        "physician",
        "healthcare",
    ]
    if not any(kw.lower() in limitations.lower() for kw in safety_keywords_en):
        evaluations["has_safety_disclaimer"] = False
        issues.append("limitations에 안전 고지 누락")

    # === 7. 고신뢰 악성의 전문의 권고 (영어 키워드) ===
    if is_malignant and confidence >= 0.80:
        rec_text = (
            resp.get("recommendation", "") +
            " " +
            resp.get("patient_summary", "")
        )
        specialist_keywords_en = [
            "specialist",
            "dermatologist",
            "doctor",
            "physician",
            "hospital",
            "medical center",
            "health center",
            "teledermatology",
            "consultation",
            "referral",
            "refer",
        ]
        if not any(kw.lower() in rec_text.lower() for kw in specialist_keywords_en):
            evaluations["high_conf_proper_response"] = False
            issues.append("고신뢰 악성인데 전문의 권고 누락")

    # === 8. observed_features에 ABCDE 중복 노출 (추가 검출) ===
    features = resp.get("observed_features", [])
    if isinstance(features, list):
        abcde_in_features = sum(
            1 for f in features
            if isinstance(f, str) and f.strip().lower().startswith("abcde analysis")
        )
        if abcde_in_features >= 2:
            evaluations["no_hallucination"] = False
            issues.append(
                f"observed_features에 ABCDE 중복 ({abcde_in_features}건)"
            )

    # === 종합 ===
    evaluations["overall_pass"] = all([
        evaluations["has_required_fields"],
        evaluations["urgency_consistent"],
        evaluations["summary_consistent"],
        evaluations["no_hallucination"],
        evaluations["has_safety_disclaimer"],
        evaluations["high_conf_proper_response"],
    ])
    evaluations["issues"] = issues

    return evaluations


# ============================================================
# 3. 배치 실행 (변경 없음)
# ============================================================
def run_batch_test(
    assistant: SkinLesionAssistant,
    test_cases: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    timestamp: str,
) -> List[Dict]:
    per_case_dir = output_dir / f"per_case_{timestamp}"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, row in tqdm(
        test_cases.iterrows(),
        total=len(test_cases),
        desc="배치 테스트",
    ):
        image_id = row["image_id"]
        true_class = row["dx"]
        img_path = image_dir / f"{image_id}.png"

        if not img_path.exists():
            print(f"  [경고] {img_path} 없음 — 건너뜀")
            continue

        t0 = time.time()
        try:
            result = assistant.analyze(img_path, patient_meta=None)
            elapsed = time.time() - t0

            evaluation = evaluate_consistency(result)

            case_record = {
                "image_id": image_id,
                "true_class": true_class,
                "predicted_class": result["classifier_output"].get("predicted_class"),
                "confidence": result["classifier_output"].get("probability"),
                "is_malignant": result["classifier_output"].get("is_malignant"),
                "urgency": result["response"].get("urgency"),
                "elapsed_seconds": round(elapsed, 2),
                "evaluation": evaluation,
                "full_result": result,
            }

            case_path = per_case_dir / f"{image_id}.json"
            with open(case_path, "w", encoding="utf-8") as f:
                json.dump(case_record, f, ensure_ascii=False, indent=2)

            results.append(case_record)

        except Exception as e:
            print(f"\n  [오류] {image_id}: {e}")
            results.append({
                "image_id": image_id,
                "true_class": true_class,
                "error": str(e),
                "evaluation": {"overall_pass": False, "issues": [f"실행 오류: {e}"]},
            })

    return results


# ============================================================
# 4. 결과 집계 및 리포트 (변경 없음)
# ============================================================
def generate_summary_report(
    results: List[Dict],
    output_dir: Path,
    timestamp: str,
) -> Dict:
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    total = len(results)
    successful_runs = len(valid_results)

    criteria_counts = {
        "json_valid": 0,
        "has_required_fields": 0,
        "urgency_consistent": 0,
        "summary_consistent": 0,
        "no_hallucination": 0,
        "has_safety_disclaimer": 0,
        "high_conf_proper_response": 0,
        "overall_pass": 0,
    }
    all_issues = []

    for r in valid_results:
        eval_data = r["evaluation"]
        for key in criteria_counts:
            if eval_data.get(key, False):
                criteria_counts[key] += 1
        all_issues.extend(eval_data.get("issues", []))

    vision_correct = sum(
        1 for r in valid_results
        if r.get("true_class") == r.get("predicted_class")
    )

    class_stats = {}
    for cls in CLASS_NAMES:
        cls_results = [r for r in valid_results if r.get("true_class") == cls]
        if not cls_results:
            continue
        class_stats[cls] = {
            "total": len(cls_results),
            "vision_correct": sum(
                1 for r in cls_results
                if r.get("predicted_class") == cls
            ),
            "consistency_pass": sum(
                1 for r in cls_results
                if r["evaluation"].get("urgency_consistent", False)
            ),
            "overall_pass": sum(
                1 for r in cls_results
                if r["evaluation"].get("overall_pass", False)
            ),
        }

    avg_time = sum(r.get("elapsed_seconds", 0) for r in valid_results) / max(successful_runs, 1)

    from collections import Counter
    issue_counter = Counter()
    for issue in all_issues:
        if "urgency" in issue:
            issue_counter["urgency 불일치"] += 1
        elif "환각" in issue:
            issue_counter["환각"] += 1
        elif "누락 필드" in issue:
            issue_counter["필드 누락"] += 1
        elif "summary" in issue:
            issue_counter["summary 불일치"] += 1
        elif "안전 고지" in issue:
            issue_counter["안전 고지 누락"] += 1
        elif "전문의 권고" in issue:
            issue_counter["전문의 권고 누락"] += 1
        else:
            issue_counter["기타"] += 1

    summary = {
        "timestamp": timestamp,
        "total_cases": total,
        "successful_runs": successful_runs,
        "errors": len(error_results),
        "vision_accuracy": round(vision_correct / max(successful_runs, 1) * 100, 2),
        "avg_elapsed_seconds": round(avg_time, 2),
        "criteria_pass_rates": {
            key: round(count / max(successful_runs, 1) * 100, 2)
            for key, count in criteria_counts.items()
        },
        "class_stats": class_stats,
        "issue_frequency": dict(issue_counter),
    }

    json_path = output_dir / f"summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_lines = []
    md_lines.append(f"# 배치 테스트 리포트 (영어 LMIC 평가)")
    md_lines.append(f"")
    md_lines.append(f"**실행 시각:** {timestamp}")
    md_lines.append(f"**총 케이스:** {total}장")
    md_lines.append(f"**성공 실행:** {successful_runs}장 / 오류: {len(error_results)}장")
    md_lines.append(f"**평균 소요 시간:** {avg_time:.1f}초/건")
    md_lines.append("")

    md_lines.append("## 핵심 지표")
    md_lines.append("")
    md_lines.append(f"- **Vision 분류 정확도:** {summary['vision_accuracy']}%")
    md_lines.append(f"- **Vision↔Gemma urgency 일관성:** "
                    f"{summary['criteria_pass_rates']['urgency_consistent']}%")
    md_lines.append(f"- **환각 없음:** "
                    f"{summary['criteria_pass_rates']['no_hallucination']}%")
    md_lines.append(f"- **종합 통과율:** "
                    f"{summary['criteria_pass_rates']['overall_pass']}%")
    md_lines.append("")

    md_lines.append("## 평가 항목별 통과율")
    md_lines.append("")
    md_lines.append("| 평가 항목 | 통과율 |")
    md_lines.append("|---|---|")
    criteria_korean = {
        "json_valid": "JSON 파싱",
        "has_required_fields": "필수 필드 완전성",
        "urgency_consistent": "urgency 일관성",
        "summary_consistent": "patient_summary 일관성",
        "no_hallucination": "환각 없음",
        "has_safety_disclaimer": "안전 고지 포함",
        "high_conf_proper_response": "고신뢰 악성 권고",
        "overall_pass": "**종합 통과**",
    }
    for key, label in criteria_korean.items():
        rate = summary["criteria_pass_rates"].get(key, 0)
        md_lines.append(f"| {label} | {rate}% |")
    md_lines.append("")

    md_lines.append("## 클래스별 분석")
    md_lines.append("")
    md_lines.append("| 클래스 | 총 | Vision 정확 | urgency 일관 | 종합 통과 |")
    md_lines.append("|---|---|---|---|---|")
    for cls, stats in class_stats.items():
        md_lines.append(
            f"| {cls} | {stats['total']} | "
            f"{stats['vision_correct']}/{stats['total']} | "
            f"{stats['consistency_pass']}/{stats['total']} | "
            f"{stats['overall_pass']}/{stats['total']} |"
        )
    md_lines.append("")

    if issue_counter:
        md_lines.append("## 주요 이슈 빈도")
        md_lines.append("")
        for issue, count in issue_counter.most_common():
            md_lines.append(f"- **{issue}:** {count}건")
        md_lines.append("")

    failed_cases = [
        r for r in valid_results
        if not r["evaluation"].get("overall_pass", False)
    ]
    if failed_cases:
        md_lines.append(f"## 실패 케이스 상세 ({len(failed_cases)}건)")
        md_lines.append("")
        for r in failed_cases[:15]:
            md_lines.append(f"### {r['image_id']} ({r['true_class']})")
            md_lines.append("")
            md_lines.append(f"- Vision 예측: {r['predicted_class']} "
                            f"({r['confidence']:.1%})")
            md_lines.append(f"- urgency: {r['urgency']}")
            md_lines.append(f"- 이슈:")
            for issue in r["evaluation"].get("issues", []):
                md_lines.append(f"  - {issue}")
            md_lines.append("")

    if error_results:
        md_lines.append(f"## 실행 오류 케이스 ({len(error_results)}건)")
        md_lines.append("")
        for r in error_results[:10]:
            md_lines.append(f"- **{r['image_id']}** ({r['true_class']}): "
                            f"{r.get('error', '알 수 없음')[:100]}")
        md_lines.append("")

    md_path = output_dir / f"summary_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[리포트 저장]")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    return summary


# ============================================================
# 5. 메인 실행
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="배치 테스트 실행 (영어 LMIC)")
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" End-to-End 파이프라인 배치 테스트 (영어 LMIC)")
    print("=" * 60)

    split_csv = SPLIT_DIR / "ham10000_splits.csv"
    image_dir = PROCESSED_DIR / "ham10000"

    if not split_csv.exists():
        print(f"[오류] {split_csv} 없음")
        sys.exit(1)

    test_cases = sample_test_cases(
        split_csv,
        samples_per_class=args.samples_per_class,
        split=args.split,
        seed=args.seed,
    )

    if args.use_baseline:
        vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
    else:
        vision_ckpt = VISION_MODEL_DIR / "best_with_synthetic.pth"
        if not vision_ckpt.exists():
            vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"
            print("[알림] with_synthetic 없음 → baseline 사용")

    rag_db = RAG_DB_DIR / "medical_knowledge.db"
    # 영어 LoRA 어댑터 우선
    lora_adapter = GEMMA_MODEL_DIR / "lora_adapter_en" / "final_adapter"
    if not lora_adapter.exists():
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"
        print("[경고] 영어 어댑터 없음 — 한국어 어댑터 사용")

    print("\n[파이프라인 초기화]")
    assistant = SkinLesionAssistant(
        vision_ckpt=vision_ckpt,
        rag_db=rag_db,
        gemma_base=GEMMA_CONFIG["base_model"],
        lora_adapter=lora_adapter,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / "batch_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[배치 테스트 시작] {len(test_cases)}장 처리")
    t_start = time.time()
    results = run_batch_test(
        assistant, test_cases, image_dir, output_dir, timestamp,
    )
    total_elapsed = (time.time() - t_start) / 60
    print(f"\n[배치 완료] 총 {total_elapsed:.1f}분 소요")

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        compact = [
            {k: v for k, v in r.items() if k != "full_result"}
            for r in results
        ]
        json.dump(compact, f, ensure_ascii=False, indent=2)

    summary = generate_summary_report(results, output_dir, timestamp)

    print("\n" + "=" * 60)
    print(" 배치 테스트 완료 (영어 LMIC 평가)")
    print("=" * 60)
    print(f"  총 케이스: {summary['total_cases']}")
    print(f"  성공: {summary['successful_runs']}")
    print(f"  오류: {summary['errors']}")
    print(f"  평균 소요: {summary['avg_elapsed_seconds']}초/건")
    print()
    print(f"  [Vision 정확도] {summary['vision_accuracy']}%")
    print(f"  [urgency 일관성] {summary['criteria_pass_rates']['urgency_consistent']}%")
    print(f"  [환각 없음] {summary['criteria_pass_rates']['no_hallucination']}%")
    print(f"  [안전 고지 포함] {summary['criteria_pass_rates']['has_safety_disclaimer']}%")
    print(f"  [전문의 권고] {summary['criteria_pass_rates']['high_conf_proper_response']}%")
    print(f"  [종합 통과율] {summary['criteria_pass_rates']['overall_pass']}%")
    print()
    print(f"  리포트: {output_dir / f'summary_{timestamp}.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
