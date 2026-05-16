"""
validate_demo_scenarios.py
===========================
시연 영상 녹화용 케이스 사전 검증 스크립트.

목적:
  1. 후보 이미지들로 다양한 환자 시나리오 테스트
  2. 반복 일관성 확인 (동일 케이스 3회 실행)
  3. 시각적 임팩트 + 응답 품질 평가
  4. 가장 좋은 케이스 자동 추천

사용:
  python validate_demo_scenarios.py

출력:
  outputs/demo_validation/
  ├── results_<timestamp>.json     ← 전체 결과
  ├── recommended_<timestamp>.md   ← 추천 케이스 (영상 녹화용)
  └── per_run_<timestamp>/         ← 개별 실행 결과
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    PROCESSED_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
    GEMMA_MODEL_DIR, OUTPUT_DIR, GEMMA_CONFIG,
)

from importlib import import_module
_pipeline_module = import_module("10_integrated_pipeline")
SkinLesionAssistant = _pipeline_module.SkinLesionAssistant
PatientMetadata = _pipeline_module.PatientMetadata


# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
HAM_IMAGE_DIR = PROCESSED_DIR / "ham10000"
OUTPUT_DIR_VAL = OUTPUT_DIR / "demo_validation"
OUTPUT_DIR_VAL.mkdir(parents=True, exist_ok=True)

REPETITIONS = 3  # 일관성 확인을 위한 반복 횟수


# ============================================================
# 후보 이미지 + 환자 시나리오 정의
# ============================================================
# HAM10000 test split의 이전 평가에서 좋은 결과를 보인 이미지들
# (실제 이미지 ID는 본인이 가지고 있는 데이터에 맞게 조정 필요)

DEMO_SCENARIOS = {
    "A_benign_safe": {
        "description": "양성 nv 케이스 - 안전한 데모용",
        "expected_urgency": "routine",
        "expected_class": "nv",
        # nv 클래스의 후보 이미지들
        "image_candidates": [
            # 이전 평가에서 nv로 정확히 분류된 케이스들
            "ISIC_0024306", "ISIC_0024428", "ISIC_0027850",
            "ISIC_0029305", "ISIC_0030174",
        ],
        "patient_meta": {
            "age": 50,
            "sex": "male",
            "body_site": "back",
            "duration_months": 12,
            "symptoms": "asymptomatic, stable lesion",
            "context": "general LMIC patient",
            "risk_factor": "limited healthcare access",
            "skin_type": "Fitzpatrick IV",
            "resource_constraint": "Standard primary care setting with limited specialist access",
        },
    },

    "B_albinism_malignant": {
        "description": "알비노 + BCC - LMIC 핵심 임팩트 시연",
        "expected_urgency": "urgent",
        "expected_class": "bcc",
        # bcc 클래스의 후보 이미지들 (이전 평가에서 80%+ 신뢰도)
        "image_candidates": [
            "ISIC_0024372", "ISIC_0024388", "ISIC_0025780",
            "ISIC_0028469", "ISIC_0030042",
        ],
        "patient_meta": {
            "age": 22,
            "sex": "female",
            "body_site": "scalp",
            "duration_months": 8,
            "symptoms": "non-healing ulceration, rapid growth over past 3 months, occasional bleeding",
            "context": "patient with albinism, multiple sun-damaged areas",
            "risk_factor": "OCA (oculocutaneous albinism)",
            "skin_type": "Fitzpatrick I (albinism)",
            "resource_constraint": "Patient travel to nearest dermatologist requires 200+ km journey",
        },
    },

    "C_outdoor_akiec": {
        "description": "농촌 노인 + 광선각화증 - 현실적 LMIC 케이스",
        "expected_urgency": "soon",
        "expected_class": "akiec",
        "image_candidates": [
            "ISIC_0024800", "ISIC_0027345", "ISIC_0028104",
            "ISIC_0030562", "ISIC_0033217",
        ],
        "patient_meta": {
            "age": 70,
            "sex": "male",
            "body_site": "face",
            "duration_months": 24,
            "symptoms": "scaly rough surface, intermittent itching, slowly enlarging",
            "context": "rural farmer with chronic UV exposure",
            "risk_factor": "occupational sun exposure",
            "skin_type": "Fitzpatrick IV",
            "resource_constraint": "Specialist referral wait time typically 4-8 weeks",
        },
    },
}


# ============================================================
# 평가 함수
# ============================================================
def find_image_path(image_id: str) -> Path:
    """HAM10000에서 이미지 찾기."""
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        path = HAM_IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path

    # 하위 폴더 검색
    for ext in [".png", ".jpg"]:
        matches = list(HAM_IMAGE_DIR.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]

    return None


def evaluate_demo_quality(
    response: Dict,
    classifier_output: Dict,
    expected_urgency: str,
    expected_class: str,
) -> Dict:
    """시연 품질 종합 평가."""
    quality = {
        "vision_correct": False,
        "high_confidence": False,
        "urgency_match": False,
        "lmic_keywords_present": False,
        "rec_synchronized": True,
        "no_hallucination": True,
        "summary_quality_good": False,
        "score": 0,
    }

    # 1. Vision 정확도
    predicted = classifier_output.get("predicted_class", "")
    confidence = classifier_output.get("probability", 0.0)
    if predicted == expected_class:
        quality["vision_correct"] = True
        quality["score"] += 25

    # 2. 고신뢰도 (시연 안정성)
    if confidence >= 0.80:
        quality["high_confidence"] = True
        quality["score"] += 15
    elif confidence >= 0.70:
        quality["score"] += 10

    # 3. urgency 매치
    actual_urgency = response.get("urgency", "")
    if actual_urgency == expected_urgency:
        quality["urgency_match"] = True
        quality["score"] += 20

    # 4. LMIC 키워드 (시연 임팩트)
    rec = response.get("recommendation", "")
    lmic_keywords = [
        "teledermatology", "telemedicine", "rural", "remote",
        "specialist access", "200 km", "200km", "limited access",
        "low-resource", "primary care", "albinism", "OCA",
    ]
    if any(kw.lower() in rec.lower() for kw in lmic_keywords):
        quality["lmic_keywords_present"] = True
        quality["score"] += 15

    # 5. recommendation-urgency 동기화
    rec_lower = rec.lower()
    if actual_urgency == "routine":
        if "urgent" in rec_lower and "specialist" in rec_lower:
            quality["rec_synchronized"] = False
            quality["score"] -= 20
    elif actual_urgency == "urgent":
        if "routine" in rec_lower or "monitor monthly" in rec_lower:
            quality["rec_synchronized"] = False
            quality["score"] -= 20

    # 6. 환각 없음
    full_text = " ".join([
        rec, response.get("patient_summary", ""),
        response.get("limitations", ""),
    ])
    import re
    hallucination_patterns = [
        r"basal cell carcinoma.*BENIGN",
        r"melanoma.*BENIGN",
        r"benign.*urgent",
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, full_text, re.IGNORECASE):
            quality["no_hallucination"] = False
            quality["score"] -= 30
            break

    # 7. 환자 요약 품질 (시연용)
    summary = response.get("patient_summary", "")
    if (
        len(summary) >= 100
        and len(summary) <= 400  # 너무 길면 시연에서 다 안 읽힘
        and "doctor" not in summary.lower()  # health worker 권장
    ):
        quality["summary_quality_good"] = True
        quality["score"] += 10
    elif len(summary) >= 50:
        quality["score"] += 5

    return quality


def run_scenario(
    assistant,
    scenario_name: str,
    scenario_config: Dict,
    image_id: str,
    repetition: int,
) -> Dict:
    """단일 시나리오 1회 실행."""
    image_path = find_image_path(image_id)
    if image_path is None:
        return {"error": f"이미지 없음: {image_id}"}

    patient_meta = PatientMetadata(**scenario_config["patient_meta"])

    t_start = time.time()
    result = assistant.analyze(image_path, patient_meta)
    elapsed = time.time() - t_start

    quality = evaluate_demo_quality(
        result["response"],
        result["classifier_output"],
        scenario_config["expected_urgency"],
        scenario_config["expected_class"],
    )

    return {
        "scenario": scenario_name,
        "image_id": image_id,
        "repetition": repetition,
        "predicted_class": result["classifier_output"].get("predicted_class"),
        "confidence": round(result["classifier_output"].get("probability", 0.0), 3),
        "urgency": result["response"].get("urgency"),
        "recommendation_preview": result["response"].get("recommendation", "")[:200],
        "patient_summary_preview": result["response"].get("patient_summary", "")[:200],
        "quality": quality,
        "score": quality["score"],
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print(" 시연 영상 케이스 사전 검증")
    print("=" * 60)
    print(f" 시나리오 수: {len(DEMO_SCENARIOS)}")
    print(f" 시나리오별 후보 이미지: 5장")
    print(f" 반복 횟수: {REPETITIONS}회")
    total_runs = sum(len(s["image_candidates"]) for s in DEMO_SCENARIOS.values()) * REPETITIONS
    print(f" 총 실행: {total_runs}회 (예상 약 {total_runs * 65 / 60:.0f}분)")
    print("=" * 60)

    # 모델 로드
    print("\n[1/3] 모델 로드...")
    vision_ckpt = VISION_MODEL_DIR / "best_with_synthetic.pth"
    if not vision_ckpt.exists():
        vision_ckpt = VISION_MODEL_DIR / "best_baseline.pth"

    rag_db = RAG_DB_DIR / "medical_knowledge.db"
    lora_adapter = GEMMA_MODEL_DIR / "lora_adapter_en" / "final_adapter"

    assistant = SkinLesionAssistant(
        vision_ckpt=vision_ckpt,
        rag_db=rag_db,
        gemma_base=GEMMA_CONFIG["base_model"],
        lora_adapter=lora_adapter,
    )
    print("  완료")

    # 출력 디렉터리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_run_dir = OUTPUT_DIR_VAL / f"per_run_{timestamp}"
    per_run_dir.mkdir(exist_ok=True)

    print(f"\n[2/3] 시나리오 검증 시작")
    print("-" * 60)

    # 결과 수집
    all_results = []
    by_scenario = defaultdict(list)

    for scenario_name, scenario_config in DEMO_SCENARIOS.items():
        print(f"\n=== {scenario_name} ===")
        print(f"  설명: {scenario_config['description']}")

        for image_id in scenario_config["image_candidates"]:
            print(f"\n  [{image_id}]")

            image_results = []
            for rep in range(1, REPETITIONS + 1):
                print(f"    실행 {rep}/{REPETITIONS}...", end=" ", flush=True)
                result = run_scenario(
                    assistant, scenario_name, scenario_config,
                    image_id, rep,
                )

                if "error" in result:
                    print(f"오류: {result['error']}")
                    continue

                print(
                    f"vision={result['predicted_class']} ({result['confidence']:.0%}), "
                    f"urgency={result['urgency']}, "
                    f"score={result['score']}"
                )

                image_results.append(result)
                all_results.append(result)

                # 개별 실행 저장
                run_file = per_run_dir / f"{scenario_name}_{image_id}_rep{rep}.json"
                with open(run_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            if image_results:
                by_scenario[scenario_name].append({
                    "image_id": image_id,
                    "runs": image_results,
                    "avg_score": np.mean([r["score"] for r in image_results]),
                    "consistency": calculate_consistency(image_results),
                })

    # 분석 및 추천
    print("\n[3/3] 결과 분석 및 추천")
    print("-" * 60)

    recommendations = analyze_and_recommend(by_scenario)

    # 저장
    summary_path = OUTPUT_DIR_VAL / f"results_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "all_results": all_results,
            "recommendations": recommendations,
        }, f, ensure_ascii=False, indent=2, default=str)

    # 추천 마크다운
    recommended_path = OUTPUT_DIR_VAL / f"recommended_{timestamp}.md"
    write_recommendations_md(recommended_path, recommendations, by_scenario)

    print(f"\n[저장]")
    print(f"  전체 결과: {summary_path}")
    print(f"  추천 케이스: {recommended_path}")

    # 콘솔에 추천 표시
    print("\n" + "=" * 60)
    print(" 시연용 추천 케이스")
    print("=" * 60)
    for scenario, rec in recommendations.items():
        print(f"\n[{scenario}] {rec['description']}")
        if rec["recommended"]:
            best = rec["recommended"]
            print(f"  추천 이미지: {best['image_id']}")
            print(f"  평균 점수: {best['avg_score']:.1f}/100")
            print(f"  일관성: {best['consistency']:.0f}%")
            print(f"  Vision: {best['vision_summary']}")
            print(f"  Urgency: {best['urgency_summary']}")
        else:
            print(f"  [경고] 추천 가능한 케이스 없음")


def calculate_consistency(runs: List[Dict]) -> float:
    """3회 실행의 일관성 계산 (0-100)."""
    if len(runs) < 2:
        return 100.0

    # urgency 일치율
    urgencies = [r["urgency"] for r in runs]
    urgency_consistent = len(set(urgencies)) == 1

    # 신뢰도 변동 (낮을수록 일관됨)
    confidences = [r["confidence"] for r in runs]
    conf_std = np.std(confidences)

    # 점수 변동
    scores = [r["score"] for r in runs]
    score_std = np.std(scores)

    # 종합 일관성 (urgency 일치 + 변동성 낮음)
    consistency = 0
    if urgency_consistent:
        consistency += 50
    consistency += max(0, 30 - conf_std * 100)
    consistency += max(0, 20 - score_std)
    return min(100, consistency)


def analyze_and_recommend(by_scenario: Dict) -> Dict:
    """시나리오별 최고 케이스 추천."""
    recommendations = {}

    for scenario_name, image_results in by_scenario.items():
        scenario_config = DEMO_SCENARIOS[scenario_name]

        # 점수 + 일관성 종합 정렬
        sorted_images = sorted(
            image_results,
            key=lambda x: (x["avg_score"], x["consistency"]),
            reverse=True,
        )

        if not sorted_images:
            recommendations[scenario_name] = {
                "description": scenario_config["description"],
                "recommended": None,
                "alternatives": [],
            }
            continue

        best = sorted_images[0]
        runs = best["runs"]

        # 요약 정보 생성
        urgencies = [r["urgency"] for r in runs]
        urgency_counts = defaultdict(int)
        for u in urgencies:
            urgency_counts[u] += 1
        urgency_summary = ", ".join(
            f"{u}: {c}/{len(runs)}" for u, c in urgency_counts.items()
        )

        confidences = [r["confidence"] for r in runs]
        vision_summary = (
            f"avg {np.mean(confidences):.0%}, "
            f"range {min(confidences):.0%}-{max(confidences):.0%}"
        )

        recommendations[scenario_name] = {
            "description": scenario_config["description"],
            "recommended": {
                "image_id": best["image_id"],
                "avg_score": best["avg_score"],
                "consistency": best["consistency"],
                "vision_summary": vision_summary,
                "urgency_summary": urgency_summary,
                "expected_class": scenario_config["expected_class"],
                "expected_urgency": scenario_config["expected_urgency"],
            },
            "alternatives": [
                {
                    "image_id": alt["image_id"],
                    "avg_score": alt["avg_score"],
                    "consistency": alt["consistency"],
                }
                for alt in sorted_images[1:3]
            ],
            "patient_meta": scenario_config["patient_meta"],
        }

    return recommendations


def write_recommendations_md(path: Path, recommendations: Dict, by_scenario: Dict):
    """추천 결과를 마크다운으로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 시연 영상 녹화용 추천 케이스\n\n")
        f.write("**자동 검증 결과 기반 권장 시연 시나리오**\n\n")
        f.write("---\n\n")

        for scenario_name, rec in recommendations.items():
            f.write(f"## {scenario_name}\n\n")
            f.write(f"**설명:** {rec['description']}\n\n")

            if not rec["recommended"]:
                f.write("**[경고]** 추천 가능한 케이스 없음. 대체 이미지 필요.\n\n")
                continue

            best = rec["recommended"]
            f.write(f"### 추천 케이스\n\n")
            f.write(f"- **이미지:** `{best['image_id']}`\n")
            f.write(f"- **평균 점수:** {best['avg_score']:.1f}/100\n")
            f.write(f"- **일관성:** {best['consistency']:.0f}%\n")
            f.write(f"- **Vision:** 예측 {best['expected_class']}, {best['vision_summary']}\n")
            f.write(f"- **Urgency:** 기대 {best['expected_urgency']}, "
                   f"실제 {best['urgency_summary']}\n\n")

            f.write(f"### 환자 정보 (시연 입력값)\n\n")
            pm = rec["patient_meta"]
            f.write(f"| 항목 | 값 |\n|---|---|\n")
            f.write(f"| Age | {pm['age']} |\n")
            f.write(f"| Sex | {pm['sex']} |\n")
            f.write(f"| Body Site | {pm['body_site']} |\n")
            f.write(f"| Duration | {pm['duration_months']} months |\n")
            f.write(f"| Symptoms | {pm['symptoms']} |\n")
            f.write(f"| Patient Profile | {pm['context']} |\n")
            f.write(f"| Resource | {pm['resource_constraint']} |\n\n")

            if rec["alternatives"]:
                f.write(f"### 대체 후보\n\n")
                for alt in rec["alternatives"]:
                    f.write(f"- `{alt['image_id']}`: "
                           f"점수 {alt['avg_score']:.1f}, "
                           f"일관성 {alt['consistency']:.0f}%\n")
                f.write("\n")

            f.write("---\n\n")

        f.write("\n## 시연 영상 시나리오 권장\n\n")
        f.write("**메인 시연: B (알비노 + BCC)**\n")
        f.write("- LMIC 임팩트 메시지와 가장 직접 연결\n")
        f.write("- Urgent 빨간 뱃지로 시각적 임팩트\n")
        f.write("- teledermatology 권고가 LMIC 컨텍스트 강조\n\n")
        f.write("**보조 시연: A (양성 nv)**\n")
        f.write("- 시스템이 false alarm 없이 안정적임을 입증\n")
        f.write("- Routine 초록 뱃지로 안전성 어필\n\n")
        f.write("**선택 시연: C (농촌 노인)**\n")
        f.write("- 더 많은 시간이 있다면 다양성 추가\n")


if __name__ == "__main__":
    main()
