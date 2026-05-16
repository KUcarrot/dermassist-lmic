"""
04_gemma_data_design.py
=======================
[Gemma 트랙 - 1주차] LoRA 학습 데이터 설계 + 템플릿 생성

실행: python3 04_gemma_data_design.py

목적:
  Gemma 4 E4B LoRA 파인튜닝용 학습 데이터의 구조와 포맷을 설계.
  Vision Classifier 출력 + 환자 메타데이터 + RAG 컨텍스트
  → 구조화된 의료 보조 응답 생성을 학습시키기 위한 데이터.

이 스크립트는 1주차에서 "설계"만 수행하며,
실제 대량 데이터 생성은 2주차에 진행합니다.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    RAG_DIR, GEMMA_MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
    CONFIDENCE_THRESHOLD,
)


# ============================================================
# 1. 학습 샘플 스키마 정의
# ============================================================
@dataclass
class ClassifierOutput:
    """Vision Classifier 출력 (Gemma 입력의 일부)."""
    predicted_class: str           # 7-class 중 예측 결과
    probability: float             # 예측 확률 (0~1)
    top3: List[Dict[str, float]]   # 상위 3개 클래스 확률
    is_malignant: bool             # 악성/양성 이진 판정
    grad_cam_description: str      # Grad-CAM 관찰 요약 (텍스트)


@dataclass
class PatientMetadata:
    """환자 컨텍스트 정보."""
    age: Optional[int] = None
    sex: Optional[str] = None      # "male" | "female"
    body_site: Optional[str] = None  # 병변 위치
    duration_months: Optional[int] = None  # 증상 지속 기간
    symptoms: Optional[str] = None   # 자유 텍스트 (가려움, 출혈 등)
    family_history: Optional[str] = None  # 가족력


@dataclass
class ExpectedResponse:
    """Gemma가 생성해야 할 목표 응답."""
    observed_features: List[str]   # ABCDE 기반 관찰 소견
    abcde_analysis: Dict[str, str] # A/B/C/D/E 각 항목 분석
    classification_summary: str    # 분류 결과 요약
    evidence_sources: List[str]    # RAG에서 가져온 근거
    recommendation: str            # 권고 행동
    urgency: str                   # "routine" | "soon" | "urgent"
    patient_summary: str           # 환자용 쉬운 언어 요약
    limitations: str               # 한계/면책 고지


@dataclass
class TrainingSample:
    """완전한 학습 샘플."""
    sample_id: str
    image_id: str                  # HAM10000 이미지 ID (참조)
    classifier_output: ClassifierOutput
    patient_metadata: PatientMetadata
    rag_context: str               # RAG 검색 결과 (텍스트)
    expected_response: ExpectedResponse


# ============================================================
# 2. 시스템 프롬프트 (Gemma 학습/추론 공통)
# ============================================================
SYSTEM_PROMPT = """당신은 피부 병변 스크리닝 보조 AI입니다.
전문 Vision Classifier의 분석 결과, 환자 정보, 의학 참고자료를 종합하여
구조화된 소견을 제공합니다.

[핵심 원칙]
1. 진단을 내리지 않습니다. 관찰된 특징과 가능성을 설명합니다.
2. 신뢰도가 낮은 경우 반드시 전문의 상담을 권고합니다.
3. ABCDE 기준에 따른 체계적 분석을 제공합니다.
4. 환자가 이해할 수 있는 쉬운 언어 요약을 항상 포함합니다.
5. AI 분석의 한계를 솔직하게 명시합니다.

[출력 형식]
반드시 아래 JSON 스키마를 따르세요:
{
  "observed_features": ["특징1", "특징2", ...],
  "abcde_analysis": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "classification_summary": "...",
  "evidence_sources": ["출처1", "출처2"],
  "recommendation": "...",
  "urgency": "routine|soon|urgent",
  "patient_summary": "...",
  "limitations": "..."
}"""


# ============================================================
# 3. 사용자 프롬프트 템플릿
# ============================================================
USER_PROMPT_TEMPLATE = """## 병변 분석 요청

### 1. Vision Classifier 분석 결과
- 예측 클래스: {predicted_class}
- 신뢰도: {probability:.1%}
- 상위 3 후보: {top3_str}
- 악성/양성 판정: {malignant_str}
- Grad-CAM 관찰: {grad_cam_description}

### 2. 환자 정보
- 나이: {age}세 | 성별: {sex}
- 병변 위치: {body_site}
- 증상 기간: {duration_months}개월
- 증상: {symptoms}
- 가족력: {family_history}

### 3. 의학 참고자료 (RAG 검색)
{rag_context}

### 4. 요청
위 정보를 종합하여 구조화된 소견을 JSON 형식으로 제공하세요.
환자용 쉬운 요약도 반드시 포함하세요."""


# ============================================================
# 4. 시나리오별 학습 샘플 생성
# ============================================================
def create_melanoma_high_confidence_sample() -> TrainingSample:
    """시나리오 1: 멜라노마 고신뢰도 — 긴급 의뢰 권고."""
    return TrainingSample(
        sample_id="mel_high_001",
        image_id="ISIC_0024306",
        classifier_output=ClassifierOutput(
            predicted_class="mel",
            probability=0.89,
            top3=[
                {"mel": 0.89}, {"nv": 0.06}, {"bkl": 0.03}
            ],
            is_malignant=True,
            grad_cam_description="병변 중앙부와 불규칙한 경계 영역에 높은 활성화 집중",
        ),
        patient_metadata=PatientMetadata(
            age=58, sex="male", body_site="등 상부",
            duration_months=8, symptoms="최근 3개월간 크기 증가, 간헐적 가려움",
            family_history="어머니 멜라노마 병력",
        ),
        rag_context=(
            "[출처: ABCDE 자가검진 규칙] ABCDE 규칙은 멜라노마 의심 병변을 식별하는 "
            "임상 가이드라인이다. E(변화)가 가장 중요한 단일 지표로 여겨진다.\n"
            "[출처: 전문의 의뢰 기준] 50세 이상에서 새로 발생한 색소성 병변, "
            "환자 또는 가족이 멜라노마 병력이 있는 경우 의뢰 권고."
        ),
        expected_response=ExpectedResponse(
            observed_features=[
                "비대칭적 형태",
                "불규칙한 경계",
                "다양한 색조 혼재 (Vision Classifier 활성화 패턴)",
                "최근 크기 변화 보고",
            ],
            abcde_analysis={
                "A": "비대칭 — Grad-CAM에서 비대칭적 활성화 관찰",
                "B": "경계 불규칙 — 경계 영역에 높은 활성화",
                "C": "색상 정보는 분류기 확률로 간접 추정; 다색조 가능성",
                "D": "직경 정보 직접 확인 불가 — 환자 보고 필요",
                "E": "변화 있음 — 환자가 3개월간 크기 증가를 보고",
            },
            classification_summary=(
                "Vision Classifier가 89% 신뢰도로 멜라노마를 예측했습니다. "
                "환자의 연령(58세), 가족력(모친 멜라노마), 최근 변화 이력을 "
                "고려하면 위험도가 높은 소견입니다."
            ),
            evidence_sources=[
                "ABCDE 자가검진 규칙",
                "전문의 의뢰 기준",
            ],
            recommendation=(
                "피부과 전문의 긴급 의뢰를 권고합니다 (2주 이내). "
                "더모스코피 검사 및 조직검사(생검)가 필요할 수 있습니다."
            ),
            urgency="urgent",
            patient_summary=(
                "점의 모양과 최근 변화가 전문의 확인이 필요한 소견을 보이고 있습니다. "
                "어머니의 병력도 고려할 때, 가능한 빨리(2주 이내) "
                "피부과를 방문하시는 것이 좋겠습니다. "
                "이 결과는 AI 보조 분석이며, 정확한 진단은 전문의만 내릴 수 있습니다."
            ),
            limitations=(
                "본 분석은 AI 보조 스크리닝 결과이며, 의학적 진단이 아닙니다. "
                "이미지 품질, 촬영 조건에 따라 분류 정확도가 달라질 수 있습니다. "
                "최종 진단은 반드시 피부과 전문의의 직접 진찰과 조직검사를 통해 확인하세요."
            ),
        ),
    )


def create_nevus_normal_sample() -> TrainingSample:
    """시나리오 2: 양성 모반 고신뢰도 — 경과 관찰 안내."""
    return TrainingSample(
        sample_id="nv_normal_001",
        image_id="ISIC_0024500",
        classifier_output=ClassifierOutput(
            predicted_class="nv",
            probability=0.94,
            top3=[
                {"nv": 0.94}, {"bkl": 0.03}, {"mel": 0.02}
            ],
            is_malignant=False,
            grad_cam_description="병변 전체에 고른 활성화, 경계부 대칭적",
        ),
        patient_metadata=PatientMetadata(
            age=32, sex="female", body_site="왼쪽 팔",
            duration_months=60, symptoms="무증상",
            family_history="특이사항 없음",
        ),
        rag_context=(
            "[출처: 멜라닌세포모반 개요] 대부분 후천적으로 발생하며 치료가 필요 없다. "
            "변화가 관찰되면 전문의 평가가 필요하다."
        ),
        expected_response=ExpectedResponse(
            observed_features=[
                "대칭적 형태",
                "규칙적 경계",
                "균일한 활성화 패턴",
            ],
            abcde_analysis={
                "A": "대칭 — 양호",
                "B": "경계 규칙적 — 양호",
                "C": "균일한 색조 추정 — 양호",
                "D": "직경 확인 필요",
                "E": "5년간 변화 없음 — 양호",
            },
            classification_summary=(
                "Vision Classifier가 94% 신뢰도로 양성 모반(점)으로 예측했습니다. "
                "ABCDE 기준에서 우려 소견이 관찰되지 않습니다."
            ),
            evidence_sources=["멜라닌세포모반 개요"],
            recommendation=(
                "현재 특별한 조치가 필요하지 않습니다. "
                "향후 크기, 색상, 모양 변화가 관찰되면 피부과를 방문하세요. "
                "정기적인 자가 검진(월 1회)을 권장합니다."
            ),
            urgency="routine",
            patient_summary=(
                "분석 결과 일반적인 양성 점으로 보입니다. "
                "현재로서는 걱정하지 않으셔도 되지만, "
                "혹시 모양이나 크기가 변하면 피부과에서 확인받으시길 권합니다. "
                "이 결과는 AI 보조 분석이며, 정확한 진단은 전문의만 내릴 수 있습니다."
            ),
            limitations=(
                "본 분석은 AI 보조 스크리닝 결과이며, 의학적 진단이 아닙니다. "
                "양성으로 분류되었더라도 100% 확실하지 않으므로, "
                "변화가 관찰되면 반드시 전문의 상담을 받으세요."
            ),
        ),
    )


def create_low_confidence_sample() -> TrainingSample:
    """시나리오 3: 낮은 신뢰도 — 판단 유보 + 전문의 의뢰."""
    return TrainingSample(
        sample_id="uncertain_001",
        image_id="ISIC_0024700",
        classifier_output=ClassifierOutput(
            predicted_class="bkl",
            probability=0.45,
            top3=[
                {"bkl": 0.45}, {"mel": 0.30}, {"akiec": 0.15}
            ],
            is_malignant=False,
            grad_cam_description="활성화가 분산되어 있으며 명확한 패턴 부재",
        ),
        patient_metadata=PatientMetadata(
            age=67, sex="male", body_site="얼굴 좌측",
            duration_months=4, symptoms="간헐적 가려움, 표면 거칠어짐",
            family_history="특이사항 없음",
        ),
        rag_context=(
            "[출처: 양성 각화증 개요] 멜라노마와의 감별진단이 중요하다.\n"
            "[출처: 광선각화증 개요] 만성 자외선 노출에 의한 전암성 병변이다. "
            "일광 노출 부위인 얼굴에 호발한다.\n"
            "[출처: 전문의 의뢰 기준] 50세 이상에서 새로 발생한 병변은 의뢰 권고."
        ),
        expected_response=ExpectedResponse(
            observed_features=[
                "분류 신뢰도 낮음 (45%)",
                "멜라노마 가능성 30% — 배제 불가",
                "광선각화증 가능성 15% — 부위(얼굴)와 일치",
            ],
            abcde_analysis={
                "A": "판단 유보 — 분류기 확신도 부족",
                "B": "판단 유보",
                "C": "판단 유보",
                "D": "직경 확인 필요",
                "E": "4개월 전 발생, 표면 질감 변화 보고 — 주의 필요",
            },
            classification_summary=(
                "Vision Classifier의 신뢰도가 45%로 기준(70%) 미만입니다. "
                "양성 각화증이 가장 높은 확률이나, 멜라노마(30%)와 "
                "광선각화증(15%)을 배제할 수 없습니다. "
                "AI 분석만으로는 정확한 판단이 어려운 사례입니다."
            ),
            evidence_sources=[
                "양성 각화증 개요",
                "광선각화증 개요",
                "전문의 의뢰 기준",
            ],
            recommendation=(
                "분류 신뢰도가 낮고 악성 가능성을 배제할 수 없어, "
                "피부과 전문의 상담을 권고합니다 (4주 이내). "
                "더모스코피 검사로 정확한 감별진단이 가능합니다."
            ),
            urgency="soon",
            patient_summary=(
                "분석 결과가 한 가지 결론으로 모이지 않아 "
                "AI만으로는 정확한 판단이 어렵습니다. "
                "피부과에서 확대경(더모스코피) 검사를 받으시면 "
                "정확한 결과를 확인할 수 있습니다. "
                "가까운 시일 내(4주 이내) 방문을 권합니다."
            ),
            limitations=(
                "본 사례는 AI 분류기의 신뢰도가 낮아 분석 결과의 "
                "신뢰성이 제한적입니다. 반드시 전문의 진찰을 받으세요."
            ),
        ),
    )


# ============================================================
# 5. Conversational 포맷 변환 (SFTTrainer용)
# ============================================================
def sample_to_conversation(sample: TrainingSample) -> List[Dict[str, str]]:
    """
    TrainingSample → HuggingFace SFTTrainer 대화 형식 변환.
    [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}]
    """
    clf = sample.classifier_output
    meta = sample.patient_metadata

    top3_str = ", ".join(
        [f"{list(d.keys())[0]}: {list(d.values())[0]:.1%}" for d in clf.top3]
    )
    malignant_str = "악성 의심" if clf.is_malignant else "양성 추정"

    user_content = USER_PROMPT_TEMPLATE.format(
        predicted_class=clf.predicted_class,
        probability=clf.probability,
        top3_str=top3_str,
        malignant_str=malignant_str,
        grad_cam_description=clf.grad_cam_description,
        age=meta.age or "미상",
        sex=meta.sex or "미상",
        body_site=meta.body_site or "미상",
        duration_months=meta.duration_months or "미상",
        symptoms=meta.symptoms or "특이사항 없음",
        family_history=meta.family_history or "특이사항 없음",
        rag_context=sample.rag_context,
    )

    assistant_content = json.dumps(
        asdict(sample.expected_response), ensure_ascii=False, indent=2
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ============================================================
# 6. Function Calling 스키마 정의
# ============================================================
FUNCTION_SCHEMAS = [
    {
        "name": "lookup_abcde_criteria",
        "description": "ABCDE 규칙 기반 병변 평가 기준을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D", "E"],
                    "description": "평가할 ABCDE 항목",
                }
            },
            "required": ["feature"],
        },
    },
    {
        "name": "find_similar_cases",
        "description": "유사 병변 사례를 RAG DB에서 검색합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_class": {"type": "string", "description": "예측된 병변 클래스"},
                "confidence": {"type": "number", "description": "분류 신뢰도"},
            },
            "required": ["predicted_class"],
        },
    },
    {
        "name": "recommend_next_step",
        "description": "심각도에 따른 다음 단계를 추천합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["low", "moderate", "high"],
                    "description": "평가된 심각도",
                },
                "is_malignant": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["severity", "is_malignant", "confidence"],
        },
    },
]


# ============================================================
# 7. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print(" [1주차] Gemma LoRA 학습 데이터 설계")
    print("=" * 60)

    output_dir = OUTPUT_DIR / "gemma_data_design"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 시드 샘플 생성 ---
    samples = [
        create_melanoma_high_confidence_sample(),  # 악성 고신뢰
        create_nevus_normal_sample(),               # 양성 고신뢰
        create_low_confidence_sample(),             # 낮은 신뢰
    ]

    # --- 대화 형식 변환 ---
    conversations = []
    for sample in samples:
        conv = sample_to_conversation(sample)
        conversations.append({
            "sample_id": sample.sample_id,
            "image_id": sample.image_id,
            "messages": conv,
        })

    # 저장: SFTTrainer 입력 형식 (JSONL)
    jsonl_path = output_dir / "seed_training_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"\n[시드 데이터] {len(conversations)}건 → {jsonl_path}")

    # --- Function Calling 스키마 저장 ---
    fc_path = output_dir / "function_calling_schemas.json"
    with open(fc_path, "w", encoding="utf-8") as f:
        json.dump(FUNCTION_SCHEMAS, ensure_ascii=False, indent=2, fp=f)
    print(f"[Function Calling] 스키마 → {fc_path}")

    # --- 시스템 프롬프트 저장 ---
    prompt_path = output_dir / "system_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT)
    print(f"[시스템 프롬프트] → {prompt_path}")

    # --- 데이터 생성 계획서 ---
    plan = {
        "1주차_완료": {
            "시드_샘플": f"{len(samples)}건 (시나리오별 대표 샘플)",
            "데이터_스키마": "TrainingSample dataclass 확정",
            "시스템_프롬프트": "확정",
            "Function_Calling_스키마": f"{len(FUNCTION_SCHEMAS)}개 함수 정의",
        },
        "2주차_계획": {
            "목표_샘플_수": "3,000~5,000건",
            "생성_방법": [
                "HAM10000 메타데이터 기반 자동 생성 (Vision Classifier 출력 시뮬레이션)",
                "LLM(Claude/GPT-4) API로 응답 생성 → 수동 검수",
                "시나리오 다양화: 연령대, 부위, 신뢰도 구간, 복합 소견",
            ],
            "품질_관리": [
                "의학 용어 정확성 검증 (용어집 매칭)",
                "안전성 검증 (악성 → 반드시 의뢰 권고 포함 여부)",
                "JSON 스키마 유효성 자동 검사",
            ],
        },
        "시나리오_분포_계획": {
            "악성_고신뢰 (mel/bcc/akiec, conf≥0.7)": "30%",
            "양성_고신뢰 (nv/bkl/df/vasc, conf≥0.7)": "30%",
            "낮은_신뢰도 (conf<0.7)": "25%",
            "경계_케이스 (악성확률 0.3~0.5)": "15%",
        },
    }

    plan_path = output_dir / "data_generation_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"[생성 계획] → {plan_path}")

    # --- 샘플 미리보기 ---
    print("\n" + "=" * 50)
    print(" 시드 샘플 미리보기 (1건)")
    print("=" * 50)
    preview = conversations[0]["messages"]
    for msg in preview:
        role = msg["role"].upper()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        print(f"\n[{role}]\n{content}")

    print("\n" + "=" * 60)
    print(" Gemma 학습 데이터 설계 완료")
    print(f" 출력 디렉터리: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
