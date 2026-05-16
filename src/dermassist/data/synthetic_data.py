"""
07_gemma_training_data_gen_en.py
==================================
[Gemma 트랙 - 영어 LMIC 버전] Gemma SFT 학습 데이터 생성

사하라 이남 아프리카 + 기타 LMIC 농촌 의료 환경 대상으로
영어 학습 데이터 5,000건을 룰 기반으로 생성합니다.

기존 한국어 버전(07_gemma_training_data_gen.py)과의 차이:
  1. 시나리오 풀 영어화 + LMIC 컨텍스트 강화
  2. 알비노/HIV+/야외노동자 등 LMIC 특이 위험군 추가
  3. 자원 제약 (specialist 거리, 시설 부족) 반영 권고
  4. 환자 친화적 영어 표현 (community health worker 대상)
  5. 5,000건으로 확장 (다양성 강화)
  6. 응답 톤 3종 (concise/detailed/patient-friendly)

실행: python 07_gemma_training_data_gen_en.py

산출물:
  outputs/gemma_training_data_en/training_data.jsonl
  outputs/gemma_training_data_en/scenario_distribution.json
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    OUTPUT_DIR, SPLIT_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
    CONFIDENCE_THRESHOLD,
)


# ============================================================
# 1. LMIC System Prompt
# ============================================================
SYSTEM_PROMPT_EN = """You are a skin lesion screening assistant designed for community health workers in low-resource settings, particularly rural areas of sub-Saharan Africa and other LMICs (Low- and Middle-Income Countries) where dermatologist density is below 1 per million population.

Your role:
1. Provide structured triage recommendations to support frontline health workers
2. Never replace specialist diagnosis - only assist with case prioritization
3. Account for limited follow-up infrastructure (long travel distances, sparse specialists, limited biopsy access) when recommending urgency
4. Suggest teledermatology referral when in-person specialist access is impractical
5. Use plain English suitable for translation by health workers

Critical safety rules:
- This system runs entirely offline; no patient data is transmitted
- Always include limitations and AI disclaimer
- For high-confidence malignant cases (>=70%), recommend urgent specialist contact
- For low-confidence cases, default to safer recommendation (specialist review)
- Account for the patient's likely access barriers (distance, cost, follow-up reliability)

Input you will receive:
- Vision Classifier output (predicted class, confidence, Grad-CAM description)
- Patient metadata (age, sex, body site, duration, symptoms, risk factors)
- RAG-retrieved medical context

Output format: Strictly valid JSON with the following fields:
{
  "observed_features": [list of clinical observations],
  "abcde_analysis": {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...},
  "classification_summary": "summary of vision findings",
  "evidence_sources": [list of evidence references],
  "recommendation": "clinical recommendation accounting for resource constraints",
  "urgency": "routine" | "soon" | "urgent",
  "patient_summary": "plain-English summary for patient",
  "limitations": "AI disclaimers and limitations"
}"""


# ============================================================
# 2. LMIC 환자 프로파일
# ============================================================
PATIENT_PROFILES = [
    {
        "name": "rural_farmer",
        "context": "outdoor agricultural worker with chronic UV exposure",
        "occupations": ["farmer", "subsistence farmer", "field worker"],
        "ages": (35, 75),
        "skin_types": ["Fitzpatrick IV", "Fitzpatrick V"],
        "risk_factors": ["chronic sun exposure", "outdoor occupation"],
    },
    {
        "name": "albinism",
        "context": "patient with albinism, extremely high UV sensitivity",
        "occupations": ["albinism patient", "OCA1 patient", "OCA2 patient"],
        "ages": (15, 50),
        "skin_types": ["Fitzpatrick I (albinism)"],
        "risk_factors": [
            "OCA (oculocutaneous albinism)",
            "absent melanin protection",
            "multiple actinic damage sites",
        ],
    },
    {
        "name": "outdoor_laborer",
        "context": "manual laborer with prolonged sun exposure",
        "occupations": ["construction worker", "fisherman", "miner"],
        "ages": (25, 65),
        "skin_types": ["Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V"],
        "risk_factors": ["chronic UV exposure", "occupational sun exposure"],
    },
    {
        "name": "hiv_positive",
        "context": "immunocompromised patient with elevated skin cancer risk",
        "occupations": ["patient on ART", "HIV positive patient"],
        "ages": (25, 60),
        "skin_types": ["Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V"],
        "risk_factors": ["HIV infection", "immunocompromised status"],
    },
    {
        "name": "elderly_rural",
        "context": "elderly patient in rural setting with cumulative sun damage",
        "occupations": ["retired farmer", "rural elderly"],
        "ages": (60, 90),
        "skin_types": ["Fitzpatrick III", "Fitzpatrick IV"],
        "risk_factors": ["cumulative lifetime UV exposure", "age-related skin changes"],
    },
    {
        "name": "remote_area",
        "context": "patient in remote area with no specialist access within 200km",
        "occupations": ["village resident", "remote community member"],
        "ages": (20, 70),
        "skin_types": ["Fitzpatrick IV", "Fitzpatrick V"],
        "risk_factors": ["limited healthcare access", "no nearby dermatologist"],
    },
    {
        "name": "child_with_albinism",
        "context": "pediatric albinism patient with early sun damage",
        "occupations": ["child with OCA", "adolescent with albinism"],
        "ages": (5, 18),
        "skin_types": ["Fitzpatrick I (albinism)"],
        "risk_factors": [
            "OCA",
            "early-onset actinic damage",
            "limited sun protection access",
        ],
    },
    {
        "name": "general_lmic",
        "context": "general adult patient in LMIC primary care setting",
        "occupations": ["adult patient", "primary care patient"],
        "ages": (20, 70),
        "skin_types": ["Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V"],
        "risk_factors": ["limited healthcare access"],
    },
]

# 신체 부위 (LMIC에서 흔한 부위 강조)
BODY_SITES_EN = [
    "scalp",
    "face",
    "forehead",
    "nose",
    "ear",
    "lip",
    "neck",
    "chest",
    "back",
    "abdomen",
    "left forearm (sun-exposed)",
    "right forearm (sun-exposed)",
    "left lower leg",
    "right lower leg",
    "hand dorsum",
    "foot sole",
]

# 증상 표현 변형
SYMPTOMS_EN = [
    "asymptomatic",
    "intermittent itching",
    "occasional bleeding when scratched",
    "rapid growth over past 3 months",
    "color change reported by patient",
    "tender to palpation",
    "non-healing ulceration",
    "scaly, rough surface",
    "raised firm nodule",
    "pigmentation darkening recently",
    "size doubled in 6 months",
    "no notable changes",
]

# 자원 제약 컨텍스트 (LMIC 특화)
RESOURCE_CONSTRAINTS_EN = [
    "Patient travel to nearest dermatologist requires 200+ km journey",
    "No biopsy facilities in this primary care setting",
    "Specialist referral wait time typically 4-8 weeks",
    "Patient has limited financial means for travel",
    "Follow-up scheduling unreliable due to distance",
    "Teledermatology service available via African Teledermatology Project",
]

# 응답 톤 변형
RESPONSE_TONES = ["concise", "detailed", "patient_friendly"]


# ============================================================
# 3. Grad-CAM 서술 템플릿 (클래스별, 5개씩)
# ============================================================
GRAD_CAM_TEMPLATES_EN = {
    "mel": [
        "Asymmetric activation pattern with strong response on irregular border regions",
        "Heterogeneous activation distribution suggesting multiple color zones",
        "Focal high-activation areas at lesion margins, characteristic of irregular pigment network",
        "Asymmetric center-to-edge gradient with hot spots at peripheral irregular areas",
        "Strong activation in areas suggestive of regression or color variation",
    ],
    "bcc": [
        "Activation focused at lesion edges with translucent border characteristics",
        "Central depression activation with peripheral nodular pattern",
        "Focal activation at telangiectatic vessel locations",
        "Pearl-like nodular activation pattern at lesion center",
        "Activation pattern consistent with rolled border morphology",
    ],
    "akiec": [
        "Diffuse activation across rough, scaly surface area",
        "Moderate activation pattern indicating keratotic surface texture",
        "Activation distributed across erythematous patch with scale",
        "Surface-prominent activation suggesting actinic damage",
        "Activation pattern at sun-exposed area with scale formation",
    ],
    "nv": [
        "Symmetric central activation indicating uniform pigmented lesion",
        "Even activation distribution following regular pigment network",
        "Centered round activation with smooth border response",
        "Homogeneous activation pattern across the lesion",
        "Symmetrical activation consistent with benign melanocytic pattern",
    ],
    "bkl": [
        "Surface-textured activation with stuck-on appearance pattern",
        "Activation pattern showing milia-like cysts and comedo-like openings",
        "Diffuse activation across scaly hyperkeratotic surface",
        "Activation following waxy, verrucous surface contours",
        "Pattern consistent with seborrheic keratosis topography",
    ],
    "df": [
        "Central depression activation with peripheral firm-papule response",
        "Activation pattern showing characteristic central white scar-like area",
        "Focused activation at firm dermal nodule location",
        "Pattern consistent with dermatofibroma dimple sign location",
        "Activation distribution indicating fibrous papule structure",
    ],
    "vasc": [
        "Strong activation at vascular lacunae (red-blue zones)",
        "Activation pattern indicating cherry angioma vessel structure",
        "Focal activation at vascular spaces typical of angiokeratoma",
        "Activation distribution showing red-purple papule structure",
        "Pattern consistent with vascular lesion morphology",
    ],
}


# ============================================================
# 4. 시나리오 데이터 클래스
# ============================================================
@dataclass
class Scenario:
    scenario_type: str
    true_class: str
    predicted_class: str
    probability: float
    is_malignant: bool
    grad_cam_desc: str
    patient_profile: Dict
    age: int
    sex: str
    body_site: str
    duration_months: int
    symptom: str
    risk_factor: str
    resource_constraint: str
    response_tone: str


# ============================================================
# 5. 시나리오 생성 로직
# ============================================================
def sample_patient_profile(target_class: str) -> Dict:
    """클래스에 맞는 환자 프로파일 샘플링."""
    # 알비노 특화 클래스에는 알비노 프로파일 우선
    if target_class in ["mel", "bcc", "akiec"] and random.random() < 0.25:
        # 25% 확률로 알비노 환자 (LMIC 특이 위험군)
        candidates = [p for p in PATIENT_PROFILES if "albinism" in p["name"]]
    elif target_class == "nv" and random.random() < 0.4:
        # 양성 모반은 일반 LMIC 환자가 더 흔함
        candidates = [p for p in PATIENT_PROFILES if p["name"] in ["general_lmic", "rural_farmer"]]
    else:
        candidates = PATIENT_PROFILES
    return random.choice(candidates)


def generate_scenario(
    true_class: str,
    scenario_type: str,
) -> Scenario:
    """단일 시나리오 생성."""
    is_malignant = true_class in MALIGNANT_CLASSES

    # 신뢰도 결정
    if scenario_type == "high_conf_malignant":
        predicted_class = true_class
        probability = random.uniform(0.85, 0.99)
    elif scenario_type == "high_conf_benign":
        predicted_class = true_class
        probability = random.uniform(0.85, 0.99)
    elif scenario_type == "borderline":
        predicted_class = true_class
        probability = random.uniform(0.70, 0.85)
    elif scenario_type == "low_conf":
        # 낮은 신뢰도: 50% 확률로 잘못된 예측
        if random.random() < 0.5:
            predicted_class = random.choice([c for c in CLASS_NAMES if c != true_class])
        else:
            predicted_class = true_class
        probability = random.uniform(0.50, 0.70)
    elif scenario_type == "albinism_specific":
        predicted_class = true_class
        probability = random.uniform(0.70, 0.95)
    else:
        predicted_class = true_class
        probability = random.uniform(0.50, 0.99)

    # 환자 프로파일
    profile = sample_patient_profile(predicted_class)
    age = random.randint(*profile["ages"])
    sex = random.choice(["male", "female"])
    body_site = random.choice(BODY_SITES_EN)
    duration_months = random.choice([1, 2, 3, 6, 12, 24, 36, 60])
    symptom = random.choice(SYMPTOMS_EN)
    risk_factor = random.choice(profile["risk_factors"])
    resource_constraint = random.choice(RESOURCE_CONSTRAINTS_EN)
    response_tone = random.choice(RESPONSE_TONES)

    grad_cam_desc = random.choice(GRAD_CAM_TEMPLATES_EN[predicted_class])

    return Scenario(
        scenario_type=scenario_type,
        true_class=true_class,
        predicted_class=predicted_class,
        probability=probability,
        is_malignant=predicted_class in MALIGNANT_CLASSES,
        grad_cam_desc=grad_cam_desc,
        patient_profile=profile,
        age=age,
        sex=sex,
        body_site=body_site,
        duration_months=duration_months,
        symptom=symptom,
        risk_factor=risk_factor,
        resource_constraint=resource_constraint,
        response_tone=response_tone,
    )


# ============================================================
# 6. User Prompt 빌드
# ============================================================
def build_user_prompt(s: Scenario, rag_context: str = "") -> str:
    """User 메시지 (Vision 결과 + 환자 정보 + RAG)."""
    parts = []

    parts.append("## Vision Classifier Output\n")
    parts.append(f"- Predicted class: {s.predicted_class}")
    parts.append(f"- Confidence: {s.probability:.1%}")
    parts.append(f"- Is malignant class: {s.is_malignant}")
    parts.append(f"- Grad-CAM description: {s.grad_cam_desc}")
    parts.append("")

    parts.append("## Patient Information\n")
    parts.append(f"- Age: {s.age}")
    parts.append(f"- Sex: {s.sex}")
    parts.append(f"- Body site: {s.body_site}")
    parts.append(f"- Lesion duration: {s.duration_months} months")
    parts.append(f"- Symptoms: {s.symptom}")
    parts.append(f"- Patient context: {s.patient_profile['context']}")
    parts.append(f"- Risk factor: {s.risk_factor}")
    parts.append(f"- Skin type: {random.choice(s.patient_profile['skin_types'])}")
    parts.append("")

    parts.append("## Healthcare Setting\n")
    parts.append(f"- Resource constraint: {s.resource_constraint}")
    parts.append("")

    if rag_context:
        parts.append("## Relevant Medical References\n")
        parts.append(rag_context)
        parts.append("")

    parts.append(f"Please provide a structured assessment in valid JSON format. "
                 f"Response style: {s.response_tone}.")

    return "\n".join(parts)


# ============================================================
# 7. Assistant 응답 생성 (룰 기반, 영어)
# ============================================================
def generate_response(s: Scenario) -> Dict:
    """클래스 + 신뢰도 + 톤에 따른 룰 기반 응답."""

    # === observed_features ===
    features = []
    features.append(f"Vision Classifier confidence: {s.probability:.1%}")
    features.append(f"Grad-CAM: {s.grad_cam_desc}")

    if s.is_malignant:
        features.append(f"Lesion classified as malignant category ({s.predicted_class})")
    else:
        features.append(f"Lesion classified as benign category ({s.predicted_class})")

    if "albinism" in s.patient_profile["name"]:
        features.append("Patient has albinism - elevated skin cancer risk profile")
    if "hiv" in s.patient_profile["name"]:
        features.append("Patient is immunocompromised - increased malignancy risk")

    # === abcde_analysis ===
    if s.probability >= CONFIDENCE_THRESHOLD:
        if s.is_malignant:
            abcde = {
                "A": "Asymmetry suggested by activation pattern",
                "B": "Border irregularity indicated",
                "C": "Color variation likely - direct inspection needed",
                "D": "Diameter requires direct measurement",
                "E": f"Evolution: {s.symptom} over {s.duration_months} months",
            }
        else:
            abcde = {
                "A": "Symmetry - favorable",
                "B": "Regular border indicated",
                "C": "Uniform color suggested",
                "D": "Diameter to be confirmed by health worker",
                "E": f"Evolution: {s.symptom} over {s.duration_months} months",
            }
    else:
        abcde = {
            "A": "Inconclusive - direct examination required",
            "B": "Inconclusive - direct examination required",
            "C": "Inconclusive - direct examination required",
            "D": "Direct measurement needed",
            "E": f"Evolution noted: {s.symptom}",
        }

    # === classification_summary ===
    if s.probability >= CONFIDENCE_THRESHOLD:
        if s.is_malignant:
            cls_summary = (
                f"Vision Classifier predicts {s.predicted_class} with "
                f"{s.probability:.1%} confidence, suggesting potential malignancy. "
                f"Specialist confirmation strongly recommended."
            )
        else:
            cls_summary = (
                f"Vision Classifier predicts benign category {s.predicted_class} "
                f"with {s.probability:.1%} confidence. "
                f"No immediate concerning features identified."
            )
    else:
        cls_summary = (
            f"Vision Classifier confidence is below threshold "
            f"({s.probability:.1%} < 70%). "
            f"Predicted class ({s.predicted_class}) requires specialist confirmation."
        )

    # === evidence_sources ===
    evidence = [
        f"Grad-CAM analysis: {s.grad_cam_desc[:80]}...",
        "ABCDE self-examination criteria (American Academy of Dermatology)",
        "Specialist referral guidelines for resource-limited settings",
    ]
    if "albinism" in s.patient_profile["name"]:
        evidence.append(
            "Fondation Pierre Fabre - Albinism skin cancer prevention guidelines"
        )
    if s.predicted_class in MALIGNANT_CLASSES:
        evidence.append("WHO cancer early detection protocols")

    # === recommendation (LMIC 자원 제약 반영) ===
    if s.is_malignant and s.probability >= CONFIDENCE_THRESHOLD:
        urgency = "urgent"
        rec = (
            f"Urgent referral to nearest dermatology service. "
            f"Given the {s.resource_constraint.lower()}, "
            f"consider teledermatology consultation via African Teledermatology Project "
            f"if in-person specialist access is impractical within 2 weeks. "
            f"Document with photograph and patient history. "
            f"If patient cannot reach specialist, prioritize phone consultation "
            f"with regional referral center."
        )
    elif s.is_malignant and s.probability < CONFIDENCE_THRESHOLD:
        urgency = "soon"
        rec = (
            f"Schedule dermatology consultation within 4 weeks. "
            f"Despite lower confidence, malignant categorization warrants caution. "
            f"Teledermatology referral recommended given {s.resource_constraint.lower()}. "
            f"Monitor for changes weekly."
        )
    elif not s.is_malignant and s.probability >= CONFIDENCE_THRESHOLD:
        urgency = "routine"
        rec = (
            f"Routine follow-up at 6-month intervals. "
            f"Educate patient on self-monitoring using ABCDE rule. "
            f"Counsel on sun protection if available. "
            f"For albinism patients: emphasize comprehensive sun protection strategy. "
            f"Return for evaluation if any changes occur."
        )
    else:
        urgency = "soon"
        rec = (
            f"Specialist review recommended due to inconclusive AI analysis. "
            f"Consider teledermatology consultation given {s.resource_constraint.lower()}. "
            f"Schedule follow-up within 6-8 weeks. "
            f"Provide patient with self-monitoring guidance in the interim."
        )

    # === patient_summary (영어, 평이한 표현) ===
    if urgency == "urgent":
        patient_summary = (
            "This skin spot needs urgent doctor check. "
            "Please go to nearest health center as soon as possible, "
            "ideally within 2 weeks. Bring this report. "
            "If you cannot travel, ask the health worker about phone consultation. "
            "This is AI screening only - doctor must confirm diagnosis."
        )
    elif urgency == "soon":
        patient_summary = (
            "This skin spot should be checked by a specialist within a few weeks. "
            "It may not be serious, but a doctor's examination is recommended to be sure. "
            "If travel is difficult, ask the health worker about teledermatology services. "
            "This is AI screening only - not a diagnosis."
        )
    else:
        patient_summary = (
            "This skin spot looks like it is likely not dangerous. "
            "Watch it carefully. Tell the health worker if it: "
            "gets bigger, changes color, bleeds, itches, or develops new symptoms. "
            "Apply sunscreen if available, especially if you have albinism. "
            "Schedule routine check in 6 months. "
            "This is AI screening - see doctor if you are worried."
        )

    # 톤별 조정
    if s.response_tone == "concise":
        # patient_summary 짧게
        patient_summary = patient_summary.split(". ")[0] + ". " + \
                          patient_summary.split(". ")[-1]
    elif s.response_tone == "detailed":
        # 더 상세하게
        patient_summary += (
            f" Your specific situation ({s.patient_profile['context']}) "
            f"warrants extra attention to skin changes."
        )

    # === limitations (LMIC 특화 안전 고지) ===
    limitations = (
        "This is an AI-assisted screening tool, not a medical diagnosis. "
        "The system relies on Vision Classifier output and limited patient information; "
        "physical examination by a qualified dermatologist remains essential. "
        "In low-resource settings, this tool is designed to assist prioritization, "
        "not replace specialist care. "
        "Image quality, lighting conditions, and skin type variations may affect accuracy. "
        "All malignant or uncertain findings should be confirmed by a specialist "
        "via direct examination or teledermatology consultation."
    )

    return {
        "observed_features": features,
        "abcde_analysis": abcde,
        "classification_summary": cls_summary,
        "evidence_sources": evidence,
        "recommendation": rec,
        "urgency": urgency,
        "patient_summary": patient_summary,
        "limitations": limitations,
    }


# ============================================================
# 8. 학습 샘플 빌드
# ============================================================
def build_training_sample(scenario: Scenario, rag_context: str = "") -> Dict:
    """전체 messages 형식의 학습 샘플."""
    user_prompt = build_user_prompt(scenario, rag_context)
    response = generate_response(scenario)

    return {
        "scenario_type": scenario.scenario_type,
        "true_class": scenario.true_class,
        "predicted_class": scenario.predicted_class,
        "probability": round(scenario.probability, 4),
        "patient_profile": scenario.patient_profile["name"],
        "response_tone": scenario.response_tone,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(response, indent=2)},
        ],
    }


# ============================================================
# 9. 분포 설계
# ============================================================
def design_distribution(total: int = 5000) -> List[Dict]:
    """5,000건의 시나리오 분포 설계."""
    distribution = {
        "high_conf_malignant": int(total * 0.30),     # 1,500
        "high_conf_benign": int(total * 0.30),        # 1,500
        "borderline": int(total * 0.15),              # 750
        "low_conf": int(total * 0.20),                # 1,000
        "albinism_specific": int(total * 0.05),       # 250
    }

    # 클래스 풀 정의
    class_pools = {
        "high_conf_malignant": MALIGNANT_CLASSES,    # mel, bcc, akiec
        "high_conf_benign": BENIGN_CLASSES,           # nv, bkl, df, vasc
        "borderline": CLASS_NAMES,
        "low_conf": CLASS_NAMES,
        "albinism_specific": ["mel", "bcc", "akiec"],  # 알비노는 피부암 위험
    }

    plan = []
    for scenario_type, count in distribution.items():
        pool = class_pools[scenario_type]
        for _ in range(count):
            true_class = random.choice(pool)
            plan.append({"scenario_type": scenario_type, "true_class": true_class})

    random.shuffle(plan)
    return plan


# ============================================================
# 10. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print(" [3주차] Gemma 학습 데이터 생성 — 영어 LMIC 버전")
    print("=" * 60)

    random.seed(42)

    # --- 분포 설계 ---
    total_samples = 5000
    plan = design_distribution(total=total_samples)

    type_dist = {}
    for p in plan:
        type_dist[p["scenario_type"]] = type_dist.get(p["scenario_type"], 0) + 1

    print(f"\n[시나리오 분포 — 총 {total_samples}건]")
    for stype, count in type_dist.items():
        print(f"  {stype:<22}: {count:>4}건 ({count/total_samples*100:.1f}%)")

    # --- 클래스별 카운트 ---
    class_count = {c: 0 for c in CLASS_NAMES}
    for p in plan:
        class_count[p["true_class"]] += 1

    print(f"\n[클래스 분포]")
    for cls, count in class_count.items():
        marker = " (악성)" if cls in MALIGNANT_CLASSES else ""
        print(f"  {cls}: {count}건{marker}")

    # --- 데이터 생성 ---
    output_dir = OUTPUT_DIR / "gemma_training_data_en"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.jsonl"

    samples = []
    profile_dist = {}
    tone_dist = {}

    print(f"\n[생성 중]")
    for spec in tqdm(plan, desc="샘플 생성"):
        scenario = generate_scenario(spec["true_class"], spec["scenario_type"])
        sample = build_training_sample(scenario)
        samples.append(sample)

        profile_dist[scenario.patient_profile["name"]] = \
            profile_dist.get(scenario.patient_profile["name"], 0) + 1
        tone_dist[scenario.response_tone] = \
            tone_dist.get(scenario.response_tone, 0) + 1

    # --- JSONL 저장 ---
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\n[저장] {output_path}")
    print(f"  파일 크기: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # --- 분포 통계 저장 ---
    stats = {
        "total_samples": len(samples),
        "scenario_distribution": type_dist,
        "class_distribution": class_count,
        "patient_profile_distribution": profile_dist,
        "response_tone_distribution": tone_dist,
        "language": "english",
        "target": "sub-Saharan Africa + LMIC general patients",
        "system_prompt_length": len(SYSTEM_PROMPT_EN),
    }
    stats_path = output_dir / "scenario_distribution.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[통계] {stats_path}")

    # --- 환자 프로파일 분포 출력 ---
    print(f"\n[환자 프로파일 분포]")
    for prof, count in sorted(profile_dist.items(), key=lambda x: -x[1]):
        print(f"  {prof:<25}: {count:>4}건 ({count/total_samples*100:.1f}%)")

    print(f"\n[응답 톤 분포]")
    for tone, count in tone_dist.items():
        print(f"  {tone:<20}: {count:>4}건 ({count/total_samples*100:.1f}%)")

    # --- 샘플 예시 출력 ---
    print(f"\n" + "=" * 60)
    print(" 샘플 예시")
    print("=" * 60)
    sample = samples[0]
    print(f"\nScenario: {sample['scenario_type']} / "
          f"True: {sample['true_class']} / "
          f"Predicted: {sample['predicted_class']} / "
          f"Profile: {sample['patient_profile']}")
    print(f"\n[User Prompt 일부]")
    user_msg = sample["messages"][1]["content"]
    print(user_msg[:600] + "...")
    print(f"\n[Assistant Response 일부]")
    asst_msg = sample["messages"][2]["content"]
    print(asst_msg[:500] + "...")

    print(f"\n" + "=" * 60)
    print(" 생성 완료")
    print(f"  다음 단계: python 09_gemma_lora_finetune_en.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
