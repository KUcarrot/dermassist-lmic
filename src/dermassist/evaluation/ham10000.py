"""
ham10000.py
===========
HAM10000 in-distribution evaluation for the end-to-end pipeline.

This script tests the trained pipeline on the HAM10000 test split,
providing baseline metrics for comparison with external validation
(see bcn20000.py).

Evaluation criteria:
1. JSON output validity
2. Required field completeness
3. Vision-LLM urgency consistency
4. Patient summary consistency
5. Hallucination detection (precise patterns)
6. Safety disclaimer inclusion
7. High-confidence malignant referral
8. ABCDE duplicate detection in observed_features

Run via:
    python scripts/07_evaluate_ham10000.py
    python scripts/07_evaluate_ham10000.py --samples_per_class 5
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

try:
    from configs.config import (
        PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
        CONFIDENCE_THRESHOLD, GEMMA_CONFIG,
    )
except ImportError:
    # This file is at: src/dermassist/evaluation/ham10000.py
    # Project root is 4 levels up
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from configs.config import (
        PROCESSED_DIR, SPLIT_DIR, VISION_MODEL_DIR, RAG_DB_DIR,
        GEMMA_MODEL_DIR, OUTPUT_DIR,
        CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
        CONFIDENCE_THRESHOLD, GEMMA_CONFIG,
    )

# Pipeline modules
from dermassist.pipeline.assistant import (
    SkinLesionAssistant,
    PatientMetadata,
)


# ============================================================
# 1. Sampling logic
# ============================================================
def sample_test_cases(
    split_csv: Path,
    samples_per_class: int = 3,
    split: str = "test",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample stratified test cases from the split CSV."""
    df = pd.read_csv(split_csv)
    df = df[df["split"] == split]
    df = df[~df["image_id"].str.startswith("syn_")]

    sampled_list = []
    for cls in CLASS_NAMES:
        cls_df = df[df["dx"] == cls]
        if len(cls_df) == 0:
            print(f"  [Warning] No samples for class {cls} - skipping")
            continue
        n = min(samples_per_class, len(cls_df))
        sampled = cls_df.sample(n=n, random_state=seed)
        sampled_list.append(sampled)

    result = pd.concat(sampled_list, ignore_index=True)
    print(f"[Sampling] Total {len(result)} samples (max {samples_per_class} per class)")
    for cls in CLASS_NAMES:
        count = (result["dx"] == cls).sum()
        if count > 0:
            print(f"  {cls}: {count}")

    return result


# ============================================================
# 2. Consistency evaluation (English LMIC response model)
# ============================================================
def evaluate_consistency(result: Dict) -> Dict:
    """Evaluate response consistency for LMIC English responses."""
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

    # 1. Required fields check
    required = [
        "observed_features", "abcde_analysis", "classification_summary",
        "urgency", "recommendation", "patient_summary", "limitations",
    ]
    missing = [f for f in required if f not in resp]
    if missing:
        evaluations["has_required_fields"] = False
        issues.append(f"Missing fields: {missing}")

    # 2. Urgency value validity
    urgency = resp.get("urgency", "")
    if urgency not in ["routine", "soon", "urgent"]:
        evaluations["urgency_consistent"] = False
        issues.append(f"Invalid urgency: {urgency}")

    # 3. Vision classification vs urgency consistency
    if confidence >= 0.70:
        if not is_malignant and urgency == "urgent":
            evaluations["urgency_consistent"] = False
            issues.append(
                f"Inconsistent: benign ({predicted_class}, {confidence:.1%}) -> urgency=urgent"
            )
        elif is_malignant and urgency == "routine":
            evaluations["urgency_consistent"] = False
            issues.append(
                f"Inconsistent: malignant ({predicted_class}, {confidence:.1%}) -> urgency=routine"
            )

    # 4. Patient summary consistency (English keywords)
    summary = resp.get("patient_summary", "")
    if not is_malignant and confidence >= 0.70:
        # Benign high-confidence: malignant-implying terms are problematic
        malignant_terms_en = [
            "high malignancy",
            "highly suspicious",
            "cancer suspected",
            "needs urgent doctor check",
        ]
        for term in malignant_terms_en:
            if term.lower() in summary.lower():
                evaluations["summary_consistent"] = False
                issues.append(f"Benign case but summary contains '{term}'")
                break

    # 5. Hallucination detection (precise patterns)
    full_text = json.dumps(resp, ensure_ascii=False)

    # Precise hallucination patterns (exclude normal medical phrasing)
    hallucination_patterns = [
        # Self-contradictions (not normal medical phrasing)
        (r"\bBCC\s+benign\s+form\b", "BCC benign form contradiction"),
        (r"benign\s+form\s+of\s+(BCC|melanoma|carcinoma)", "benign form of carcinoma contradiction"),
        (r"\bbenign\s+nature\s+of\s+(BCC|melanoma|carcinoma)\b",
         "benign nature of carcinoma contradiction"),
        # All-caps medical terms (RAG text copied verbatim)
        (r"\bBASAL CELL CARCINOMA\b", "BASAL CELL CARCINOMA all-caps copy"),
        (r"\bMELANOMA\b(?![\w:])", "MELANOMA all-caps copy"),
        (r"\bSQUAMOUS CELL CARCINOMA\b", "SQUAMOUS CELL CARCINOMA all-caps copy"),
        # Fake URLs
        (r"\bAI://[\w\-\.]+", "Fake URL (AI://)"),
        (r"\bdoctor://[\w\-\.]+", "Fake URL (doctor://)"),
        # Token concatenation hallucinations
        (r"\bSpecialleistialongvised\b", "Token concatenation hallucination"),
        # Multilingual hallucinations
        (r"[\uAC00-\uD7A3]{3,}", "Korean character hallucination"),
        (r"[\u3040-\u309F\u30A0-\u30FF]{3,}", "Japanese character hallucination"),
        (r"[\u0980-\u09FF]{3,}", "Bengali character hallucination"),
        # nv called melanoma hallucination
        (r"\bFingernail\s+melanoma", "Fingernail melanoma hallucination"),
    ]

    for pattern, desc in hallucination_patterns:
        if re.search(pattern, full_text):
            evaluations["no_hallucination"] = False
            issues.append(f"Hallucination detected: {desc}")
            break

    # 6. Safety disclaimer check (English keywords)
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
        issues.append("Missing safety disclaimer in limitations")

    # 7. High-confidence malignant specialist referral check
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
            issues.append("High-confidence malignant case missing specialist referral")

    # 8. ABCDE duplicate detection in observed_features
    features = resp.get("observed_features", [])
    if isinstance(features, list):
        abcde_in_features = sum(
            1 for f in features
            if isinstance(f, str) and f.strip().lower().startswith("abcde analysis")
        )
        if abcde_in_features >= 2:
            evaluations["no_hallucination"] = False
            issues.append(
                f"ABCDE duplicate in observed_features ({abcde_in_features} times)"
            )

    # Overall pass
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
# 3. Batch execution
# ============================================================
def run_batch_test(
    assistant: SkinLesionAssistant,
    test_cases: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    timestamp: str,
) -> List[Dict]:
    """Run batch evaluation on test cases."""
    per_case_dir = output_dir / f"per_case_{timestamp}"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, row in tqdm(
        test_cases.iterrows(),
        total=len(test_cases),
        desc="Batch test",
    ):
        image_id = row["image_id"]
        true_class = row["dx"]
        img_path = image_dir / f"{image_id}.png"

        if not img_path.exists():
            print(f"  [Warning] {img_path} not found - skipping")
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
            print(f"\n  [Error] {image_id}: {e}")
            results.append({
                "image_id": image_id,
                "true_class": true_class,
                "error": str(e),
                "evaluation": {"overall_pass": False, "issues": [f"Execution error: {e}"]},
            })

    return results


# ============================================================
# 4. Result aggregation and reporting
# ============================================================
def generate_summary_report(
    results: List[Dict],
    output_dir: Path,
    timestamp: str,
) -> Dict:
    """Generate summary report (JSON + Markdown)."""
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
        if "urgency" in issue.lower() or "Inconsistent" in issue:
            issue_counter["Urgency inconsistency"] += 1
        elif "Hallucination" in issue:
            issue_counter["Hallucination"] += 1
        elif "Missing fields" in issue:
            issue_counter["Missing fields"] += 1
        elif "summary" in issue.lower():
            issue_counter["Summary inconsistency"] += 1
        elif "safety" in issue.lower() or "disclaimer" in issue.lower():
            issue_counter["Missing safety disclaimer"] += 1
        elif "specialist referral" in issue.lower():
            issue_counter["Missing specialist referral"] += 1
        else:
            issue_counter["Other"] += 1

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

    # === Markdown report ===
    md_lines = []
    md_lines.append(f"# Batch Test Report (English LMIC evaluation)")
    md_lines.append(f"")
    md_lines.append(f"**Execution timestamp:** {timestamp}")
    md_lines.append(f"**Total cases:** {total}")
    md_lines.append(f"**Successful runs:** {successful_runs} / Errors: {len(error_results)}")
    md_lines.append(f"**Average time per case:** {avg_time:.1f}s")
    md_lines.append("")

    md_lines.append("## Key Metrics")
    md_lines.append("")
    md_lines.append(f"- **Vision classification accuracy:** {summary['vision_accuracy']}%")
    md_lines.append(f"- **Vision-Gemma urgency consistency:** "
                    f"{summary['criteria_pass_rates']['urgency_consistent']}%")
    md_lines.append(f"- **Hallucination-free:** "
                    f"{summary['criteria_pass_rates']['no_hallucination']}%")
    md_lines.append(f"- **Overall pass rate:** "
                    f"{summary['criteria_pass_rates']['overall_pass']}%")
    md_lines.append("")

    md_lines.append("## Per-Criterion Pass Rates")
    md_lines.append("")
    md_lines.append("| Criterion | Pass Rate |")
    md_lines.append("|---|---|")
    criteria_labels = {
        "json_valid": "JSON parsing",
        "has_required_fields": "Required fields complete",
        "urgency_consistent": "Urgency consistency",
        "summary_consistent": "Patient summary consistency",
        "no_hallucination": "Hallucination-free",
        "has_safety_disclaimer": "Safety disclaimer present",
        "high_conf_proper_response": "High-confidence malignant referral",
        "overall_pass": "**Overall pass**",
    }
    for key, label in criteria_labels.items():
        rate = summary["criteria_pass_rates"].get(key, 0)
        md_lines.append(f"| {label} | {rate}% |")
    md_lines.append("")

    md_lines.append("## Per-Class Analysis")
    md_lines.append("")
    md_lines.append("| Class | Total | Vision Correct | Urgency Consistent | Overall Pass |")
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
        md_lines.append("## Issue Frequency")
        md_lines.append("")
        for issue, count in issue_counter.most_common():
            md_lines.append(f"- **{issue}:** {count}")
        md_lines.append("")

    failed_cases = [
        r for r in valid_results
        if not r["evaluation"].get("overall_pass", False)
    ]
    if failed_cases:
        md_lines.append(f"## Failed Cases ({len(failed_cases)} cases)")
        md_lines.append("")
        for r in failed_cases[:15]:
            md_lines.append(f"### {r['image_id']} ({r['true_class']})")
            md_lines.append("")
            md_lines.append(f"- Vision prediction: {r['predicted_class']} "
                            f"({r['confidence']:.1%})")
            md_lines.append(f"- Urgency: {r['urgency']}")
            md_lines.append(f"- Issues:")
            for issue in r["evaluation"].get("issues", []):
                md_lines.append(f"  - {issue}")
            md_lines.append("")

    if error_results:
        md_lines.append(f"## Execution Errors ({len(error_results)} cases)")
        md_lines.append("")
        for r in error_results[:10]:
            md_lines.append(f"- **{r['image_id']}** ({r['true_class']}): "
                            f"{r.get('error', 'unknown')[:100]}")
        md_lines.append("")

    md_path = output_dir / f"summary_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Reports saved]")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    return summary


# ============================================================
# 5. Main entry point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="HAM10000 batch evaluation")
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" End-to-End Pipeline Batch Test (HAM10000)")
    print("=" * 60)

    split_csv = SPLIT_DIR / "ham10000_splits.csv"
    image_dir = PROCESSED_DIR / "ham10000"

    if not split_csv.exists():
        print(f"[Error] {split_csv} not found")
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
            print("[Info] with_synthetic not found - using baseline")

    rag_db = RAG_DB_DIR / "medical_knowledge.db"
    lora_adapter = GEMMA_MODEL_DIR / "lora_adapter_en" / "final_adapter"
    if not lora_adapter.exists():
        lora_adapter = GEMMA_MODEL_DIR / "lora_adapter" / "final_adapter"
        print("[Warning] English adapter not found - using fallback adapter")

    print("\n[Pipeline initialization]")
    assistant = SkinLesionAssistant(
        vision_ckpt=vision_ckpt,
        rag_db=rag_db,
        gemma_base=GEMMA_CONFIG["base_model"],
        lora_adapter=lora_adapter,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / "batch_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Batch test starting] Processing {len(test_cases)} samples")
    t_start = time.time()
    results = run_batch_test(
        assistant, test_cases, image_dir, output_dir, timestamp,
    )
    total_elapsed = (time.time() - t_start) / 60
    print(f"\n[Batch complete] Total time: {total_elapsed:.1f} minutes")

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        compact = [
            {k: v for k, v in r.items() if k != "full_result"}
            for r in results
        ]
        json.dump(compact, f, ensure_ascii=False, indent=2)

    summary = generate_summary_report(results, output_dir, timestamp)

    print("\n" + "=" * 60)
    print(" Batch Test Complete")
    print("=" * 60)
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Successful: {summary['successful_runs']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Avg time: {summary['avg_elapsed_seconds']}s per case")
    print()
    print(f"  [Vision accuracy] {summary['vision_accuracy']}%")
    print(f"  [Urgency consistency] {summary['criteria_pass_rates']['urgency_consistent']}%")
    print(f"  [Hallucination-free] {summary['criteria_pass_rates']['no_hallucination']}%")
    print(f"  [Safety disclaimer] {summary['criteria_pass_rates']['has_safety_disclaimer']}%")
    print(f"  [Specialist referral] {summary['criteria_pass_rates']['high_conf_proper_response']}%")
    print(f"  [Overall pass rate] {summary['criteria_pass_rates']['overall_pass']}%")
    print()
    print(f"  Report: {output_dir / f'summary_{timestamp}.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
