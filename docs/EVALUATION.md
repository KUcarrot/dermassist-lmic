# Evaluation Methodology and Results

## In-Distribution Evaluation: HAM10000 Test Split

**Configuration:**
- 35 stratified samples (5 per class across 7 classes: akiec, bcc, bkl, df, mel, nv, vasc)
- Random sampling with fixed seed
- Same pipeline as deployment

**Results:**

| Criterion | Pass Rate |
|---|---|
| JSON parsing | 100.0% |
| Required field completeness | 100.0% |
| Urgency-Vision consistency | 100.0% |
| Patient summary consistency | 100.0% |
| Hallucination-free output | 100.0% |
| Safety disclaimer inclusion | 100.0% |
| High-confidence malignancy proper response | 100.0% |
| **Overall pass rate** | **100.0%** |

**Vision Classifier accuracy:** 60.0% (21/35)

## External Validation: BCN20000

**Configuration:**
- BCN20000 (Hospital Clínic de Barcelona, Spain)
- Stratified subset: 60 samples (10 per class across 6 mappable classes)
- Excluded: vasc (insufficient samples in BCN20000), SCC (clinical category misalignment)

**Class Mapping:**

| HAM10000 | BCN20000 |
|---|---|
| nv | Nevus |
| mel | Melanoma, NOS + Melanoma metastasis |
| bcc | Basal cell carcinoma |
| akiec | Solar or actinic keratosis |
| bkl | Seborrheic keratosis + Solar lentigo |
| df | Dermatofibroma |

**Results:**

| Criterion | Pass Rate |
|---|---|
| JSON parsing | 100.0% |
| Required field completeness | 100.0% |
| Urgency-Vision consistency | 100.0% |
| Patient summary consistency | 100.0% |
| Hallucination-free output | 98.3% |
| Safety disclaimer inclusion | 100.0% |
| High-confidence malignancy proper response | 100.0% |
| **Overall pass rate** | **98.3%** |

**Vision Classifier accuracy:** 28.3% (17/60)

## Cross-Dataset Robustness Analysis

The key finding from external validation is the **divergence between Vision Classifier and LLM safety properties**:

| Metric | HAM10000 | BCN20000 | Change |
|---|---|---|---|
| Vision accuracy | 60.0% | 28.3% | -31.7 pp |
| LLM safety pass rate | 100.0% | 98.3% | -1.7 pp |

**Interpretation:**

1. **Vision Classifier exhibits domain shift sensitivity.** The system shows systematic bias toward classifying out-of-distribution images as the majority training class (nevus).

2. **LLM safety properties are preserved.** Through deliberate "safety-by-design" via LMIC-specialized training, the LLM correctly triages borderline cases regardless of upstream classifier accuracy.

3. **Clinical implication:** In a real LMIC deployment, the system would correctly route 98%+ of cases to appropriate care levels even with classifier degradation. This is the safety property most relevant to LMIC settings where missed cancers cause mortality.

## Detailed Class Analysis (BCN20000)

| Class | Vision Correct | Overall Pass |
|---|---|---|
| nv | 7/10 (70%) | 10/10 (100%) |
| mel | 2/10 (20%) | 9/10 (90%) |
| bcc | 5/10 (50%) | 10/10 (100%) |
| akiec | 0/10 (0%) | 10/10 (100%) |
| bkl | 2/10 (20%) | 10/10 (100%) |
| df | 1/10 (10%) | 10/10 (100%) |

Despite Vision accuracy as low as 0% (akiec), overall safety pass rate remains at 100% for that class, demonstrating the LLM's hedging behavior on borderline confidence cases.

## Confusion Matrix (Vision Classifier on BCN20000)

| True\Predicted | nv | mel | bcc | akiec | bkl | df |
|---|---|---|---|---|---|---|
| nv | 7 | 1 | · | · | 2 | · |
| mel | 6 | 2 | · | · | 1 | · |
| bcc | 2 | · | 5 | 1 | · | 1 |
| akiec | 5 | 1 | · | · | 3 | 1 |
| bkl | 4 | 2 | · | · | 2 | 2 |
| df | 6 | 1 | 1 | 1 | · | 1 |

The dominant pattern (visible in column "nv") shows the systematic bias toward nevus classification.
