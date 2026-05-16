# Examples

This directory contains example dermoscopy images and recommended patient scenarios for testing the DermAssist LMIC pipeline.

## Quick Start

Launch the Gradio UI and use the example images below to reproduce the scenarios from the demo video.

```bash
python scripts/06_run_demo.py
```

Then upload an image from `examples/images/` and enter the corresponding patient information.

---

## Scenario A: Benign Case (Routine Follow-up)

**Image:** `images/nv_ISIC_0025610.jpg`

A common benign melanocytic nevus. Demonstrates that the system avoids false alarms and correctly routes low-risk cases.

### Expected Pipeline Output

- **Vision Classifier:** `nv` (benign nevus) with high confidence
- **Urgency:** Routine (green badge)
- **Recommendation:** Routine follow-up, no urgent action needed

### Patient Information

| Field | Value |
|---|---|
| Age | 50 |
| Sex | male |
| Body site | back |
| Lesion duration | 12 months |
| Symptoms | asymptomatic (no notable symptoms) |
| Recent size increase | No |
| Itching / pain | No |
| Profile | General LMIC patient |
| Resource setting | Standard primary care (specialist available) |

---

## Scenario B: Suspicious Case (Urgent Referral)

**Image:** `images/bcc_ISIC_0031527.jpg`

A pigmented lesion on the scalp of a patient with albinism. This scenario demonstrates the **safety-by-design** behavior: even when the Vision Classifier is uncertain, the LLM layer maintains safe urgency assignment based on patient risk factors.

### Expected Pipeline Output

- **Vision Classifier:** May predict `bcc` or another class with variable confidence
- **Urgency:** Urgent (red badge)
- **Recommendation:** Urgent specialist referral with explicit acknowledgment of the 200+ km travel barrier; teledermatology suggested

### Patient Information

| Field | Value |
|---|---|
| Age | 22 |
| Sex | female |
| Body site | scalp |
| Lesion duration | 8 months |
| Symptoms (multi-select) | `rapid growth over past 3 months`, `non-healing ulceration` |
| Recent size increase | Yes |
| Itching / pain | No |
| Profile | Patient with albinism (high cancer risk) |
| Resource setting | Long distance (200+ km to specialist) |

### Why This Scenario Matters

In Sub-Saharan Africa, patients with oculocutaneous albinism (OCA) face significantly elevated skin cancer risk due to lack of melanin protection. Combined with limited specialist access (often 200+ km travel), early detection and triage are critical. This case demonstrates the system's primary use case: providing high-stakes triage support where dermatologist density is below 1 per million population.

---

## Image License and Attribution

The example images are sourced from the **ISIC Archive** (International Skin Imaging Collaboration).

- **Source:** https://www.isic-archive.com
- **Dataset:** HAM10000 (Tschandl et al., 2018)
- **License:** CC BY-NC 4.0 (Attribution-NonCommercial)
- **Usage:** Research and educational use only. Commercial use is not permitted.

### Citation

```
Tschandl, P., Rosendahl, C. & Kittler, H.
The HAM10000 dataset, a large collection of multi-source dermatoscopic images
of common pigmented skin lesions.
Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
```

---

## Notes for Demo Reproducibility

- **Greedy decoding:** The Gemma 4 inference uses `do_sample=False`, so outputs are deterministic for the same input.
- **First run latency:** Initial model loading takes 30-60 seconds. Subsequent inferences run in approximately 60 seconds per case.

For the full pipeline architecture, see the main [README.md](../README.md) at the project root.
