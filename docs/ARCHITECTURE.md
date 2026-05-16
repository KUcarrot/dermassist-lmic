# Architecture

## System Overview

DermAssist LMIC consists of 5 main components running sequentially.

```
[Skin Lesion Image]
    |
    v
[1. DullRazor Preprocessing]
    |
    v
[2. Vision Classifier (EfficientNet-B4)]
    |
    +--> Predicted class, confidence, Grad-CAM
    |
    v
[3. RAG Retrieval (BAAI/bge-m3)]
    |
    +--> Top-5 medical context documents
    |
    v
[4. Gemma 4 E4B + LoRA Reasoning]
    |
    +--> Structured JSON output
    |
    v
[5. Output Validation]
    |
    v
[Urgency Triage | Recommendation | Patient Summary | Limitations]
```

## Component Details

### 1. DullRazor Hair Removal
- **Module:** `src/dermassist/pipeline/assistant.py`
- **Algorithm:** Lee et al. (1997) - Black-hat morphology + inpainting
- **Purpose:** Reduce hair artifacts that affect classifier accuracy
- **Processing time:** ~50ms per image

### 2. Vision Classifier
- **Module:** `src/dermassist/vision/`
- **Architecture:** EfficientNet-B4
- **Training data:** HAM10000 (10,015 dermatoscopic images, 7 classes)
- **Output:** Predicted class, top-3 probabilities, Grad-CAM attention map

### 3. RAG Knowledge Base
- **Module:** `src/dermassist/rag/`
- **Embedding model:** BAAI/bge-m3 (1024-dimensional)
- **Storage:** SQLite + FAISS
- **Sources:** DermNet, BAD guidelines, WHO LMIC dermatology protocols

### 4. Gemma 4 LLM Reasoning
- **Module:** `src/dermassist/llm/`
- **Base model:** Gemma 4 E4B (8B parameters)
- **Quantization:** 4-bit (bitsandbytes)
- **Adapter:** LoRA (9.1M trainable parameters)
- **Training data:** 5,000 LMIC-specialized samples

### 5. Output Validation
- **Module:** `src/dermassist/pipeline/assistant.py`
- **Functions:**
  - JSON parsing with robust error recovery
  - Urgency-recommendation consistency enforcement
  - Hallucination pattern detection

## Patient Context Awareness

The system supports 8 patient risk profiles aligned with LMIC realities:
- General LMIC patient
- Rural farmer (chronic UV exposure)
- Patient with albinism (high cancer risk)
- Child with albinism (early sun damage)
- HIV positive (immunocompromised)
- Outdoor laborer
- Elderly rural
- Remote area (no specialist access)

And 5 healthcare resource constraints:
- Standard primary care (specialist available)
- Long distance (200+ km to specialist)
- No biopsy available locally
- Long wait time (4-8 weeks)
- Teledermatology available

These contexts are integrated into both the RAG retrieval queries and the LLM prompt construction.
