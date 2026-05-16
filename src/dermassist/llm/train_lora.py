"""
09_gemma_lora_finetune.py (v4 — 진단 결과 반영 최종본)
========================================================
[Gemma 트랙 - 3주차] Gemma 4 E4B LoRA 파인튜닝

변경사항 (v3 → v4):
  - target_modules 제거 (PEFT 0.19+의 Gemma 4 자동 탐지 사용)
    → 멀티모달 ClippableLinear 문제 회피
  - torch_dtype → dtype (deprecation 해결)
  - prepare_model_for_kbit_training에 use_reentrant=False 명시
  - 검증용 max_steps 옵션 추가 (환경변수로 제어)

실행:
  # 검증용 (100 스텝만, 30~40분)
  $env:QUICK_TEST="1"
  python 09_gemma_lora_finetune.py

  # 본 학습
  python 09_gemma_lora_finetune.py
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    OUTPUT_DIR, GEMMA_MODEL_DIR, LOG_DIR, GEMMA_CONFIG,
)


# ============================================================
# 1. 환경 설정
# ============================================================
def setup_environment():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip().strip('"')
                    os.environ["HF_TOKEN"] = hf_token
                    break

    if not hf_token:
        print("[경고] HF_TOKEN 없음")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("[오류] CUDA 사용 불가")
        sys.exit(1)

    import transformers, peft, trl
    print(f"[transformers] {transformers.__version__}")
    print(f"[peft] {peft.__version__}")
    print(f"[trl] {trl.__version__}")


# ============================================================
# 2. 학습 데이터 로드
# ============================================================
def load_training_data(jsonl_path: Path) -> List[Dict]:
    if not jsonl_path.exists():
        print(f"[오류] {jsonl_path} 없음")
        sys.exit(1)

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"[데이터] {len(samples)}건 로드")
    type_counts = {}
    for s in samples:
        st = s.get("scenario_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
    print(f"  분포: {type_counts}")
    return samples


def split_train_val(samples: List[Dict], val_ratio: float = 0.05):
    import random
    random.seed(42)
    random.shuffle(samples)
    val_size = max(10, int(len(samples) * val_ratio))
    train = samples[val_size:]
    val = samples[:val_size]
    print(f"  Train: {len(train)} / Val: {len(val)}")
    return train, val


# ============================================================
# 3. 모델 & 프로세서 로드
# ============================================================
def load_model_and_processor(base_model_id: str):
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[모델 ID] {base_model_id}")
    print(f"[모델 클래스] AutoModelForCausalLM (공식 권장)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(base_model_id)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dtype 사용 (torch_dtype은 deprecated)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "14GiB", "cpu": "30GiB"},
        dtype=torch.bfloat16,
        #attn_implementation="sdpa",
    )
    model.config.use_cache = False

    print(f"  Device: {next(model.parameters()).device}")
    print(f"  VRAM 사용: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return model, processor, tokenizer


# ============================================================
# 4. LoRA 설정 — target_modules 제거 (PEFT 자동 탐지)
# ============================================================
def setup_lora(model):
    """
    Gemma 4 + PEFT 0.19+ 조합 권장 방식.

    PEFT 0.19+는 Gemma 4에 대해 language_model 레이어만 자동으로 타겟팅합니다.
    target_modules를 명시하면 multimodal ClippableLinear 문제로
    gradient가 LoRA 어댑터에 흐르지 않습니다 (grad_norm=0 증상).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # gradient checkpointing 옵션 명시 (진단 스크립트와 동일)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # target_modules 완전 생략 — PEFT가 Gemma 4 default regex 자동 사용
    lora_config = LoraConfig(
        r=GEMMA_CONFIG["lora_r"],
        lora_alpha=GEMMA_CONFIG["lora_alpha"],
        lora_dropout=GEMMA_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules 라인 없음 → 자동 탐지
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# 5. 데이터 포매팅
# ============================================================
def format_sample(sample: Dict, processor) -> Dict:
    messages = sample["messages"]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}


def prepare_dataset(samples: List[Dict], processor):
    from datasets import Dataset
    formatted = [format_sample(s, processor) for s in samples]
    return Dataset.from_list(formatted)


# ============================================================
# 6. 학습 실행
# ============================================================
def train(model, tokenizer, train_ds, val_ds, output_dir: Path):
    from trl import SFTTrainer, SFTConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    # 환경변수로 빠른 검증 모드 (100 스텝만)
    quick_test = os.getenv("QUICK_TEST", "0") == "1"

    training_args_dict = dict(
        output_dir=str(output_dir),

        num_train_epochs=GEMMA_CONFIG["epochs"],
        per_device_train_batch_size=GEMMA_CONFIG["batch_size"],
        per_device_eval_batch_size=GEMMA_CONFIG["batch_size"],
        gradient_accumulation_steps=GEMMA_CONFIG["gradient_accumulation_steps"],

        learning_rate=GEMMA_CONFIG["lr"],
        #warmup_ratio=GEMMA_CONFIG["warmup_ratio"],
        warmup_steps=GEMMA_CONFIG["warmup_steps"],
        lr_scheduler_type="cosine",

        bf16=True,
        max_grad_norm=0.3,   # 안전망 1.0
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", # "paged_adamw_8bit",

        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,

        max_length=GEMMA_CONFIG["max_seq_length"],
        packing=False,
        dataset_text_field="text",

        seed=42,
        report_to="none",
    )

    if quick_test:
        training_args_dict["max_steps"] = 100
        training_args_dict["eval_steps"] = 50
        print("[검증 모드] max_steps=100 — 100 스텝만 학습")

    training_args = SFTConfig(**training_args_dict)

    trainer = SFTTrainer(
        model=model,
        #tokenizer=tokenizer,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    print("\n[학습 시작]")
    if quick_test:
        print("  [모드] 검증 (100 스텝)")
    else:
        print(f"  [모드] 본 학습 (Epochs: {GEMMA_CONFIG['epochs']})")
    print(f"  Train: {len(train_ds)} / Val: {len(val_ds)}")
    effective_bs = GEMMA_CONFIG['batch_size'] * GEMMA_CONFIG['gradient_accumulation_steps']
    print(f"  Effective batch size: {effective_bs}")

    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 3600
    print(f"\n[학습 완료] {elapsed:.2f}시간")

    # 검증 모드에서는 최종 저장 건너뜀
    if not quick_test:
        final_dir = output_dir / "final_adapter"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"[어댑터 저장] {final_dir}")

    return trainer, quick_test


# ============================================================
# 7. 샘플 생성 테스트
# ============================================================
def test_generation(
    model, processor, tokenizer,
    val_samples: List[Dict], num_tests: int = 3,
):
    print("\n" + "=" * 60)
    print(" 샘플 생성 테스트")
    print("=" * 60)

    model.eval()

    for i, sample in enumerate(val_samples[:num_tests]):
        print(f"\n--- Test {i+1} / Scenario: {sample.get('scenario_type')} ---")

        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                top_k=64,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        preview = generated[:500] + ("..." if len(generated) > 500 else "")
        print(f"[Generated]\n{preview}")

        try:
            start = generated.find("{")
            end = generated.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(generated[start:end])
                has_urgency = "urgency" in parsed
                has_summary = "patient_summary" in parsed
                print(f"[검증] JSON OK, urgency={has_urgency}, "
                      f"patient_summary={has_summary}")
        except json.JSONDecodeError:
            print("[검증] JSON 파싱 실패")

    print("\n" + "=" * 60)


# ============================================================
# 8. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print(" [3주차] Gemma 4 E4B LoRA 파인튜닝 (v4)")
    print("=" * 60)

    setup_environment()

    #jsonl_path = OUTPUT_DIR / "gemma_training_data" / "training_data.jsonl"
    jsonl_path = OUTPUT_DIR / "gemma_training_data_en" / "training_data.jsonl"
    
    samples = load_training_data(jsonl_path)
    train_samples, val_samples = split_train_val(samples, val_ratio=0.05)

    model, processor, tokenizer = load_model_and_processor(
        GEMMA_CONFIG["base_model"]
    )

    print("\n[LoRA 설정]")
    model = setup_lora(model)

    print("\n[데이터셋 포매팅]")
    train_ds = prepare_dataset(train_samples, processor)
    val_ds = prepare_dataset(val_samples, processor)

    import numpy as np
    lengths = [len(tokenizer.encode(s["text"])) for s in train_ds]
    print(f"  토큰 길이: mean={np.mean(lengths):.0f}, "
          f"max={np.max(lengths)}, min={np.min(lengths)}")
    if np.max(lengths) > GEMMA_CONFIG["max_seq_length"]:
        print(f"  [경고] 최대 길이 초과")

    #lora_output_dir = GEMMA_MODEL_DIR / "lora_adapter"
    lora_output_dir = GEMMA_MODEL_DIR / "lora_adapter_en"  # 영어 어댑터 별도 저장
    trainer, quick_test = train(model, tokenizer, train_ds, val_ds, lora_output_dir)

    # 본 학습에서만 샘플 생성 테스트 & 이력 저장
    if not quick_test:
        test_generation(model, processor, tokenizer, val_samples, num_tests=3)

        history_path = lora_output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)

        processor_path = lora_output_dir / "final_adapter"
        processor.save_pretrained(str(processor_path))
        print(f"\n[Processor 저장] {processor_path}")

        print("\n" + "=" * 60)
        print(" Gemma LoRA 파인튜닝 완료")
        print(f"  어댑터: {lora_output_dir / 'final_adapter'}")
        print(f"  다음 단계: python 10_integrated_pipeline.py")
        print("=" * 60)
    else:
        # 검증 결과 요약
        log = trainer.state.log_history
        train_losses = [l["loss"] for l in log if "loss" in l and "eval_loss" not in l]

        print("\n" + "=" * 60)
        print(" 검증 완료 — 결과 분석")
        print("=" * 60)
        if len(train_losses) >= 2:
            first_loss = float(train_losses[0])
            last_loss = float(train_losses[-1])
            delta = last_loss - first_loss
            print(f"  첫 loss: {first_loss:.4f}")
            print(f"  마지막 loss: {last_loss:.4f}")
            print(f"  변화: {delta:+.4f}")

            if first_loss > 20:
                print("\n  [비정상] 초기 loss가 너무 높습니다.")
                print("  → 데이터 포맷 또는 토크나이저 문제")
            elif delta < -0.3:
                print("\n  [정상] Loss가 감소하고 있습니다.")
                print("  → QUICK_TEST 환경변수 해제 후 본 학습 진행:")
                print("    Remove-Item Env:QUICK_TEST")
                print("    python 09_gemma_lora_finetune.py")
            elif delta > -0.1:
                print("\n  [비정상] Loss가 거의 감소하지 않습니다.")
                print("  → grad_norm 확인 필요")


if __name__ == "__main__":
    main()
