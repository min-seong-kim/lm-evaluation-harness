import argparse
import datetime
import os
import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

def parse_args():
    p = argparse.ArgumentParser(description="Full Fine-tune Llama-3.2-3B Instruct on GSM8K using TRL")

    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./gsm8k_llama32_3b_full_sft")
    
    # GSM8K 추론 과정 잘림 방지를 위해 충분한 길이 확보
    p.add_argument("--max_seq_length", type=int, default=2048)
    
    # Full FT는 메모리 소모가 크므로 배치 사이즈를 작게 유지하고 grad_accum으로 보완합니다.
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    
    # Full FT의 경우 기존 지식 망각(Catastrophic Forgetting)을 막기 위해 LoRA보다 훨씬 낮은 LR 사용
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    
    return p.parse_args()


def build_prompt_and_target(example):
    """Build plain-text prompt/target to match lm_eval's GSM8K input style."""
    prompt = f"Question: {example['question']}\nAnswer:"
    target = example["answer"].strip()
    return prompt, target


def tokenize_and_mask(example, tokenizer, max_length):
    """Mask prompt tokens and keep only assistant response tokens for loss."""
    prompt_text, target_text = build_prompt_and_target(example)
    full_text = f"{prompt_text} {target_text}"
    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]

    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)
    prompt_ids = prompt_encoded["input_ids"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(resolved_output_dir, exist_ok=True)
    print(f"Resolved output directory: {resolved_output_dir}")

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 패딩 토큰 설정 (Llama 3는 기본 pad_token이 없으므로 eos_token으로 대체)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model for Full Fine-Tuning...")
    # bfloat16 사용으로 메모리 절약 및 학습 안정성 확보
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading GSM8K dataset...")
    train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)

    if len(train_ds) > 0:
        print("\n=== First raw GSM8K sample ===")
        print("Question:")
        print(train_ds[0]["question"])
        print("\nAnswer:")
        print(train_ds[0]["answer"])
        print("=== End raw sample ===\n")

    if len(train_ds) > 0:
        print("\n=== First formatted training text (preview) ===")
        first_prompt, first_target = build_prompt_and_target(train_ds[0])
        first_text = f"{first_prompt} {first_target}"
        print(first_text[:1200])
        print("\n=== End formatted preview ===\n")

    print("Tokenizing and masking dataset (assistant-only loss)...")
    train_tok = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_length),
        remove_columns=train_ds.column_names,
        desc="Tokenizing and Masking",
    )
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=resolved_output_dir,
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    print("Starting full parameter fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)
    print("Full Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()