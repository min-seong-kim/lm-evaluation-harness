import argparse
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
    p = argparse.ArgumentParser(description="Optimized Full FT Llama-3.2-3B on MetaMathQA")
    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./metamath_llama32_3b_optimized")
    
    # 길이가 긴 MetaMathQA를 위해 2048로 확장
    p.add_argument("--max_length", type=int, default=1024)
    # 데이터 수를 10만개로 증강
    p.add_argument("--num_train_samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=3)
    
    # OOM 방지를 위해 batch는 줄이고 grad_accum을 늘려 글로벌 배치 확보
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def tokenize_and_mask(example, tokenizer, max_length):
    """프롬프트는 학습(Loss)에서 제외하고 Assistant 답변만 학습하도록 라벨 마스킹"""
    system_msg = "You are a logical assistant that solves math problems step by step. Show your complete reasoning."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example['query']},
        {"role": "assistant", "content": example['response']}
    ]

    # 1. 템플릿을 문자열(Text)로 먼저 변환 (버전 충돌 원천 차단)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 기본 tokenizer로 토크나이징 (항상 순수 파이썬 리스트 반환)
    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]
    
    # 2. Assistant 답변을 제외한 프롬프트 부분만 문자열로 변환
    prompt_only = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
    
    # 프롬프트 토크나이징
    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)
    prompt_ids = prompt_encoded["input_ids"]

    # 3. Labels 마스킹 처리 (-100)
    labels = input_ids.copy()

    # 프롬프트 길이를 구해서 그만큼 -100(Loss 계산 제외) 처리
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

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
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Loading and preparing MetaMathQA dataset...")
    full_ds = load_dataset("meta-math/MetaMathQA", split="train")
    full_ds = full_ds.shuffle(seed=args.seed)
    train_ds = full_ds.select(range(min(args.num_train_samples, len(full_ds))))

    # 병렬 토크나이징 진행 (TRL 대신 HF Trainer 방식 적용)
    train_tok = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Tokenizing & Masking"
    )

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=200,
        remove_unused_columns=False,
    )

    print("Starting optimized full fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    print("Saving optimized model...")
    trainer.save_model(args.output_dir)
    print("Optimization Complete!")

if __name__ == "__main__":
    main()