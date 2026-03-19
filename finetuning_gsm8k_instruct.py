import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

def parse_args():
    p = argparse.ArgumentParser(description="Full Fine-tune Llama-3.2-3B Instruct on GSM8K using TRL")

    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./gsm8k_llama32_3b_full_sft")
    
    # GSM8K 추론 과정 잘림 방지를 위해 충분한 길이 확보
    p.add_argument("--max_seq_length", type=int, default=1024)
    
    # Full FT는 메모리 소모가 크므로 배치 사이즈를 작게 유지하고 grad_accum으로 보완합니다.
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    
    # Full FT의 경우 기존 지식 망각(Catastrophic Forgetting)을 막기 위해 LoRA보다 훨씬 낮은 LR 사용
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

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
    train_ds = load_dataset("openai/gsm8k", "main", split="train")

    # 채팅 템플릿 포맷팅 함수
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            sys_msg = (
                "You are a helpful assistant that solves math problems step by step. "
                "Always show your reasoning and provide the final numerical answer after ####."
            )
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{example['question'][i]}"},
                {"role": "assistant", "content": example['answer'][i]}
            ]
            # apply_chat_template을 사용해 Llama 3 포맷 문자열로 변환
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return {"text": output_texts}

    # 포맷팅 함수를 dataset에 미리 적용 (completion_only_loss와 함께 사용하기 위해)
    train_ds = train_ds.map(formatting_prompts_func, batched=True, remove_columns=train_ds.column_names)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True, 
        gradient_checkpointing=True, # 메모리 절약을 위한 필수 옵션
        logging_steps=10,
        save_strategy="no", 
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_seq_length,  # 최대 시퀀스 길이 설정
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting full parameter fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Full Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()