'''
python finetuning_gsm8k_instruct_flexible.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./gsm8k_llama32_3b_full_sft_flexible \
  --max_seq_length 2048 \
  --batch_size 4 \
  --grad_accum 8 \
  --epochs 1 \
  --learning_rate 7e-7
'''
import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Full Fine-tune Llama-3.2-3B Instruct on GSM8K (flexible-focused)"
    )

    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./gsm8k_llama32_3b_full_sft_flexible")

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=7e-7)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--use_chat_template",
        action="store_true",
        default=True,
        help="Format training samples with chat template to better match instruct behavior.",
    )
    p.add_argument(
        "--no_use_chat_template",
        action="store_false",
        dest="use_chat_template",
        help="Disable chat template and use plain prompt format.",
    )

    return p.parse_args()


def split_gsm8k_answer(answer: str) -> Tuple[str, str]:
    text = answer.strip()
    if "####" in text:
        rationale, final = text.rsplit("####", 1)
        return rationale.strip(), final.strip()
    return text, text


def build_flexible_target(answer: str) -> str:
    rationale, final = split_gsm8k_answer(answer)
    # Avoid overfitting to strict marker `####`; keep a natural final sentence.
    if rationale:
        return f"{rationale}\nTherefore, the answer is {final}."
    return f"The answer is {final}."


def build_plain_prompt_and_target(example: Dict[str, str]) -> Tuple[str, str]:
    prompt = (
        "Solve the following math word problem. "
        "Explain your reasoning and provide the final numeric answer in the last sentence.\n\n"
        f"Question: {example['question']}\nAnswer:"
    )
    target = build_flexible_target(example["answer"])
    return prompt, target


def build_chat_prompt_and_target(example: Dict[str, str]) -> Tuple[str, str]:
    user_content = (
        "Solve the following grade-school math problem. "
        "Show your reasoning and end with one clear final numeric answer.\n\n"
        f"Question: {example['question']}"
    )
    target = build_flexible_target(example["answer"])
    return user_content, target


def tokenize_and_mask(example, tokenizer, max_length, use_chat_template):
    if use_chat_template:
        user_content, target_text = build_chat_prompt_and_target(example)

        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_text},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text, target_text = build_plain_prompt_and_target(example)
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model for Full Fine-Tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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

        print("\n=== First formatted training text (preview) ===")
        if args.use_chat_template:
            user_content, target = build_chat_prompt_and_target(train_ds[0])
            preview = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt, target = build_plain_prompt_and_target(train_ds[0])
            preview = f"{prompt} {target}"
        print(preview[:1400])
        print("\n=== End formatted preview ===\n")

    print("Tokenizing and masking dataset (assistant-only loss)...")
    train_tok = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_length, args.use_chat_template),
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

    print("Starting full parameter fine-tuning (flexible-focused)...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)
    print("Flexible-focused full fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
