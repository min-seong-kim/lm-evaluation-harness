'''
python finetuning_gsm8k_instruct_v2.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./gsm8k_llama32_3b_full_sft \
  --max_seq_length 2048 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --grad_accum 8 \
  --epochs 2 \
  --learning_rate 5e-6 \
  --validation_size 750 \
  --strip_calculator_annotations \
  --train_on_mixed_formats
'''
import argparse
import datetime
import os
import random
import re
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


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
CALC_ANNOTATION_RE = re.compile(r"<<.*?>>")


def parse_args():
    p = argparse.ArgumentParser(
        description="Full Fine-tune Llama-3.2-3B Instruct on GSM8K with chat template, mixed target formats, and validation split"
    )

    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./gsm8k_llama32_3b_full_sft")

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--validation_size", type=int, default=750,
                   help="Number of samples split from GSM8K train for validation.")
    p.add_argument("--strip_calculator_annotations", action="store_true",
                   help="Remove <<...>> calculator annotations from GSM8K rationale text.")
    p.add_argument("--train_on_mixed_formats", action="store_true",
                   help="Train with a mixture of answer formats instead of only the original #### format.")
    p.add_argument("--system_prompt", type=str,
                   default="You are a helpful math tutor. Solve the problem carefully and give the final answer clearly.")

    return p.parse_args()


def maybe_strip_annotations(text: str, strip: bool) -> str:
    text = text.strip()
    if strip:
        text = CALC_ANNOTATION_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_final_answer_number(answer_text: str) -> str:
    match = FINAL_ANSWER_RE.search(answer_text)
    if not match:
        raise ValueError(f"Could not extract final answer from: {answer_text!r}")
    return match.group(1).replace(",", "")


def build_mixed_target(answer_text: str, rng: random.Random) -> str:
    """
    Mix target formats so the model learns the numeric answer more robustly,
    instead of overfitting only to the exact '#### answer' style.
    """
    cleaned = answer_text.strip()
    final_number = extract_final_answer_number(cleaned)
    rationale_wo_hash = FINAL_ANSWER_RE.sub("", cleaned).strip()

    draw = rng.random()
    if draw < 0.50:
        # Original GSM8K style
        target = cleaned
    elif draw < 0.80:
        # Natural-language final answer style
        if rationale_wo_hash:
            target = f"{rationale_wo_hash}\nTherefore, the answer is {final_number}."
        else:
            target = f"Therefore, the answer is {final_number}."
    else:
        # Short answer only
        target = f"The answer is {final_number}."
    return target.strip()


def build_messages(question: str, target_text: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": target_text},
    ]


def render_prompt_only(tokenizer, question: str, system_prompt: str) -> str:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_full_conversation(tokenizer, question: str, target_text: str, system_prompt: str) -> str:
    messages = build_messages(question, target_text, system_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_and_mask(example, tokenizer, max_length: int, system_prompt: str,
                      strip_calculator_annotations: bool, train_on_mixed_formats: bool,
                      seed: int, idx: int):
    question = example["question"].strip()
    raw_answer = maybe_strip_annotations(example["answer"], strip_calculator_annotations)

    if train_on_mixed_formats:
        rng = random.Random(seed + idx)
        target_text = build_mixed_target(raw_answer, rng)
    else:
        target_text = raw_answer

    # Explicit EOS for cleaner stopping behavior.
    target_text = target_text + tokenizer.eos_token

    prompt_text = render_prompt_only(tokenizer, question, system_prompt)
    full_text = render_full_conversation(tokenizer, question, target_text, system_prompt)

    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]
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
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

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


def preview_examples(tokenizer, dataset, args):
    if len(dataset) == 0:
        return

    sample = dataset[0]
    question = sample["question"].strip()
    answer = maybe_strip_annotations(sample["answer"], args.strip_calculator_annotations)
    target_text = build_mixed_target(answer, random.Random(args.seed)) if args.train_on_mixed_formats else answer

    prompt_text = render_prompt_only(tokenizer, question, args.system_prompt)
    full_text = render_full_conversation(tokenizer, question, target_text + tokenizer.eos_token, args.system_prompt)

    print("\n=== First raw GSM8K sample ===")
    print("Question:")
    print(question)
    print("\nAnswer:")
    print(sample["answer"])
    print("=== End raw sample ===\n")

    print("\n=== Prompt-only preview ===")
    print(prompt_text[:1200])
    print("=== End prompt preview ===\n")

    print("\n=== Full formatted training text preview ===")
    print(full_text[:1500])
    print("=== End formatted preview ===\n")


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
    tokenizer.padding_side = "right"

    print("Loading model for Full Fine-Tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading GSM8K dataset...")
    full_train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)

    val_size = min(args.validation_size, max(1, len(full_train_ds) // 10))
    split_ds = full_train_ds.train_test_split(test_size=val_size, seed=args.seed)
    train_ds = split_ds["train"]
    valid_ds = split_ds["test"]

    print(f"Train size: {len(train_ds)} | Validation size: {len(valid_ds)}")
    preview_examples(tokenizer, train_ds, args)

    print("Tokenizing and masking datasets (assistant-only loss)...")
    train_tok = train_ds.map(
        lambda ex, idx: tokenize_and_mask(
            ex,
            tokenizer,
            args.max_seq_length,
            args.system_prompt,
            args.strip_calculator_annotations,
            args.train_on_mixed_formats,
            args.seed,
            idx,
        ),
        with_indices=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )

    valid_tok = valid_ds.map(
        lambda ex, idx: tokenize_and_mask(
            ex,
            tokenizer,
            args.max_seq_length,
            args.system_prompt,
            args.strip_calculator_annotations,
            False,  # validation은 원본 형식으로 고정
            args.seed,
            idx,
        ),
        with_indices=True,
        remove_columns=valid_ds.column_names,
        desc="Tokenizing validation set",
    )

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=resolved_output_dir,
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
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
        eval_dataset=valid_tok,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    print("Starting full parameter fine-tuning...")
    trainer.train()

    print("Running final validation loss evaluation...")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    print("Saving fine-tuned model...")
    trainer.save_model(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)
    print("Full fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
