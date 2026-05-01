#!/usr/bin/env python3
"""Check tokenizer/model vocab alignment without fully loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_from_safetensors(st_path: Path, key: str) -> Optional[Tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception:
        print("[ERROR] safetensors package is required but not available.")
        return None

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        if key not in f.keys():
            return None
        return tuple(f.get_slice(key).get_shape())


def _find_weight_shape(model_dir: Path, candidate_keys: list[str]) -> tuple[Optional[str], Optional[Tuple[int, ...]], Optional[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    single_st_path = model_dir / "model.safetensors"

    if index_path.exists():
        index = _read_json(index_path)
        weight_map = index.get("weight_map", {})
        for key in candidate_keys:
            if key in weight_map:
                shard = model_dir / weight_map[key]
                shape = _get_from_safetensors(shard, key)
                if shape is not None:
                    return key, shape, shard.name
        return None, None, None

    if single_st_path.exists():
        for key in candidate_keys:
            shape = _get_from_safetensors(single_st_path, key)
            if shape is not None:
                return key, shape, single_st_path.name
        return None, None, None

    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tokenizer vocab and model embedding size alignment.")
    parser.add_argument("--model-path", required=True, help="Local path to model directory.")
    args = parser.parse_args()

    model_dir = Path(args.model_path).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"[ERROR] model path is invalid: {model_dir}")
        return 2

    print(f"[INFO] model_dir={model_dir}")

    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as e:
        print(f"[ERROR] transformers import failed: {e}")
        return 2

    # Tokenizer/config are lightweight and safe to load.
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] failed to load tokenizer/config: {e}")
        return 2

    tokenizer_vocab_size = int(getattr(tok, "vocab_size", -1))
    tokenizer_total_tokens = len(tok)
    config_vocab_size = int(getattr(cfg, "vocab_size", -1))

    input_candidates = [
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
    ]
    output_candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
        "output.weight",
    ]

    in_key, in_shape, in_file = _find_weight_shape(model_dir, input_candidates)
    out_key, out_shape, out_file = _find_weight_shape(model_dir, output_candidates)

    print("\n=== Vocab Report ===")
    print(f"tokenizer.vocab_size: {tokenizer_vocab_size}")
    print(f"len(tokenizer):       {tokenizer_total_tokens}")
    print(f"config.vocab_size:    {config_vocab_size}")

    if in_shape is not None:
        print(f"input embedding:      {in_key} shape={in_shape} ({in_file})")
    else:
        print("input embedding:      [NOT FOUND]")

    if out_shape is not None:
        print(f"lm_head:              {out_key} shape={out_shape} ({out_file})")
    else:
        print("lm_head:              [NOT FOUND]")

    checks = []

    if in_shape is not None:
        checks.append(("embed_vs_len(tokenizer)", in_shape[0], tokenizer_total_tokens))
        checks.append(("embed_vs_config", in_shape[0], config_vocab_size))

    if out_shape is not None:
        checks.append(("lm_head_vs_len(tokenizer)", out_shape[0], tokenizer_total_tokens))
        checks.append(("lm_head_vs_config", out_shape[0], config_vocab_size))

    print("\n=== Checks ===")
    mismatch = False
    if not checks:
        print("No comparable embedding/lm_head weights found in safetensors.")
        return 3

    for name, left, right in checks:
        ok = left == right
        status = "OK" if ok else "MISMATCH"
        print(f"{name}: {left} vs {right} -> {status}")
        if not ok:
            mismatch = True

    if mismatch:
        print("\n[RESULT] MISMATCH DETECTED: tokenizer/config vocab and model weight vocab dimensions differ.")
        return 1

    print("\n[RESULT] OK: tokenizer/config vocab and model weight vocab dimensions are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
