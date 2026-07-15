"""Merge the trained LoRA adapter into the base model weights, producing a
plain HF checkpoint that can be served by vLLM exactly like the original
base model (no --enable-lora / --lora-modules serving complexity needed).

Loads the base model in bf16 (not 4-bit) for merging -- merging into a
quantized base would compound quantization error into the merged weights.

Usage: python training/merge_adapter.py
Output: training/output/nemotron-sentiment-merged/ (~16GB safetensors)
"""
from __future__ import annotations

import os
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "training", "lora_config.yaml")
MERGED_DIR = os.path.join(REPO_ROOT, "training", "output", "nemotron-sentiment-merged")


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_model_path = cfg["base_model_path"]
    adapter_path = os.path.join(REPO_ROOT, cfg["output_dir"])

    print(f"Loading base model in bf16 from {base_model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Attaching adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter into base weights ...")
    merged = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"Saving merged model to {MERGED_DIR} ...")
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"\nDone. Merged model ready to serve at {MERGED_DIR}")


if __name__ == "__main__":
    main()
