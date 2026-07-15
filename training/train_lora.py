"""LoRA fine-tuning of Llama-3.1-Nemotron-Nano-8B-v1 on the AFR/RBA
sentiment+direction dataset, using Hugging Face transformers + peft + trl.

This is the "Route B" fallback documented in training/MODEL_CARD.md: the
event's reference path (NeMo Customizer microservice inside
nvcr.io/nvidia/nemo:25.09) wasn't reachable in this environment (no
Customizer service running, no docker permission to pull the container), so
we fine-tune directly against the local base-model weights with the same
reference hyperparameters (rank 32, seq_len 512, lr 5e-5, 100 steps).

Loads the base model in 4-bit (QLoRA) because this box's unified memory is
shared with two live inference servers (agent-brain, domain-ft) -- see
logs/gpu_memory_notes.md for the measured headroom this was sized against.

Usage:
  python training/train_lora.py --config training/lora_config.yaml
  python training/train_lora.py --config training/lora_config.yaml --dry-run   # loads+tokenizes only, no training
"""
from __future__ import annotations

import argparse
import json
import os
import time

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _verify_masking(tokenizer, train_dataset, n_samples: int = 10) -> None:
    """Guard against the exact bug this pipeline hit once already: plain-string
    prompt/completion caused TRL to diff raw-text tokenizations at a ragged,
    no-separator boundary, which silently mismatched and masked the real
    completion out of the loss (the model then just learned to emit EOS
    immediately -- confirmed empirically: 0/58 non-empty generations on the
    held-out val set). With conversational messages, `add_generation_prompt`
    gives a clean, unambiguous boundary. Verify that here, every time, before
    spending GPU time training on it -- abort loudly instead of training on a
    silently-broken mask again.
    """
    n = min(n_samples, len(train_dataset))
    failures = []
    for i in range(n):
        ex = train_dataset[i]
        prompt_msgs, completion_msgs = ex["prompt"], ex["completion"]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, return_dict=True, add_generation_prompt=True
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(prompt_msgs + completion_msgs, tokenize=True, return_dict=True)[
            "input_ids"
        ]
        prefix_ok = full_ids[: len(prompt_ids)] == prompt_ids
        completion_ids = full_ids[len(prompt_ids) :]
        non_empty = len(completion_ids) > 0
        decoded_completion = tokenizer.decode(completion_ids)
        looks_right = "Sentiment:" in decoded_completion and "Likely direction:" in decoded_completion
        if not (prefix_ok and non_empty and looks_right):
            failures.append(
                {
                    "index": i,
                    "prefix_ok": prefix_ok,
                    "non_empty": non_empty,
                    "decoded_completion": decoded_completion,
                }
            )

    if failures:
        print(f"\n[MASKING VERIFICATION FAILED] {len(failures)}/{n} samples failed:")
        for f in failures[:3]:
            print(f"  {f}")
        raise SystemExit(
            "Aborting before spending GPU time: prompt/completion boundary is not clean. "
            "Do not proceed with training until this is fixed."
        )
    print(f"[masking verification] {n}/{n} samples: clean prefix match, non-empty completion, "
          f"'Sentiment:'/'Likely direction:' both present in the isolated completion tokens. OK.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "lora_config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="Load model/data and report shapes/memory only.")
    ap.add_argument(
        "--skip-model",
        action="store_true",
        help="Only load tokenizer + dataset and report token lengths. Allocates no GPU memory.",
    )
    ap.add_argument(
        "--verify-masking-only",
        action="store_true",
        help="Alias for --skip-model, emphasizing this only runs the masking-boundary check.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    base_model_path = cfg["base_model_path"]
    seq_len = cfg["sequence_length"]

    print(f"[1/5] Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.skip_model or args.verify_masking_only:
        data_files = {
            "train": os.path.join(repo_root, cfg["train_file"]),
            "validation": os.path.join(repo_root, cfg["val_file"]),
        }
        ds = load_dataset("json", data_files=data_files)
        _verify_masking(tokenizer, ds["train"], n_samples=10)

        lengths = [
            len(
                tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=True, return_dict=True)[
                    "input_ids"
                ]
            )
            for ex in ds["train"]
        ]
        lengths.sort()
        n = len(lengths)
        print(f"\n[skip-model] train examples: {n}, val examples: {len(ds['validation'])}")
        print(f"[skip-model] token length: min={lengths[0]} p50={lengths[n // 2]} "
              f"p95={lengths[int(n * 0.95)]} max={lengths[-1]} (budget: {seq_len})")
        over_budget = sum(1 for l in lengths if l > seq_len)
        print(f"[skip-model] {over_budget}/{n} examples exceed the {seq_len}-token budget (will be truncated)")
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[2/5] Loading base model (4bit={cfg['load_in_4bit']}) ...")
    t0 = time.time()
    quant_config = None
    if cfg["load_in_4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
        device_map="auto",
    )
    print(f"    loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"    GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    print("[3/5] Attaching LoRA adapter")
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[4/5] Loading dataset")
    data_files = {
        "train": os.path.join(repo_root, cfg["train_file"]),
        "validation": os.path.join(repo_root, cfg["val_file"]),
    }
    ds = load_dataset("json", data_files=data_files)

    # Conversational prompt/completion (list-of-messages) -- do NOT flatten to
    # a "text" field. TRL's SFTTrainer natively detects this format and uses
    # add_generation_prompt to find a clean, unambiguous completion boundary.
    # See _verify_masking's docstring for why this matters.
    _verify_masking(tokenizer, ds["train"], n_samples=10)

    if args.dry_run:
        print(f"\n[dry-run] train examples: {len(ds['train'])}, val examples: {len(ds['validation'])}")
        print("[dry-run] OK -- exiting before training.")
        return

    from trl import SFTConfig, SFTTrainer

    output_dir = os.path.join(repo_root, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=seq_len,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        logging_steps=5,
        save_steps=cfg["checkpoint_every"],
        eval_strategy="steps",
        eval_steps=cfg["checkpoint_every"],
        save_total_limit=6,
        seed=cfg["seed"],
        report_to=[],
    )

    print("[5/5] Training")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    log_path = os.path.join(repo_root, "logs", "train_lora_metrics.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        for cb_state in trainer.state.log_history:
            log_f.write(json.dumps(cb_state) + "\n")

    train_result = trainer.train()

    with open(log_path, "w", encoding="utf-8") as log_f:
        for entry in trainer.state.log_history:
            log_f.write(json.dumps(entry) + "\n")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"\nDone. Adapter + logs written to {output_dir}")
    print(f"Metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
