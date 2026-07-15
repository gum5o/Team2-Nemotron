"""Compare the base Nemotron-Nano-8B model against the LoRA fine-tuned
adapter, on:
  1. The held-out validation split (58 examples, never trained on).
  2. The 3 real calibration articles referenced by public_questions.jsonl
     (MHQ058/067/080) with their official expected_fact strings, so we can
     check directly against what the actual grader checks for.

Loads the base model once (4-bit, matching how it was trained) and swaps the
LoRA adapter on/off via disable_adapter(), so this never touches the live
vLLM serving stack on :8000/:8001.

Usage: python training/eval_adapter.py
Output: logs/eval_adapter_report.json
"""
from __future__ import annotations

import json
import os
import re
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "training", "lora_config.yaml")
VAL_PATH = os.path.join(REPO_ROOT, "training", "data", "lora", "val.jsonl")
PUBLIC_Q_PATH = (
    "/home/cognitivo/Desktop/Cognitivo_Training/Mock_Hackathon_Participant_Package/public_questions.jsonl"
)
REPORT_PATH = os.path.join(REPO_ROOT, "logs", "eval_adapter_report.json")

CALIBRATION_ARTICLE_IDS = {
    "MHQ058": 184114,  # "Travel stocks take off on vaccine rollout", 2021-02-23, rate 0.10%
    "MHQ067": 213882,  # "Why investors don't believe the RBA on interest rates", 2021-11-25, rate 0.10%
    "MHQ080": 175725,  # "Energy stocks shine as vaccines fuel oil rally", 2020-11-28, rate 0.10%
}

LINE_RE = re.compile(
    r"Sentiment:\s*(?P<sentiment>[^\n]+)\s*\n\s*Likely direction:\s*(?P<direction>[^\n]+)", re.IGNORECASE
)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 70) -> str:
    """`prompt_text` is the raw user-turn content; wrapped in the model's own
    chat template with add_generation_prompt=True, matching exactly how the
    conversational training data (and production inference via vLLM's
    /v1/chat/completions) presents a prompt to the model."""
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
    return text.strip()


def parse_answer(text: str) -> dict:
    m = LINE_RE.search(text)
    if not m:
        return {"sentiment": None, "direction": None, "raw": text}
    return {
        "sentiment": m.group("sentiment").strip().rstrip("."),
        "direction": m.group("direction").strip().rstrip("."),
        "raw": text,
    }


def sentiment_bucket(s: str | None) -> str:
    if not s:
        return "unparsed"
    s = s.lower()
    if "mixed" in s:
        return "mixed"
    if "positive" in s:
        return "positive"
    if "negative" in s:
        return "negative"
    return "other"


def main() -> None:
    cfg = load_config()
    base_model_path = cfg["base_model_path"]
    adapter_path = os.path.join(REPO_ROOT, cfg["output_dir"])

    print(f"Loading tokenizer + base model from {base_model_path} (4-bit) ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, quantization_config=quant_config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"  base model loaded in {time.time() - t0:.1f}s")

    print(f"Attaching LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # ---------- 1. Held-out validation set ----------
    val_rows = []
    with open(VAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_rows.append(json.loads(line))

    print(f"\nEvaluating on {len(val_rows)} held-out validation examples (base vs fine-tuned) ...")
    val_results = []
    agree_with_label = {"base": 0, "finetuned": 0}
    for i, row in enumerate(val_rows):
        prompt_text = row["prompt"][0]["content"]
        completion_text = row["completion"][0]["content"]
        expected = parse_answer(completion_text)
        expected_bucket = sentiment_bucket(expected["sentiment"])

        with model.disable_adapter():
            base_raw = generate(model, tokenizer, prompt_text)
        base_parsed = parse_answer(base_raw)

        ft_raw = generate(model, tokenizer, prompt_text)
        ft_parsed = parse_answer(ft_raw)

        base_match = sentiment_bucket(base_parsed["sentiment"]) == expected_bucket
        ft_match = sentiment_bucket(ft_parsed["sentiment"]) == expected_bucket
        agree_with_label["base"] += int(base_match)
        agree_with_label["finetuned"] += int(ft_match)

        val_results.append(
            {
                "headline": prompt_text.split("\n")[0],
                "expected_bucket": expected_bucket,
                "base_output": base_parsed,
                "finetuned_output": ft_parsed,
                "base_matches_teacher_bucket": base_match,
                "finetuned_matches_teacher_bucket": ft_match,
            }
        )
        if (i + 1) % 10 == 0 or i == len(val_rows) - 1:
            print(f"  [{i + 1}/{len(val_rows)}] base_agree={agree_with_label['base']} ft_agree={agree_with_label['finetuned']}")

    # ---------- 2. Real calibration articles (official expected_fact strings) ----------
    import sys

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from tools import afr_search, rba_tool  # noqa: E402
    from tools.sentiment_tool import STUDENT_PROMPT_TEMPLATE  # noqa: E402

    public_questions = {}
    with open(PUBLIC_Q_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            public_questions[d["id"]] = d

    rba = rba_tool.get_dataset()
    calibration_results = []
    for qid, article_id in CALIBRATION_ARTICLE_IDS.items():
        art = afr_search.get_article(article_id)
        pubdate_iso = f"{art['pubdate'][0:4]}-{art['pubdate'][4:6]}-{art['pubdate'][6:8]}"
        rate = rba.rate_on_date(pubdate_iso)["target_pct"]
        prompt = STUDENT_PROMPT_TEMPLATE.format(headline=art["headline"], rate=rate, text=art["text"][:1200])

        with model.disable_adapter():
            base_raw = generate(model, tokenizer, prompt)
        ft_raw = generate(model, tokenizer, prompt)

        expected_facts = [c["expected_fact"] for c in public_questions[qid]["grading"]["components"]]
        calibration_results.append(
            {
                "id": qid,
                "headline": art["headline"],
                "rba_target_pct": rate,
                "expected_facts": expected_facts,
                "base_output": parse_answer(base_raw),
                "finetuned_output": parse_answer(ft_raw),
            }
        )
        print(f"\n### {qid}: {art['headline']}")
        print(f"  expected: {expected_facts}")
        print(f"  base:       {parse_answer(base_raw)}")
        print(f"  fine-tuned: {parse_answer(ft_raw)}")

    n = len(val_results)
    report = {
        "val_set_size": n,
        "val_sentiment_bucket_agreement_with_teacher_label": {
            "base_model": f"{agree_with_label['base']}/{n}",
            "fine_tuned_model": f"{agree_with_label['finetuned']}/{n}",
        },
        "val_results": val_results,
        "calibration_article_results": calibration_results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Held-out val set ({n} examples) sentiment-bucket agreement with teacher label:")
    print(f"  base model:       {agree_with_label['base']}/{n}")
    print(f"  fine-tuned model: {agree_with_label['finetuned']}/{n}")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
