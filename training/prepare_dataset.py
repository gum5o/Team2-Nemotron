"""Stage 3 of the fine-tuning pipeline: turn teacher-labeled examples into the
conversational {prompt, completion} shape the student model (Nemotron
`domain-ft`) will actually see at inference time via the agent's sentiment
tool, then split into train/val.

The teacher-labeling prompt (label_with_teacher.py) included a long reasoning
guidance essay to help the *teacher* reason well. The student does not need
that scaffolding at inference time -- it needs to learn the direct mapping
from (article, rate) -> (sentiment, direction), so the training prompt here
matches src/tools/sentiment_tool.py's production prompt shape exactly.

IMPORTANT: prompt/completion are lists of chat messages (not plain strings).
An earlier version used plain strings, which made TRL's SFTTrainer fall back
to raw-text concatenation with no separator between prompt and completion --
a tokenization boundary ambiguity that silently masked out the real
completion tokens from the loss (confirmed via the repeated "Mismatch
between tokenized prompt..." warning during training) and produced a model
that just learned to emit EOS immediately. The conversational format below
makes TRL take its proper chat-templated path (`add_generation_prompt=True`
for the prompt, then a clean diff against prompt+completion), which both
fixes the masking and matches how the model is actually invoked in
production (vLLM applies the same chat template via /v1/chat/completions).
Verified directly against this tokenizer before relying on it -- see the
boundary check in training/README.md's "Training bug and fix" section.

Run: python training/prepare_dataset.py
Output: training/data/lora/train.jsonl, training/data/lora/val.jsonl
"""
from __future__ import annotations

import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LABELED_PATH = os.path.join(DATA_DIR, "labeled.jsonl")
OUT_DIR = os.path.join(DATA_DIR, "lora")

MIN_TEXT_CHARS = 200
MIN_ALPHA_RATIO = 0.75  # filters out market-data tables (tickers/numbers), keeps prose articles
STUDENT_ARTICLE_CHARS = 1200
VALID_SENTIMENT_PREFIXES = ("positive", "negative", "mixed")


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    return alpha / len(text)

# Must match sentiment_tool.STUDENT_PROMPT_TEMPLATE exactly so train == inference shape.
STUDENT_PROMPT_TEMPLATE = """Article headline: {headline}
Applicable RBA cash rate target on publication date: {rate}%

Article text:
{text}

Classify this article's financial-market sentiment as positive, negative, or mixed \
(optionally qualified, e.g. "mixed with a negative bias"), and state the likely \
direction for the relevant shares or sector. Answer in exactly two short lines:
Sentiment: <label>
Likely direction: <short phrase>"""


def build_example(row: dict) -> dict:
    prompt = STUDENT_PROMPT_TEMPLATE.format(
        headline=row["headline"],
        rate=row["rba_target_pct"],
        text=row["text"][:STUDENT_ARTICLE_CHARS],
    )
    completion = f"Sentiment: {row['sentiment']}\nLikely direction: {row['direction']}"
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def main(seed: int = 0, val_frac: float = 0.1) -> None:
    rows = []
    with open(LABELED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row["headline"].strip():
                continue
            if len(row["text"]) < MIN_TEXT_CHARS:
                continue
            if _alpha_ratio(row["text"]) < MIN_ALPHA_RATIO:
                continue
            if not row["sentiment"].strip().lower().startswith(VALID_SENTIMENT_PREFIXES):
                continue
            rows.append(row)

    total_labeled = sum(1 for _ in open(LABELED_PATH, encoding="utf-8"))

    random.seed(seed)
    random.shuffle(rows)
    cut = int(len(rows) * (1 - val_frac))
    train_rows, val_rows = rows[:cut], rows[cut:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, chunk in [("train", train_rows), ("val", val_rows)]:
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(build_example(row)) + "\n")
        print(f"wrote {path} ({len(chunk)} examples)")

    print(f"\nFiltered out {total_labeled - len(rows)} of {total_labeled} labeled examples "
          f"(short/tabular text or off-spec sentiment label); kept {len(rows)}.")


if __name__ == "__main__":
    main()
