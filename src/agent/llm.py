"""Thin OpenAI-compatible client for the LiteLLM proxy's agent-brain alias.

`enable_thinking: False` is required -- the underlying model (Qwen3.6-35B)
otherwise emits a long chain-of-thought preamble and never reaches a final
answer within any reasonable token budget (confirmed empirically: with
thinking left on, 700 tokens wasn't enough to finish reasoning about a
single sentiment question; with it off, the same question resolved
correctly in under a second).
"""
from __future__ import annotations

import httpx

from agent import config


def chat(messages: list[dict], max_tokens: int = 600, temperature: float = 0.0) -> str:
    with httpx.Client() as client:
        resp = client.post(
            f"{config.LITELLM_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.LITELLM_KEY}"},
            json={
                "model": config.BRAIN_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
