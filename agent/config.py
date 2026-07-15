import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.6-35B-A3B-Instruct-FP8"

FINANCE_BASE_URL = "http://localhost:8001/v1"
FINANCE_MODEL = "nemotron-8b-finance"

_llm = ChatOpenAI(base_url=BASE_URL, api_key="none", model=MODEL,
                  temperature=0.2, max_tokens=550,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})

_llm_finance = ChatOpenAI(base_url=FINANCE_BASE_URL, api_key="none", model=FINANCE_MODEL,
                          temperature=0.1, max_tokens=400)

class _CleanLLM:
    """Strips <think> reasoning blocks so .content is only the real answer."""
    def __init__(self, inner):
        self._inner = inner
    def invoke(self, prompt):
        r = self._inner.invoke(prompt)
        r.content = re.sub(r"^.*?</think>\s*", "", r.content, flags=re.S).strip()
        return r

llm = _CleanLLM(_llm)
llm_finance = _CleanLLM(_llm_finance)
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")