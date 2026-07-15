"""Environment-driven agent configuration. Keep endpoints/credentials out of
source -- everything here is read from the environment with sane local
defaults for this event's Atom box."""
from __future__ import annotations

import os

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "EMPTY")

BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "agent-brain")
DOMAIN_FT_MODEL = os.environ.get("DOMAIN_FT_MODEL", "domain-ft")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "")

MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "12"))
