"""
src/llm/llm_client.py
──────────────────────
Central LLM client — uses Groq (free, fast, llama3 model).
"""

import os
from pathlib import Path
from src.utils import get_logger

logger = get_logger("llm.client")

DEFAULT_MODEL       = "llama-3.1-8b-instant"   # free on Groq, very fast
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 600


def _load_env():
    """Load env.example manually."""
    project_root = Path(__file__).resolve().parents[2]
    for fname in ["env.example", ".env"]:
        env_path = project_root / fname
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())
            return

_load_env()


def chat(
    system_prompt: str,
    user_prompt:   str,
    model:         str   = DEFAULT_MODEL,
    temperature:   float = DEFAULT_TEMPERATURE,
    max_tokens:    int   = DEFAULT_MAX_TOKENS,
) -> str:
    """Single call — returns Groq's reply as plain string."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your-groq-key-here":
        raise EnvironmentError(
            "\n\n❌  Groq API key not set!\n"
            "  1. Go to https://console.groq.com\n"
            "  2. Sign up free → API Keys → Create API Key\n"
            "  3. Open hotel-rag-system/env.example\n"
            "  4. Add line: GROQ_API_KEY=your-key-here\n"
            "  5. Save and run again\n"
        )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
