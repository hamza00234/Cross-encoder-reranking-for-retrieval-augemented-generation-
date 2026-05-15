"""Resolve OpenRouter API key: environment first, then optional local dev file."""

from __future__ import annotations

import os


def resolve_openrouter_api_key(env_name: str = "OPENROUTER_API_KEY") -> str:
    key = (os.environ.get(env_name, "") or "").strip()
    if key:
        return key
    try:
        from src.openrouter_dev_key import OPENROUTER_API_KEY as _fallback
    except ImportError:
        return ""
    return str(_fallback or "").strip()
