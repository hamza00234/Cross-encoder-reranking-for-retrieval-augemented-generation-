"""Resolve OpenRouter / OpenAI-compatible API keys from the environment only."""

from __future__ import annotations

from src.env import env_value, load_project_env


def resolve_openrouter_api_key(env_name: str = "OPENROUTER_API_KEY") -> str:
    load_project_env()
    names = [env_name, "OPENROUTER_API_KEY", "OPENAI_API_KEY"]
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return env_value(*seen)
