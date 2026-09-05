"""Load project secrets from ``.env`` and resolve paths inside the repo."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_LOADED = False


def load_project_env(*, override: bool = False) -> Path:
    """Load ``.env`` from the repository root. Safe to call more than once."""
    global _ENV_LOADED
    env_path = PROJECT_ROOT / ".env"
    if not _ENV_LOADED or override:
        load_dotenv(env_path, override=override)
        _ENV_LOADED = True
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return env_path


def env_value(*names: str) -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    """
    Resolve a config path against the project root.

    Relative paths stay inside the repo. Absolute paths are allowed as-is.
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()
    resolved = (PROJECT_ROOT / raw).resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path {path!r} resolves outside the project directory")
    return resolved
