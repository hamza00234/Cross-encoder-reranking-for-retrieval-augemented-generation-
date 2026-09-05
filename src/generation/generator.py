from __future__ import annotations

import time
from typing import Any, Protocol

import ollama
from openai import OpenAI

from src.env import env_value
from src.openrouter_key import resolve_openrouter_api_key


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    context_block = "\n\n".join(chunk["content"] for chunk in context_chunks)
    return (
        "SYSTEM:\n"
        "You are a helpful assistant. Answer the question using only the provided context.\n"
        "Do not use any external knowledge. If the context does not contain enough\n"
        'information to answer, say "I cannot answer based on the provided context."\n\n'
        "CONTEXT:\n"
        f"{context_block}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "ANSWER:\n"
    )


class AnswerGenerator(Protocol):
    """Shared interface for Ollama and OpenRouter backends."""

    def generate(self, query: str, context_chunks: list[dict]) -> str: ...

    def chat_text(self, prompt: str) -> str: ...


class OllamaGenerator:
    """
    Calls a locally-served Ollama LLM for answer generation.
    """

    def __init__(
        self,
        model: str = "mistral:7b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.client = ollama.Client(host=base_url, timeout=60.0)

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        prompt = build_rag_prompt(query, context_chunks)
        return self.chat_text(prompt)

    def chat_text(self, prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature},
                )
                return response["message"]["content"].strip()
            except Exception as exc:
                last_err = exc
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Ollama generation failed after retry: {exc}") from exc
        raise RuntimeError(f"Ollama generation failed: {last_err}")


class OpenRouterGenerator:
    """
    OpenAI-compatible chat completions against OpenRouter (https://openrouter.ai).
    API key: set ``OPENROUTER_API_KEY`` or the env name in config ``api_key_env``.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        fallback_model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_sec: float = 180.0,
        http_referer: str | None = None,
        x_title: str | None = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers: dict[str, str] = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
        client_kw: dict[str, Any] = {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "timeout": timeout_sec,
        }
        if headers:
            client_kw["default_headers"] = headers
        self.client = OpenAI(**client_kw)

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        return self.chat_text(build_rag_prompt(query, context_chunks))

    def _complete_one(self, model: str, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            return ""
        return content.strip()

    def chat_text(self, prompt: str) -> str:
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        last_err: BaseException | None = None
        for attempt in range(2):
            try:
                return self._complete_one(self.model, messages)
            except Exception as exc:
                last_err = exc
                if attempt == 0:
                    time.sleep(1)
                    continue
                break

        if self.fallback_model and last_err is not None:
            try:
                return self._complete_one(self.fallback_model, messages)
            except Exception as exc:
                raise RuntimeError(
                    f"OpenRouter fallback model {self.fallback_model!r} failed: {exc}"
                ) from exc

        raise RuntimeError(f"OpenRouter generation failed for model {self.model!r}: {last_err}") from last_err


def build_answer_generator(gen: dict[str, Any]) -> AnswerGenerator:
    backend = str(gen.get("backend", "ollama")).lower().strip()
    if backend == "openrouter":
        env_name = str(gen.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = resolve_openrouter_api_key(env_name)
        if not api_key:
            raise RuntimeError(
                f"generation.backend is openrouter but no API key was found. "
                f"Set {env_name} (or OPENROUTER_API_KEY / OPENAI_API_KEY) in your .env file."
            )
        return OpenRouterGenerator(
            model=gen["model"],
            base_url=gen["openrouter_base_url"],
            api_key=api_key,
            fallback_model=gen.get("fallback_model") or None,
            temperature=float(gen.get("temperature", 0.0)),
            max_tokens=int(gen.get("max_tokens", 2048)),
            timeout_sec=float(gen.get("timeout_sec", 180.0)),
            http_referer=gen.get("openrouter_http_referer"),
            x_title=gen.get("openrouter_x_title"),
        )
    return OllamaGenerator(
        model=gen["model"],
        base_url=env_value("OLLAMA_BASE_URL") or gen.get("ollama_base_url") or "http://localhost:11434",
        temperature=float(gen.get("temperature", 0.0)),
    )
