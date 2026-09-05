from __future__ import annotations

from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from src.env import env_value
from src.openrouter_key import resolve_openrouter_api_key

# Align with ingestion default; override via evaluation.ragas.embedding_model in YAML.
DEFAULT_RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _merged_generation(generation: dict[str, Any] | None, ragas: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(generation or {})
    ragas = ragas or {}
    if ragas.get("model"):
        out["ragas_model"] = ragas["model"]
    if ragas.get("embedding_model"):
        out["ragas_embedding_model"] = ragas["embedding_model"]
    return out


def _openrouter_judge_api_key(gen: dict[str, Any]) -> str:
    """Resolve an OpenAI-compatible key from ``.env`` / the process environment."""
    env_name = str(gen.get("api_key_env", "OPENROUTER_API_KEY"))
    return resolve_openrouter_api_key(env_name)


def _build_ragas_llm(gen: dict[str, Any], timeout_sec: float) -> Any:
    backend = str(gen.get("backend", "")).lower().strip()
    if not backend:
        if _openrouter_judge_api_key({}):
            backend = "openrouter"
        elif env_value("OPENAI_API_KEY"):
            backend = "openai"
        else:
            backend = "ollama"

    timeout = max(30.0, float(timeout_sec))

    if backend == "openrouter":
        api_key = _openrouter_judge_api_key(gen)
        if not api_key:
            raise RuntimeError(
                "RAGAS OpenRouter judge: set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your .env file."
            )
        from langchain_openai import ChatOpenAI

        base_url = str(gen.get("openrouter_base_url", "https://openrouter.ai/api/v1")).rstrip("/")
        model = str(gen.get("ragas_model") or gen.get("model") or "openai/gpt-4o-mini")
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": 0.0,
            "timeout": timeout,
            "max_retries": 2,
        }
        referer = gen.get("openrouter_http_referer")
        title = gen.get("openrouter_x_title")
        if referer or title:
            headers: dict[str, str] = {}
            if referer:
                headers["HTTP-Referer"] = str(referer)
            if title:
                headers["X-Title"] = str(title)
            kwargs["default_headers"] = headers
        return ChatOpenAI(**kwargs)

    if backend == "openai":
        api_key = env_value("OPENAI_API_KEY", "OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RAGAS needs OPENAI_API_KEY (or OPENROUTER_API_KEY) in your .env file "
                "when generation.backend is openai."
            )
        from langchain_openai import ChatOpenAI

        model = str(gen.get("ragas_model") or gen.get("model") or "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.0, timeout=timeout, max_retries=2)

    from langchain_community.chat_models import ChatOllama

    model = str(gen.get("model", "mistral:7b-instruct"))
    base_url = env_value("OLLAMA_BASE_URL") or str(gen.get("ollama_base_url", "http://localhost:11434"))
    return ChatOllama(model=model, base_url=base_url, temperature=0.0)


def _build_ragas_embeddings(gen: dict[str, Any]) -> HuggingfaceEmbeddings:
    emb_name = str(gen.get("ragas_embedding_model") or DEFAULT_RAGAS_EMBEDDING_MODEL)
    return HuggingfaceEmbeddings(model_name=emb_name)


class RAGASEvaluator:
    """
    RAGAS faithfulness + answer_relevancy.

    LLM: OpenRouter when ``generation.backend`` is openrouter (keys from ``.env``).
    Embeddings: local sentence-transformers (no API calls).
    """

    def __init__(
        self,
        generation: dict[str, Any] | None = None,
        ragas: dict[str, Any] | None = None,
    ) -> None:
        self._gen = _merged_generation(generation, ragas)
        self._llm: Any | None = None
        self._embeddings: HuggingfaceEmbeddings | None = None
        self._run_config: RunConfig | None = None

    def prepare_models(self) -> None:
        if self._llm is not None:
            return
        timeout_sec = float(self._gen.get("ragas_timeout_sec", self._gen.get("timeout_sec", 120.0)))
        self._run_config = RunConfig(timeout=int(max(30.0, timeout_sec)))
        self._llm = _build_ragas_llm(self._gen, timeout_sec=timeout_sec)
        self._embeddings = _build_ragas_embeddings(self._gen)

    def score_single_row(self, row: dict[str, Any]) -> tuple[float | None, float | None]:
        """Run faithfulness + answer_relevancy on one RAGAS row dict."""
        self.prepare_models()
        assert self._llm is not None and self._embeddings is not None and self._run_config is not None
        ds = Dataset.from_list([row])
        scores = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=self._llm,
            embeddings=self._embeddings,
            run_config=self._run_config,
            is_async=False,
            raise_exceptions=False,
        ).to_pandas()
        f_val = scores["faithfulness"].iloc[0]
        r_val = scores["answer_relevancy"].iloc[0]
        f_out = float(f_val) if f_val == f_val else None
        r_out = float(r_val) if r_val == r_val else None
        return f_out, r_out

    def evaluate(self, samples: list[dict]) -> dict:
        valid_rows = []
        for sample in samples:
            try:
                valid_rows.append(
                    {
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "contexts": sample["contexts"],
                        "ground_truth": sample["ground_truth"],
                    }
                )
            except Exception as exc:
                print(f"[RAGAS] Skipping malformed sample: {exc}")
        if not valid_rows:
            return {"answer_faithfulness": 0.0, "answer_relevance": 0.0}

        faith_scores: list[float] = []
        rel_scores: list[float] = []
        for row in valid_rows:
            try:
                f_out, r_out = self.score_single_row(row)
                if f_out is not None:
                    faith_scores.append(f_out)
                if r_out is not None:
                    rel_scores.append(r_out)
            except Exception as exc:
                print(f"[RAGAS] Failed sample; skipping: {exc}")
                continue
        if not faith_scores or not rel_scores:
            return {"answer_faithfulness": 0.0, "answer_relevance": 0.0}
        return {
            "answer_faithfulness": float(sum(faith_scores) / len(faith_scores)),
            "answer_relevance": float(sum(rel_scores) / len(rel_scores)),
        }
