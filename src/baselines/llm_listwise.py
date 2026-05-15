from __future__ import annotations

from src.generation.generator import AnswerGenerator
from src.retrieval.dense_retriever import DenseRetriever


class LLMListwiseReranker:
    def __init__(self, retriever: DenseRetriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def _build_rerank_prompt(self, query: str, candidates: list[dict]) -> str:
        passages = "\n".join([f"[{i}] {c['content']}" for i, c in enumerate(candidates)])
        return (
            "You are a passage reranker. Given a query and a list of passages, return the\n"
            "indices of the passages in order from most to least relevant to the query.\n"
            "Return ONLY a comma-separated list of indices (0-based), nothing else.\n\n"
            f"Query: {query}\n\n"
            "Passages:\n"
            f"{passages}\n\n"
            "Ranked indices (most to least relevant):\n"
        )

    def _parse_indices(self, text: str, n: int) -> list[int]:
        out: list[int] = []
        for token in text.replace("\n", ",").split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < n and idx not in out:
                    out.append(idx)
        missing = [i for i in range(n) if i not in out]
        return out + missing

    def run(self, query: str, top_k: int = 5, *, skip_generation: bool = False) -> dict:
        candidates = self.retriever.retrieve(query)
        if skip_generation:
            reranked = sorted(candidates, key=lambda x: x["cosine_score"], reverse=True)[:top_k]
            return {"answer": "", "chunks": reranked}
        prompt = self._build_rerank_prompt(query, candidates)
        ranking_text = self.generator.chat_text(prompt)
        order = self._parse_indices(ranking_text, len(candidates))
        reranked = [candidates[i] for i in order][:top_k]
        answer = self.generator.generate(query=query, context_chunks=reranked)
        return {"answer": answer, "chunks": reranked}
