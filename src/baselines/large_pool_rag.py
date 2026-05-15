from __future__ import annotations

from src.generation.generator import AnswerGenerator
from src.retrieval.dense_retriever import DenseRetriever


class LargePoolRAG:
    def __init__(self, retriever: DenseRetriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5, *, skip_generation: bool = False) -> dict:
        candidates = self.retriever.retrieve(query)
        top_chunks = sorted(candidates, key=lambda x: x["cosine_score"], reverse=True)[:top_k]
        if skip_generation:
            return {"answer": "", "chunks": top_chunks}
        answer = self.generator.generate(query=query, context_chunks=top_chunks)
        return {"answer": answer, "chunks": top_chunks}
