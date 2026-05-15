from __future__ import annotations

from src.generation.generator import AnswerGenerator
from src.retrieval.dense_retriever import DenseRetriever


class StandardRAG:
    def __init__(self, retriever: DenseRetriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5, *, skip_generation: bool = False) -> dict:
        candidates = self.retriever.retrieve(query)[:top_k]
        if skip_generation:
            return {"answer": "", "chunks": candidates}
        answer = self.generator.generate(query=query, context_chunks=candidates)
        return {"answer": answer, "chunks": candidates}
