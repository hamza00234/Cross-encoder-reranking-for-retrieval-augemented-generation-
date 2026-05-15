from __future__ import annotations

from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.indexer import FAISSIndexer


class DenseRetriever:
    """
    Wraps FAISSIndexer and ChunkEmbedder for query-time dense retrieval.
    """

    def __init__(
        self,
        embedder: ChunkEmbedder,
        indexer: FAISSIndexer,
        candidate_pool_size: int = 25,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer
        self.candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str) -> list[dict]:
        q_emb = self.embedder.embed_query(query)
        return self.indexer.search(q_emb, top_k=self.candidate_pool_size)
