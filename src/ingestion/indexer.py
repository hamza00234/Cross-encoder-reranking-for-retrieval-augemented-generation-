from __future__ import annotations

import os
import pickle

import faiss
import numpy as np


class FAISSIndexer:
    """
    Builds and persists a FAISS index over chunk embeddings.
    """

    def __init__(
        self,
        index_type: str,
        embedding_dim: int,
        index_path: str,
        metadata_path: str,
    ) -> None:
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(embeddings)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
        self.index = index
        self.metadata = metadata

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            if self.index is None:
                raise ValueError("Index is not built")
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as exc:
            raise RuntimeError(f"Failed to save FAISS index or metadata: {exc}") from exc

    def load(self) -> None:
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load FAISS index or metadata: {exc}") from exc

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if self.index is None:
            raise ValueError("Index is not loaded/built")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = dict(self.metadata[idx])
            row["cosine_score"] = float(score)
            results.append(row)
        return results
