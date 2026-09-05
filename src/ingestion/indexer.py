from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np


class FAISSIndexer:
    """
    Builds and persists a FAISS index over chunk embeddings.
    Metadata is stored as JSON (not pickle) to avoid arbitrary-code execution on load.
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
            json_path = self._metadata_json_path()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to save FAISS index or metadata: {exc}") from exc

    def load(self) -> None:
        try:
            self.index = faiss.read_index(self.index_path)
            self.metadata = self._load_metadata()
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

    def _metadata_json_path(self) -> Path:
        path = Path(self.metadata_path)
        if path.suffix == ".pkl":
            return path.with_suffix(".json")
        return path

    def _load_metadata(self) -> list[dict]:
        json_path = self._metadata_json_path()
        if json_path.is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Metadata JSON must be a list, got {type(data).__name__}")
            return data
        pkl_path = Path(self.metadata_path)
        if pkl_path.suffix == ".pkl" and pkl_path.is_file():
            raise RuntimeError(
                f"Refusing to load pickle metadata at {pkl_path} (insecure). "
                "Rebuild the index so metadata is written as JSON."
            )
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
