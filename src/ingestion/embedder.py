from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ChunkEmbedder:
    """
    Encodes chunks into dense vectors using sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_chunks(self, chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
        texts = [c["content"] for c in chunks]
        all_vectors: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding chunks"):
            batch = texts[i : i + self.batch_size]
            vecs = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_vectors.append(vecs.astype(np.float32))
        embeddings = (
            np.concatenate(all_vectors, axis=0)
            if all_vectors
            else np.zeros((0, 768), dtype=np.float32)
        )
        return embeddings, chunks

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float32)
