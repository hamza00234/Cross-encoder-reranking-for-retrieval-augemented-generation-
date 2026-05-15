from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Scores (query, chunk) pairs using a cross-encoder for joint encoding.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["content"]) for c in candidates]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        if float(raw_scores.max()) == float(raw_scores.min()):
            norm_scores = np.ones_like(raw_scores, dtype=np.float32)
        else:
            norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

        reranked: list[dict] = []
        for c, raw, norm in zip(candidates, raw_scores, norm_scores):
            row = dict(c)
            row["raw_ce_score"] = float(raw)
            row["norm_ce_score"] = float(norm)
            reranked.append(row)
        reranked.sort(key=lambda x: x["norm_ce_score"], reverse=True)
        return reranked
