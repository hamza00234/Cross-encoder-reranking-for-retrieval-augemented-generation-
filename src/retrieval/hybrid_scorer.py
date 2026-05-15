from __future__ import annotations

import numpy as np


class HybridScorer:
    """
    Combines normalized cosine similarity with normalized cross-encoder score.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def score(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        cos = np.asarray([c["cosine_score"] for c in candidates], dtype=np.float32)
        if float(cos.max()) == float(cos.min()):
            norm_cos = np.ones_like(cos, dtype=np.float32)
        else:
            norm_cos = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)

        scored: list[dict] = []
        for c, nc in zip(candidates, norm_cos):
            row = dict(c)
            row["norm_cosine_score"] = float(nc)
            row["final_score"] = float(self.alpha * nc + (1 - self.alpha) * row["norm_ce_score"])
            scored.append(row)
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored

    def pure_ce_score(self, candidates: list[dict]) -> list[dict]:
        scored = [dict(c) for c in candidates]
        scored.sort(key=lambda x: x["norm_ce_score"], reverse=True)
        return scored
