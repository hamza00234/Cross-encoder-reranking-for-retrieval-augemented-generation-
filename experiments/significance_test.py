from __future__ import annotations

import json
import os

from scipy.stats import wilcoxon


def _mrr_per_query(rows: list[dict]) -> list[float]:
    vals = []
    for row in rows:
        gold = set(row["gold_chunk_ids"])
        ranked = row["retrieved_ids"]
        mrr = 0.0
        for idx, rid in enumerate(ranked, start=1):
            if rid in gold:
                mrr = 1.0 / idx
                break
        vals.append(float(mrr))
    return vals


def main() -> None:
    conditions = ["standard_rag", "large_pool", "llm_listwise", "cross_encoder_pure"]
    pairs = [
        ("standard_rag", "large_pool"),
        ("standard_rag", "llm_listwise"),
        ("standard_rag", "cross_encoder_pure"),
        ("large_pool", "llm_listwise"),
        ("large_pool", "cross_encoder_pure"),
        ("llm_listwise", "cross_encoder_pure"),
    ]
    alpha = 0.05 / 6

    loaded = {}
    for c in conditions:
        path = f"results/hotpotqa_{c}_mistral_top5.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            loaded[c] = json.load(f)

    results = {"bonferroni_alpha": alpha, "comparisons": []}
    for a, b in pairs:
        xa = _mrr_per_query(loaded[a]["per_query_results"])
        xb = _mrr_per_query(loaded[b]["per_query_results"])
        stat, p_value = wilcoxon(xa, xb, alternative="two-sided")
        results["comparisons"].append(
            {
                "a": a,
                "b": b,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "reject_null": bool(p_value < alpha),
            }
        )

    with open("results/significance_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results/significance_tests.json")


if __name__ == "__main__":
    main()
