from __future__ import annotations


def recall_at_k(results: list[dict], k: int) -> float:
    scores = []
    for row in results:
        gold = set(row.get("gold_chunk_ids", []))
        if not gold:
            continue
        retrieved = set(row.get("retrieved_ids", [])[:k])
        scores.append(len(retrieved.intersection(gold)) / len(gold))
    return float(sum(scores) / len(scores)) if scores else 0.0


def precision_at_k(results: list[dict], k: int) -> float:
    scores = []
    for row in results:
        gold = set(row.get("gold_chunk_ids", []))
        retrieved = set(row.get("retrieved_ids", [])[:k])
        scores.append(len(retrieved.intersection(gold)) / k)
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_reciprocal_rank(results: list[dict]) -> float:
    rr = []
    for row in results:
        gold = set(row.get("gold_chunk_ids", []))
        ranked = row.get("retrieved_ids", [])
        val = 0.0
        for idx, chunk_id in enumerate(ranked, start=1):
            if chunk_id in gold:
                val = 1.0 / idx
                break
        rr.append(val)
    return float(sum(rr) / len(rr)) if rr else 0.0


def compute_all_retrieval_metrics(results: list[dict]) -> dict:
    return {
        "recall@5": recall_at_k(results, 5),
        "recall@10": recall_at_k(results, 10),
        "precision@5": precision_at_k(results, 5),
        "mrr": mean_reciprocal_rank(results),
    }
