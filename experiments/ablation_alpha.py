from __future__ import annotations

import json
import os

from src.evaluation.ragas_eval import RAGASEvaluator
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.pipeline import RAGPipeline, load_hotpotqa


def main() -> None:
    pipeline = RAGPipeline("configs/default.yaml")
    corpus, queries = load_hotpotqa(100)
    if os.path.exists(pipeline.config["ingestion"]["faiss_index_path"]) and os.path.exists(
        pipeline.config["ingestion"]["metadata_path"]
    ):
        pipeline.load_index()
    else:
        pipeline.build_index(corpus)

    sweep = []
    for step in range(11):
        alpha = round(step * 0.1, 1)
        pipeline.hybrid.alpha = alpha
        per_q = []
        ragas_samples = []
        for q in queries:
            out = pipeline.run_query(q["question"], condition="cross_encoder_hybrid", top_k=5)
            retrieved = [c["chunk_id"] for c in out["retrieved_chunks"]]
            per_q.append({"gold_chunk_ids": q["gold_chunk_ids"], "retrieved_ids": retrieved})
            ragas_samples.append(
                {
                    "question": q["question"],
                    "answer": out["answer"],
                    "contexts": [c["content"] for c in out["retrieved_chunks"]],
                    "ground_truth": q["answer"],
                }
            )
        r = compute_all_retrieval_metrics(per_q)
        gen = pipeline.config.get("generation") or {}
        ragas_opts = (pipeline.config.get("evaluation") or {}).get("ragas") or {}
        g = RAGASEvaluator(generation=gen, ragas=ragas_opts).evaluate(ragas_samples)
        row = {
            "alpha": alpha,
            "recall@5": float(r["recall@5"]),
            "precision@5": float(r["precision@5"]),
            "answer_faithfulness": float(g["answer_faithfulness"]),
        }
        sweep.append(row)
        print(row)

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_alpha.json", "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)


if __name__ == "__main__":
    main()
