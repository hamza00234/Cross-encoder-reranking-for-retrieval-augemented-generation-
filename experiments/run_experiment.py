from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from src.baselines.large_pool_rag import LargePoolRAG
from src.baselines.llm_listwise import LLMListwiseReranker
from src.baselines.standard_rag import StandardRAG
from src.evaluation.ragas_eval import RAGASEvaluator
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.pipeline import RAGPipeline, load_hotpotqa, load_musique


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "musique"], required=True)
    parser.add_argument(
        "--condition",
        choices=["standard_rag", "large_pool", "llm_listwise", "cross_encoder_pure", "cross_encoder_hybrid"],
        required=True,
    )
    parser.add_argument(
        "--generator",
        choices=["mistral", "llama3"],
        default="mistral",
        help="Only used when generation.backend is ollama (selects mistral vs llama3 tag).",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["hybrid"]["alpha"] = args.alpha
    gen_cfg = config["generation"]
    if str(gen_cfg.get("backend", "ollama")).lower() == "ollama":
        gen_cfg["model"] = (
            "mistral:7b-instruct" if args.generator == "mistral" else "llama3:8b-instruct"
        )

    temp_cfg_path = "configs/.runtime.yaml"
    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    pipeline = RAGPipeline(config_path=temp_cfg_path)
    if args.dataset == "hotpotqa":
        corpus, queries = load_hotpotqa(config["evaluation"]["datasets"]["hotpotqa"]["n_samples"])
    else:
        corpus, queries = load_musique(config["evaluation"]["datasets"]["musique"]["n_samples"])

    if os.path.exists(config["ingestion"]["faiss_index_path"]) and os.path.exists(config["ingestion"]["metadata_path"]):
        pipeline.load_index()
    else:
        pipeline.build_index(corpus)

    standard = StandardRAG(pipeline.retriever, pipeline.generator)
    large_pool = LargePoolRAG(pipeline.retriever, pipeline.generator)
    listwise = LLMListwiseReranker(pipeline.retriever, pipeline.generator)

    per_query_results = []
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        if args.condition == "standard_rag":
            out = standard.run(q["question"], top_k=args.top_k)
            selected = out["chunks"]
            answer = out["answer"]
        elif args.condition == "large_pool":
            out = large_pool.run(q["question"], top_k=args.top_k)
            selected = out["chunks"]
            answer = out["answer"]
        elif args.condition == "llm_listwise":
            out = listwise.run(q["question"], top_k=args.top_k)
            selected = out["chunks"]
            answer = out["answer"]
        else:
            out = pipeline.run_query(q["question"], condition=args.condition, top_k=args.top_k)
            selected = out["retrieved_chunks"]
            answer = out["answer"]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(float(latency_ms))

        per_query_results.append(
            {
                "query_id": q["query_id"],
                "query": q["question"],
                "gold_chunk_ids": q["gold_chunk_ids"],
                "retrieved_ids": [c["chunk_id"] for c in selected],
                "answer": answer,
                "contexts": [c["content"] for c in selected],
                "ground_truth": q["answer"],
                "latency_ms": float(latency_ms),
            }
        )

    retrieval_metrics = compute_all_retrieval_metrics(per_query_results)
    ragas_samples = [
        {
            "question": r["query"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in per_query_results
    ]
    ragas_opts = (config.get("evaluation") or {}).get("ragas") or {}
    generation_metrics = RAGASEvaluator(generation=gen_cfg, ragas=ragas_opts).evaluate(ragas_samples)

    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    throughput = float((1000.0 / mean_ms) if mean_ms > 0 else 0.0)
    output = {
        "config": config,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "per_query_results": per_query_results,
        "latency_stats": {"mean_ms": mean_ms, "p95_ms": p95, "throughput_qps": throughput},
    }

    os.makedirs(config["evaluation"]["results_dir"], exist_ok=True)
    if str(gen_cfg.get("backend", "ollama")).lower() == "openrouter":
        gen_tag = gen_cfg.get("model", "openrouter").replace("/", "_").replace(":", "_")
        if len(gen_tag) > 48:
            gen_tag = gen_tag[:48]
    else:
        gen_tag = args.generator
    out_path = f"results/{args.dataset}_{args.condition}_{gen_tag}_top{args.top_k}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=_json_default)
    except Exception as exc:
        raise RuntimeError(f"Failed to write results JSON: {exc}") from exc
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
