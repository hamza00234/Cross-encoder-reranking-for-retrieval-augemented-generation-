"""
Full Experiment Runner — Cross-Encoder RAG vs Baselines
========================================================
Runs all conditions defined in this repo (same set as ``experiments/run_experiment.py``),
computes retrieval metrics locally, optional generation + RAGAS,
writes per-condition JSON + a **comparison** JSON / tables (deltas vs several references).

Usage:
    python run_all_experiments.py \\
        --dataset hotpotqa \\
        --n_samples 100 \\
        --config configs/default.yaml \\
        --output_dir results/

    python run_all_experiments.py --dataset hotpotqa --n_samples 500 --retrieval_only
    python run_all_experiments.py --dataset both --n_samples 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines.large_pool_rag import LargePoolRAG
from src.baselines.llm_listwise import LLMListwiseReranker
from src.baselines.standard_rag import StandardRAG
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.pipeline import RAGPipeline, load_hotpotqa, load_musique

# Same condition IDs as experiments/run_experiment.py
CONDITION_IDS = [
    "standard_rag",
    "large_pool",
    "llm_listwise",
    "cross_encoder_pure",
    "cross_encoder_hybrid",
]

# Default comparison references (order: minimal baseline → proposed anchor)
DELTA_REFERENCE_ORDER = [
    ("standard_rag", "Standard RAG"),
    ("large_pool", "Large-pool cosine"),
    ("llm_listwise", "LLM listwise"),
    ("cross_encoder_pure", "CE pure (primary)"),
]

RETRIEVAL_KEYS = ("recall@5", "recall@10", "precision@5", "mrr")
GENERATION_KEYS = ("answer_faithfulness", "answer_relevance")


def load_conditions(config_path: str) -> list[dict[str, Any]]:
    """Return runnable condition rows (id, label, group, uses_generation). Hybrid label uses config α."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    alpha = float(cfg.get("hybrid", {}).get("alpha", 0.3))
    return [
        {
            "id": "standard_rag",
            "label": "Standard RAG (dense top-k from pool head)",
            "group": "baseline",
            "uses_generation": True,
        },
        {
            "id": "large_pool",
            "label": "Large-pool RAG (cosine over full pool, top-k)",
            "group": "baseline",
            "uses_generation": True,
        },
        {
            "id": "llm_listwise",
            "label": "LLM listwise reranking + answer",
            "group": "baseline",
            "uses_generation": True,
        },
        {
            "id": "cross_encoder_pure",
            "label": "Cross-encoder rerank (pure CE score)",
            "group": "proposed",
            "uses_generation": True,
        },
        {
            "id": "cross_encoder_hybrid",
            "label": f"Cross-encoder hybrid (α={alpha})",
            "group": "proposed",
            "uses_generation": True,
        },
    ]


def _runtime_config_for_dataset(base_config: str, dataset_name: str, output_dir: str) -> str:
    with open(base_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    idx_dir = Path(output_dir) / "indexes" / dataset_name
    idx_dir.mkdir(parents=True, exist_ok=True)
    cfg["ingestion"]["faiss_index_path"] = str(idx_dir / "corpus.index")
    cfg["ingestion"]["metadata_path"] = str(idx_dir / "metadata.json")
    out = Path(output_dir) / f".runtime_{dataset_name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(out)


def load_dataset(dataset_name: str, n_samples: int):
    print(f"\n📂 Loading {dataset_name} ({n_samples} samples)...")
    if dataset_name == "hotpotqa":
        corpus, queries = load_hotpotqa(n_samples=n_samples)
    elif dataset_name == "musique":
        corpus, queries = load_musique(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    print(f"   Corpus: {len(corpus)} passages | Queries: {len(queries)}")
    return corpus, queries


def build_or_load_index(pipeline: RAGPipeline, corpus, dataset_name: str, output_dir: str):
    index_flag = Path(output_dir) / f"{dataset_name}_index_built.flag"
    if index_flag.exists():
        print(f"📦 Loading existing {dataset_name} index...")
        pipeline.load_index()
    else:
        print(f"🔨 Building {dataset_name} index (this takes ~5-10 min)...")
        pipeline.build_index(corpus)
        index_flag.touch()
        print("   Index saved.")


def run_condition(
    pipeline: RAGPipeline,
    queries: list,
    condition_id: str,
    top_k: int,
    retrieval_only: bool,
):
    standard = StandardRAG(pipeline.retriever, pipeline.generator)
    large_pool = LargePoolRAG(pipeline.retriever, pipeline.generator)
    listwise = LLMListwiseReranker(pipeline.retriever, pipeline.generator)
    skip_gen = retrieval_only

    per_query: list[dict[str, Any]] = []
    failed = 0

    for q in tqdm(queries, desc=f"  {condition_id}", ncols=90):
        try:
            t0 = time.time()
            if condition_id == "standard_rag":
                out = standard.run(q["question"], top_k=top_k, skip_generation=skip_gen)
                result = {
                    "query": q["question"],
                    "answer": out["answer"],
                    "retrieved_chunks": out["chunks"],
                }
            elif condition_id == "large_pool":
                out = large_pool.run(q["question"], top_k=top_k, skip_generation=skip_gen)
                result = {
                    "query": q["question"],
                    "answer": out["answer"],
                    "retrieved_chunks": out["chunks"],
                }
            elif condition_id == "llm_listwise":
                out = listwise.run(q["question"], top_k=top_k, skip_generation=skip_gen)
                result = {
                    "query": q["question"],
                    "answer": out["answer"],
                    "retrieved_chunks": out["chunks"],
                }
            else:
                result = pipeline.run_query(
                    q["question"],
                    condition=condition_id,
                    top_k=top_k,
                    skip_generation=skip_gen,
                )
            latency_ms = (time.time() - t0) * 1000

            retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]

            per_query.append(
                {
                    "query_id": q.get("query_id", ""),
                    "question": q["question"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "retrieved_ids": retrieved_ids,
                    "answer": result["answer"] if not retrieval_only else "",
                    "context": [c["content"] for c in result["retrieved_chunks"]],
                    "ground_truth": q.get("answer", ""),
                    "latency_ms": latency_ms,
                }
            )

        except Exception as e:
            failed += 1
            print(f"\n  ⚠ Query failed ({q['question'][:40]}...): {e}")
            per_query.append(
                {
                    "query_id": q.get("query_id", ""),
                    "question": q["question"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "retrieved_ids": [],
                    "answer": "ERROR",
                    "context": [],
                    "ground_truth": q.get("answer", ""),
                    "latency_ms": 0,
                }
            )

    if failed:
        print(f"  ⚠ {failed}/{len(queries)} queries failed.")
    return per_query


def _p95_sorted(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = int(math.ceil(0.95 * n)) - 1
    return float(sorted_vals[min(max(idx, 0), n - 1)])


def compute_latency_stats(per_query: list[dict[str, Any]]) -> dict[str, float]:
    latencies = [float(r["latency_ms"]) for r in per_query if float(r["latency_ms"]) > 0]
    if not latencies:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "throughput_qps": 0.0}
    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p95_ms = _p95_sorted(latencies)
    throughput_qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "mean_ms": round(mean_ms, 1),
        "p95_ms": round(p95_ms, 1),
        "throughput_qps": round(throughput_qps, 3),
    }


def run_ragas(
    per_query: list[dict[str, Any]],
    generation: dict[str, Any] | None = None,
    ragas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from src.evaluation.ragas_eval import RAGASEvaluator

        evaluator = RAGASEvaluator(generation=generation, ragas=ragas)
        samples = [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["context"],
                "ground_truth": r["ground_truth"],
            }
            for r in per_query
            if r["answer"] not in ("", "ERROR")
        ]
        if not samples:
            return {"answer_faithfulness": None, "answer_relevance": None}
        return evaluator.evaluate(samples)
    except Exception as e:
        print(f"  ⚠ RAGAS skipped: {e}")
        return {"answer_faithfulness": None, "answer_relevance": None}


def float_or_none(v: Any) -> float | None:
    return round(float(v), 4) if v is not None else None


def _f(x: Any) -> float:
    if x is None:
        return 0.0
    return float(x)


def _retrieval_deltas(
    target: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in RETRIEVAL_KEYS:
        out[k] = round(_f(target.get(k)) - _f(reference.get(k)), 4)
    return out


def _generation_deltas(
    target: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in GENERATION_KEYS:
        tv, rv = target.get(k), reference.get(k)
        if tv is None or rv is None:
            out[k] = None
        else:
            out[k] = round(float(tv) - float(rv), 4)
    return out


def build_comparison_payload(all_results: dict[str, Any]) -> dict[str, Any]:
    """Pairwise deltas: for each reference R, each other condition C has metrics(C) - metrics(R)."""
    by_id = {cid: all_results[cid] for cid in all_results if cid in CONDITION_IDS}
    retrieval_block: dict[str, Any] = {}
    generation_block: dict[str, Any] = {}
    latency_block: dict[str, Any] = {}

    for ref_id, ref_label in DELTA_REFERENCE_ORDER:
        if ref_id not in by_id:
            continue
        ref_rm = by_id[ref_id]["retrieval_metrics"]
        ref_gm = by_id[ref_id]["generation_metrics"]
        ref_ls = by_id[ref_id]["latency_stats"]
        retrieval_block[ref_id] = {
            "label": ref_label,
            "vs_reference": {},
        }
        generation_block[ref_id] = {"label": ref_label, "vs_reference": {}}
        latency_block[ref_id] = {"label": ref_label, "vs_reference": {}}

        for cid, row in by_id.items():
            if cid == ref_id:
                continue
            retrieval_block[ref_id]["vs_reference"][cid] = {
                "label": row.get("display_name", cid),
                "delta_retrieval": _retrieval_deltas(row["retrieval_metrics"], ref_rm),
            }
            gen_d = _generation_deltas(row["generation_metrics"], ref_gm)
            if any(v is not None for v in gen_d.values()):
                generation_block[ref_id]["vs_reference"][cid] = {
                    "label": row.get("display_name", cid),
                    "delta_generation": gen_d,
                }
            latency_block[ref_id]["vs_reference"][cid] = {
                "label": row.get("display_name", cid),
                "mean_ms_delta": round(
                    float(row["latency_stats"]["mean_ms"])
                    - float(ref_ls["mean_ms"]),
                    2,
                ),
            }

    # Headline: best R@5 among completed conditions
    best_cid = None
    best_r5 = -1.0
    for cid in CONDITION_IDS:
        if cid not in by_id:
            continue
        r5 = _f(by_id[cid]["retrieval_metrics"].get("recall@5"))
        if r5 > best_r5:
            best_r5 = r5
            best_cid = cid

    labels = {cid: all_results[cid].get("display_name", cid) for cid in by_id}
    return {
        "condition_order": CONDITION_IDS,
        "condition_labels": labels,
        "retrieval_by_reference": retrieval_block,
        "generation_by_reference": generation_block,
        "latency_mean_ms_by_reference": latency_block,
        "best_recall_at_5": {"condition_id": best_cid, "value": round(best_r5, 4)}
        if best_cid
        else None,
    }


def print_results_table(all_results: dict[str, Any], condition_rows: list[dict[str, Any]]):
    col_w = 42
    print("\n")
    print("=" * 118)
    print("RESULTS — all setups (same top_k, same corpus index per dataset)")
    print("=" * 118)
    header = (
        f"{'Setup':<{col_w}} {'Grp':>8} "
        f"{'R@5':>6} {'R@10':>6} {'P@5':>6} {'MRR':>6} "
        f"{'Faith':>7} {'Relev':>7} "
        f"{'Lat(ms)':>9} {'QPS':>7}"
    )
    print(header)
    print("-" * 118)

    label_by_id = {c["id"]: c["label"] for c in condition_rows}
    group_by_id = {c["id"]: c["group"] for c in condition_rows}

    for cid in CONDITION_IDS:
        if cid not in all_results:
            continue
        r = all_results[cid]
        rm = r["retrieval_metrics"]
        gm = r["generation_metrics"]
        ls = r["latency_stats"]
        grp = group_by_id.get(cid, "?")[:8]

        def fmt(v: Any, decimals: int = 3) -> str:
            return f"{v:.{decimals}f}" if v is not None else "  N/A"

        disp = all_results[cid].get("display_name") or label_by_id.get(cid, cid)
        row = (
            f"{str(disp)[:col_w]:<{col_w}} {grp:>8} "
            f"{fmt(rm.get('recall@5')):>6} "
            f"{fmt(rm.get('recall@10')):>6} "
            f"{fmt(rm.get('precision@5')):>6} "
            f"{fmt(rm.get('mrr')):>6} "
            f"{fmt(gm.get('answer_faithfulness')):>7} "
            f"{fmt(gm.get('answer_relevance')):>7} "
            f"{fmt(ls.get('mean_ms'), 1):>9} "
            f"{fmt(ls.get('throughput_qps'), 3):>7}"
        )
        print(row)

    print("=" * 118)
    print("Grp: baseline vs proposed | Faith/Relev = RAGAS (N/A if skipped or retrieval-only)")
    print()


def print_delta_sections(all_results: dict[str, Any]):
    """Print Δ(retrieval) for each reference in DELTA_REFERENCE_ORDER."""
    for ref_id, ref_label in DELTA_REFERENCE_ORDER:
        if ref_id not in all_results:
            continue
        base = all_results[ref_id]["retrieval_metrics"]
        print(f"Δ RETRIEVAL vs {ref_label} (`{ref_id}`)")
        print("-" * 86)
        hdr = f"{'Other setup':<42} {'ΔR@5':>8} {'ΔR@10':>8} {'ΔP@5':>8} {'ΔMRR':>8}"
        print(hdr)
        print("-" * 86)

        for cid in CONDITION_IDS:
            if cid == ref_id or cid not in all_results:
                continue
            name = all_results[cid].get("display_name", cid)
            rm = all_results[cid]["retrieval_metrics"]

            def fmt_delta(v: float) -> str:
                sign = "+" if v >= 0 else ""
                return f"{sign}{v:.3f}"

            d = _retrieval_deltas(rm, base)
            print(
                f"{name[:42]:<42} "
                f"{fmt_delta(d['recall@5']):>8} "
                f"{fmt_delta(d['recall@10']):>8} "
                f"{fmt_delta(d['precision@5']):>8} "
                f"{fmt_delta(d['mrr']):>8}"
            )
        print()

    # Generation deltas vs standard_rag and vs cross_encoder_pure when possible
    for ref_id, ref_label in (("standard_rag", "Standard RAG"), ("cross_encoder_pure", "CE pure")):
        if ref_id not in all_results:
            continue
        ref_g = all_results[ref_id]["generation_metrics"]
        if all(ref_g.get(k) is None for k in GENERATION_KEYS):
            continue
        print(f"Δ GENERATION (RAGAS) vs {ref_label}")
        print("-" * 70)
        printed = False
        for cid in CONDITION_IDS:
            if cid == ref_id or cid not in all_results:
                continue
            dg = _generation_deltas(all_results[cid]["generation_metrics"], ref_g)
            if all(v is None for v in dg.values()):
                continue
            name = all_results[cid].get("display_name", cid)[:38]
            f1 = dg["answer_faithfulness"]
            f2 = dg["answer_relevance"]
            s1 = f"{f1:+.3f}" if f1 is not None else "  N/A"
            s2 = f"{f2:+.3f}" if f2 is not None else "  N/A"
            print(f"  {name:<38} ΔFaith {s1:>8}  ΔRelev {s2:>8}")
            printed = True
        if printed:
            print()


def print_ranking_line(all_results: dict[str, Any]):
    best_cid: str | None = None
    best_r5 = -1.0
    for cid in CONDITION_IDS:
        if cid not in all_results:
            continue
        r5 = _f(all_results[cid]["retrieval_metrics"].get("recall@5"))
        if r5 > best_r5:
            best_r5 = r5
            best_cid = cid
    if not best_cid:
        return
    name = all_results[best_cid].get("display_name", best_cid)
    print(f"Best retrieval (Recall@5): {name} — R@5 = {best_r5:.3f}\n")


def run_dataset(
    dataset_name: str,
    n_samples: int,
    config_path: str,
    output_dir: str,
    retrieval_only: bool,
    top_k: int,
    conditions_to_run: list[dict[str, Any]],
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f)
    generation_cfg = file_cfg.get("generation") or {}
    ragas_cfg = (file_cfg.get("evaluation") or {}).get("ragas") or {}

    runtime_cfg = _runtime_config_for_dataset(config_path, dataset_name, output_dir)
    pipeline = RAGPipeline(runtime_cfg)

    corpus, queries = load_dataset(dataset_name, n_samples)
    build_or_load_index(pipeline, corpus, dataset_name, output_dir)

    all_results: dict[str, Any] = {}
    summary_path = Path(output_dir) / f"{dataset_name}_summary.json"

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"\n▶ Resuming — already completed: {list(all_results.keys())}")

    for cond in conditions_to_run:
        cid = cond["id"]
        display_name = cond["label"]
        needs_gen = cond["uses_generation"]

        if cid in all_results:
            print(f"\n⏭  Skipping {display_name} (already in results)")
            continue

        print(f"\n{'─'*60}")
        print(f"▶ Running: {display_name}")
        print(f"{'─'*60}")

        per_query = run_condition(
            pipeline,
            queries,
            cid,
            top_k,
            retrieval_only=(retrieval_only or not needs_gen),
        )

        retrieval_input = [
            {"gold_chunk_ids": r["gold_chunk_ids"], "retrieved_ids": r["retrieved_ids"]}
            for r in per_query
        ]
        retrieval_metrics = compute_all_retrieval_metrics(retrieval_input)

        generation_metrics: dict[str, Any] = {
            "answer_faithfulness": None,
            "answer_relevance": None,
        }
        if not retrieval_only and needs_gen:
            print("  Computing RAGAS generation metrics...")
            generation_metrics = run_ragas(per_query, generation_cfg, ragas_cfg)

        latency_stats = compute_latency_stats(per_query)

        condition_result = {
            "condition": cid,
            "display_name": display_name,
            "group": cond["group"],
            "dataset": dataset_name,
            "n_samples": len(queries),
            "top_k": top_k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_metrics": {k: float_or_none(v) for k, v in retrieval_metrics.items()},
            "generation_metrics": {k: float_or_none(v) for k, v in generation_metrics.items()},
            "latency_stats": latency_stats,
            "per_query_results": per_query,
        }

        per_condition_path = Path(output_dir) / f"{dataset_name}_{cid}_top{top_k}.json"
        with open(per_condition_path, "w", encoding="utf-8") as f:
            json.dump(condition_result, f, indent=2, default=str)
        print(f"  💾 Saved → {per_condition_path}")

        rm = retrieval_metrics
        print(f"\n  📊 {display_name}")
        print(f"     Recall@5:    {rm['recall@5']:.3f}")
        print(f"     Recall@10:   {rm['recall@10']:.3f}")
        print(f"     Precision@5: {rm['precision@5']:.3f}")
        print(f"     MRR:         {rm['mrr']:.3f}")
        if generation_metrics.get("answer_faithfulness") is not None:
            print(f"     Faithfulness:{generation_metrics['answer_faithfulness']:.3f}")
            print(f"     Relevance:   {generation_metrics['answer_relevance']:.3f}")
        print(f"     Avg Latency: {latency_stats['mean_ms']:.0f}ms")

        all_results[cid] = condition_result
        with open(summary_path, "w", encoding="utf-8") as f:
            summary_data = {
                k: {kk: vv for kk, vv in v.items() if kk != "per_query_results"}
                for k, v in all_results.items()
            }
            json.dump(summary_data, f, indent=2, default=str)

    print_results_table(all_results, conditions_to_run)
    print_delta_sections(all_results)
    print_ranking_line(all_results)

    comparison = build_comparison_payload(all_results)
    cmp_path = Path(output_dir) / f"{dataset_name}_comparison.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"📎 Pairwise comparison JSON → {cmp_path}")

    print(f"✅ All results saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run all RAG experiment conditions")
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=["hotpotqa", "musique", "both"],
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of queries to evaluate per dataset",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--retrieval_only",
        action="store_true",
        help="Skip generation — compute retrieval metrics only (no API calls)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITION_IDS,
        choices=CONDITION_IDS,
        help="Subset of conditions (same IDs as experiments/run_experiment.py)",
    )
    args = parser.parse_args()

    all_rows = load_conditions(args.config)
    conditions_to_run = [c for c in all_rows if c["id"] in args.conditions]
    if not conditions_to_run:
        print("No valid conditions specified.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("RAG Experiment Runner")
    print(f"{'='*60}")
    print(f"Dataset:        {args.dataset}")
    print(f"Samples:        {args.n_samples}")
    print(f"Top-K:          {args.top_k}")
    print(f"Retrieval only: {args.retrieval_only}")
    print(f"Conditions:     {[c['id'] for c in conditions_to_run]}")
    print(f"Config:         {args.config}")
    print(f"Output dir:     {args.output_dir}")

    datasets = ["hotpotqa", "musique"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        rows = load_conditions(args.config)
        filtered = [c for c in rows if c["id"] in args.conditions]
        print(f"\n{'#'*60}")
        print(f"# DATASET: {ds.upper()}")
        print(f"{'#'*60}")
        run_dataset(
            dataset_name=ds,
            n_samples=args.n_samples,
            config_path=args.config,
            output_dir=args.output_dir,
            retrieval_only=args.retrieval_only,
            top_k=args.top_k,
            conditions_to_run=filtered,
        )


if __name__ == "__main__":
    main()
