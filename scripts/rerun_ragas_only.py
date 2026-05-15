#!/usr/bin/env python3
"""
Recompute RAGAS (faithfulness + answer_relevancy) from saved per-query JSON only.

Does not run retrieval or answer generation. Requires OPENAI_API_KEY set to your
OpenRouter key (OpenAI-compatible client + https://openrouter.ai/api/v1).

Usage:
  export OPENAI_API_KEY=sk-or-v1-...
  python scripts/rerun_ragas_only.py --conditions standard_rag large_pool llm_listwise --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all_experiments import (  # noqa: E402
    CONDITION_IDS,
    build_comparison_payload,
    load_conditions,
    print_results_table,
)
from src.evaluation.ragas_eval import RAGASEvaluator  # noqa: E402


def _require_openai_key_for_openrouter() -> None:
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "Set it to your OpenRouter API key (OpenRouter accepts OpenAI-compatible clients):\n"
            "  export OPENAI_API_KEY='sk-or-v1-...'\n"
            "Then re-run this script."
        )


def _find_condition_path(output_dir: Path, dataset: str, condition_id: str) -> Path:
    matches = sorted(output_dir.glob(f"{dataset}_{condition_id}_top*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No results file matching {dataset}_{condition_id}_top*.json under {output_dir}"
        )
    return matches[0]


def _load_checkpoint(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw = data.get("scores_by_query_id") or {}
    out: dict[str, dict[str, float]] = {}
    for qid, v in raw.items():
        if not isinstance(v, dict):
            continue
        f_, r_ = v.get("faithfulness"), v.get("answer_relevancy")
        if f_ is None and r_ is None:
            continue
        entry: dict[str, float] = {}
        if isinstance(f_, (int, float)) and f_ == f_:
            entry["faithfulness"] = float(f_)
        if isinstance(r_, (int, float)) and r_ == r_:
            entry["answer_relevancy"] = float(r_)
        if entry:
            out[str(qid)] = entry
    return out


def _save_checkpoint(path: Path, scores: dict[str, dict[str, float]]) -> None:
    path.write_text(
        json.dumps({"scores_by_query_id": scores}, indent=2),
        encoding="utf-8",
    )


def _per_query_ragas_rows(per_query: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for r in per_query:
        qid = str(r.get("query_id", ""))
        if r.get("answer") in ("", "ERROR"):
            continue
        ctx = r.get("context") or []
        if not ctx:
            continue
        row = {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": ctx,
            "ground_truth": r.get("ground_truth", ""),
        }
        rows.append((qid, row))
    return rows


def _mean_two(
    scores: dict[str, dict[str, float]],
) -> tuple[float | None, float | None]:
    fs = [v["faithfulness"] for v in scores.values() if "faithfulness" in v]
    rs = [v["answer_relevancy"] for v in scores.values() if "answer_relevancy" in v]
    if not fs or not rs:
        return None, None
    return sum(fs) / len(fs), sum(rs) / len(rs)


def _resume_skip_nonzero_in_json(
    per_query: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """If per-query rows already store ragas_* from a prior partial run, treat as done."""
    out: dict[str, dict[str, float]] = {}
    for r in per_query:
        qid = str(r.get("query_id", ""))
        if not qid:
            continue
        f_ = r.get("ragas_faithfulness")
        r_ = r.get("ragas_answer_relevancy")
        if not isinstance(f_, (int, float)) or not isinstance(r_, (int, float)):
            continue
        if f_ <= 0.0 and r_ <= 0.0:
            continue
        if f_ == f_ and r_ == r_:
            out[qid] = {"faithfulness": float(f_), "answer_relevancy": float(r_)}
    return out


def rerun_one_condition(
    *,
    dataset: str,
    condition_id: str,
    output_dir: Path,
    config_path: Path,
    sleep_seconds: float,
    resume: bool,
    resume_json_scores: bool,
    clear_checkpoint: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Returns (old_generation_metrics, new_generation_metrics, retrieval_metrics unchanged)."""
    path = _find_condition_path(output_dir, dataset, condition_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    old_gm = {
        "answer_faithfulness": float(payload["generation_metrics"]["answer_faithfulness"]),
        "answer_relevance": float(payload["generation_metrics"]["answer_relevance"]),
    }
    retrieval_snapshot = json.loads(json.dumps(payload["retrieval_metrics"]))

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    generation_cfg = cfg.get("generation") or {}
    ragas_cfg = (cfg.get("evaluation") or {}).get("ragas") or {}

    evaluator = RAGASEvaluator(generation=generation_cfg, ragas=ragas_cfg)
    evaluator.prepare_models()

    ckpt_path = output_dir / f".ragas_rerun_ckpt_{dataset}_{condition_id}.json"
    if resume:
        scores = _load_checkpoint(ckpt_path)
        if resume_json_scores:
            jq = _resume_skip_nonzero_in_json(payload.get("per_query_results") or [])
            scores = {**jq, **scores}
    else:
        scores = {}
        if ckpt_path.exists() and clear_checkpoint:
            ckpt_path.unlink()

    rows = _per_query_ragas_rows(payload.get("per_query_results") or [])
    total = len(rows)
    for i, (qid, row) in enumerate(rows):
        if qid in scores and "faithfulness" in scores[qid] and "answer_relevancy" in scores[qid]:
            continue
        f_out, r_out = evaluator.score_single_row(row)
        if f_out is not None and r_out is not None:
            scores[qid] = {"faithfulness": f_out, "answer_relevancy": r_out}
        _save_checkpoint(ckpt_path, scores)
        if sleep_seconds > 0 and i + 1 < total:
            time.sleep(sleep_seconds)

    mean_f, mean_r = _mean_two(scores)
    if mean_f is None or mean_r is None:
        raise RuntimeError(f"RAGAS produced no scores for {condition_id} (check API key and logs).")

    new_gm = {
        "answer_faithfulness": round(mean_f, 4),
        "answer_relevance": round(mean_r, 4),
    }

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)

    payload["generation_metrics"]["answer_faithfulness"] = new_gm["answer_faithfulness"]
    payload["generation_metrics"]["answer_relevance"] = new_gm["answer_relevance"]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return old_gm, new_gm, retrieval_snapshot


def _verify_retrieval_unchanged(path: Path, before: dict[str, Any]) -> None:
    after = json.loads(path.read_text(encoding="utf-8"))["retrieval_metrics"]
    if after != before:
        raise RuntimeError(f"retrieval_metrics changed unexpectedly in {path}")


def _load_all_results_for_comparison(output_dir: Path, dataset: str) -> dict[str, Any]:
    all_results: dict[str, Any] = {}
    for cid in CONDITION_IDS:
        try:
            p = _find_condition_path(output_dir, dataset, cid)
        except FileNotFoundError:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        all_results[cid] = {
            "condition": cid,
            "display_name": data.get("display_name", cid),
            "group": data.get("group", ""),
            "dataset": data.get("dataset", dataset),
            "n_samples": data.get("n_samples"),
            "top_k": data.get("top_k"),
            "timestamp": data.get("timestamp"),
            "retrieval_metrics": data["retrieval_metrics"],
            "generation_metrics": data["generation_metrics"],
            "latency_stats": data["latency_stats"],
        }
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute RAGAS from saved *_top5.json files only.")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["standard_rag", "large_pool", "llm_listwise"],
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with generation + evaluation.ragas (default: .runtime_{dataset}.yaml if present, else configs/default.yaml)",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file and/or per-query ragas_* fields with non-zero values.",
    )
    parser.add_argument(
        "--resume-json-scores",
        action="store_true",
        help="With --resume, also treat per_query_results[].ragas_* non-zero as completed.",
    )
    parser.add_argument(
        "--no-clear-checkpoint",
        action="store_true",
        help="If set, do not delete checkpoint when starting without --resume (default clears stale ckpt).",
    )
    args = parser.parse_args()

    _require_openai_key_for_openrouter()

    output_dir: Path = args.output_dir
    dataset: str = args.dataset
    runtime_cfg = output_dir / f".runtime_{dataset}.yaml"
    if args.config is not None:
        config_path = args.config
    elif runtime_cfg.is_file():
        config_path = runtime_cfg
    else:
        config_path = ROOT / "configs" / "default.yaml"

    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    comparison_rows: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
    print(f"Using config: {config_path}\n")

    clear_ckpt = not args.no_clear_checkpoint

    for cid in args.conditions:
        if cid not in CONDITION_IDS:
            raise SystemExit(f"Unknown condition {cid!r}; expected one of {CONDITION_IDS}")
        print(f"=== {cid} ===")
        p = _find_condition_path(output_dir, dataset, cid)
        before_rm = json.loads(p.read_text(encoding="utf-8"))["retrieval_metrics"]
        old_gm, new_gm, rm_snap = rerun_one_condition(
            dataset=dataset,
            condition_id=cid,
            output_dir=output_dir,
            config_path=config_path,
            sleep_seconds=args.sleep_seconds,
            resume=args.resume,
            resume_json_scores=args.resume_json_scores,
            clear_checkpoint=clear_ckpt,
        )
        _verify_retrieval_unchanged(p, before_rm)
        if rm_snap != before_rm:
            raise RuntimeError("retrieval snapshot mismatch")
        comparison_rows.append(
            (
                cid,
                f"{old_gm['answer_faithfulness']:.4f}",
                f"{old_gm['answer_relevance']:.4f}",
                f"{new_gm['answer_faithfulness']:.4f}",
                f"{new_gm['answer_relevance']:.4f}",
            )
        )
        print(f"  faithfulness: {old_gm['answer_faithfulness']:.4f} -> {new_gm['answer_faithfulness']:.4f}")
        print(f"  relevance:    {old_gm['answer_relevance']:.4f} -> {new_gm['answer_relevance']:.4f}")
        print(f"  backup: {p.with_suffix(p.suffix + '.bak')}\n")

    # Update summary JSON
    summary_path = output_dir / f"{dataset}_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shutil.copy2(summary_path, summary_path.with_suffix(".json.bak"))
        for cid in args.conditions:
            if cid not in summary:
                continue
            p = _find_condition_path(output_dir, dataset, cid)
            data = json.loads(p.read_text(encoding="utf-8"))
            gm = data["generation_metrics"]
            summary[cid]["generation_metrics"]["answer_faithfulness"] = gm["answer_faithfulness"]
            summary[cid]["generation_metrics"]["answer_relevance"] = gm["answer_relevance"]
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Updated {summary_path} (backup: {summary_path.with_suffix('.json.bak')})\n")

    # Regenerate comparison JSON
    all_results = _load_all_results_for_comparison(output_dir, dataset)
    cmp_payload = build_comparison_payload(all_results)
    cmp_path = output_dir / f"{dataset}_comparison.json"
    if cmp_path.is_file():
        shutil.copy2(cmp_path, cmp_path.with_name(cmp_path.name + ".bak"))
    cmp_path.write_text(json.dumps(cmp_payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {cmp_path}\n")

    # Final table: old vs new for touched conditions + full 5-way from disk
    print("=" * 100)
    print("RAGAS old -> new (updated conditions only)")
    print("-" * 100)
    hdr = f"{'condition':<22} {'Faith(old)':>12} {'Relev(old)':>12} {'Faith(new)':>12} {'Relev(new)':>12}"
    print(hdr)
    print("-" * 100)
    for row in comparison_rows:
        print(f"{row[0]:<22} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
    print("=" * 100)

    labels_cfg = ROOT / "configs" / "default.yaml"
    if not labels_cfg.is_file():
        raise SystemExit(f"Missing {labels_cfg} for condition labels.")
    condition_rows = load_conditions(str(labels_cfg))

    print_results_table(all_results, condition_rows)


if __name__ == "__main__":
    main()
