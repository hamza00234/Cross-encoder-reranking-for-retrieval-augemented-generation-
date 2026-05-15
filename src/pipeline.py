from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.generation.generator import AnswerGenerator, build_answer_generator
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.indexer import FAISSIndexer
from src.retrieval.cross_encoder import CrossEncoderReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_scorer import HybridScorer


def _load_dataset_robust(*args: Any, **kwargs: Any):
    """
    Hugging Face ``datasets`` can throw TypeError(dataclass) when the local
    cache metadata was written by a different ``datasets`` major version.
    Retry with a forced re-download; if that still fails, point to upgrade + cache wipe.
    """
    try:
        return load_dataset(*args, **kwargs)
    except TypeError as exc:
        if "dataclass" not in str(exc).lower():
            raise
        try:
            return load_dataset(*args, **kwargs, download_mode="force_redownload")
        except Exception as exc2:
            raise RuntimeError(
                "Hugging Face dataset load failed (old `datasets` or a broken cache). "
                "Run: pip install -U 'datasets>=4.0.0,<6.0.0' "
                "then optionally: rm -rf ~/.cache/huggingface/datasets/hotpot_qa* "
                "~/.cache/huggingface/datasets/musique*"
            ) from exc2


def load_hotpotqa(n_samples: int = 500) -> tuple[list[dict], list[dict]]:
    dataset = _load_dataset_robust("hotpot_qa", "distractor", split="validation")
    corpus: list[dict] = []
    queries: list[dict] = []

    for i, row in enumerate(dataset.select(range(min(n_samples, len(dataset))))):
        try:
            titles = row["context"]["title"]
            sentences_by_title = row["context"]["sentences"]
            support_titles = set(row["supporting_facts"]["title"])
        except Exception:
            # fallback for canonical structure
            try:
                titles = row["context"]["title"]
                sentences_by_title = row["context"]["sentences"]
                support_titles = set(row["supporting_facts"]["title"])
            except Exception as exc:
                print(f"[HotpotQA] Skipping malformed sample: {exc}")
                continue

        query_gold: list[str] = []
        for j, (title, sentence_list) in enumerate(zip(titles, sentences_by_title)):
            doc_id = f"hotpot_{i}_{j}"
            text = " ".join(sentence_list)
            corpus.append({"document_id": doc_id, "source_title": title, "content": text})
            if title in support_titles:
                query_gold.append(f"{doc_id}_chunk_0")

        queries.append(
            {
                "query_id": f"hotpot_q_{i}",
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "gold_chunk_ids": query_gold,
            }
        )
    return corpus, queries


def load_musique(n_samples: int = 200) -> tuple[list[dict], list[dict]]:
    dataset = _load_dataset_robust("musique_ans_v1.0", split="validation")
    corpus: list[dict] = []
    queries: list[dict] = []

    for i, row in enumerate(dataset.select(range(min(n_samples, len(dataset))))):
        try:
            paragraphs = row["paragraphs"]
            question = row["question"]
            answer = row["answer"]
        except Exception as exc:
            print(f"[MuSiQue] Skipping malformed sample: {exc}")
            continue

        query_gold: list[str] = []
        for j, p in enumerate(paragraphs):
            if not isinstance(p, dict) or "paragraph_text" not in p:
                print(f"[MuSiQue] Skipping malformed paragraph in sample {i}")
                continue
            doc_id = f"musique_{i}_{j}"
            corpus.append(
                {
                    "document_id": doc_id,
                    "source_title": p.get("title", f"musique_title_{j}"),
                    "content": p["paragraph_text"],
                }
            )
            if p.get("is_supporting", False):
                query_gold.append(f"{doc_id}_chunk_0")
        queries.append(
            {
                "query_id": f"musique_q_{i}",
                "question": question,
                "answer": answer,
                "gold_chunk_ids": query_gold,
            }
        )
    return corpus, queries


class RAGPipeline:
    """
    Orchestrates the full cross-encoder RAG pipeline.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

        ing = self.config["ingestion"]
        ret = self.config["retrieval"]
        gen = self.config["generation"]
        hyb = self.config["hybrid"]

        self.chunker = DocumentChunker(ing["chunk_size"], ing["chunk_overlap"])
        self.embedder = ChunkEmbedder(model_name=ing["embedding_model"])
        self.indexer = FAISSIndexer(
            index_type=ing["faiss_index_type"],
            embedding_dim=ing["embedding_dim"],
            index_path=ing["faiss_index_path"],
            metadata_path=ing["metadata_path"],
        )
        self.retriever = DenseRetriever(
            embedder=self.embedder,
            indexer=self.indexer,
            candidate_pool_size=ret["candidate_pool_size"],
        )
        self.reranker = CrossEncoderReranker(model_name=ret["cross_encoder_model"])
        self.hybrid = HybridScorer(alpha=hyb["alpha"])
        self.generator: AnswerGenerator = build_answer_generator(gen)

    def build_index(self, corpus: list[dict]) -> None:
        chunks = self.chunker.chunk_documents(corpus)
        embeddings, metadata = self.embedder.embed_chunks(chunks)
        self.indexer.build(embeddings, metadata)
        self.indexer.save()

    def load_index(self) -> None:
        self.indexer.load()

    def _select_chunks(self, query: str, condition: str, top_k: int) -> list[dict]:
        candidates = self.retriever.retrieve(query)
        if condition in {"standard_rag", "large_pool"}:
            return sorted(candidates, key=lambda x: x["cosine_score"], reverse=True)[:top_k]
        reranked = self.reranker.rerank(query, candidates)
        if condition == "cross_encoder_hybrid":
            return self.hybrid.score(reranked)[:top_k]
        return self.hybrid.pure_ce_score(reranked)[:top_k]

    def run_query(
        self,
        query: str,
        condition: str = "cross_encoder_pure",
        top_k: int = 5,
        *,
        skip_generation: bool = False,
    ) -> dict:
        start = time.perf_counter()
        chunks = self._select_chunks(query, condition, top_k)
        if skip_generation:
            answer = ""
        else:
            answer = self.generator.generate(query=query, context_chunks=chunks)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "scores": [
                float(c.get("final_score", c.get("norm_ce_score", c.get("cosine_score", 0.0))))
                for c in chunks
            ],
            "latency_ms": float(latency_ms),
        }

    def run_batch(self, queries: list[dict], condition: str, top_k: int) -> list[dict]:
        outputs = []
        for row in tqdm(queries, desc=f"Running {condition}"):
            outputs.append(self.run_query(query=row["question"], condition=condition, top_k=top_k))
        return outputs
