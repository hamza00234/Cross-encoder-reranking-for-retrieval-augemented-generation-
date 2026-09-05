# Cross-Encoder Reranking RAG System

Two-stage RAG pipeline with:
- Dense bi-encoder retrieval (top-25 candidate pool)
- Cross-encoder reranking (primary condition uses pure CE ranking)
- Optional hybrid scoring ablation
- OpenRouter or local Ollama generation (`generation.backend` in `configs/default.yaml`)
- Retrieval + RAGAS evaluation

## Architecture

```text
Query
  |
  v
[Dense Retriever: all-mpnet-base-v2 + FAISS] --> top-25 candidates
  |
  v
[Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2] --> reranked candidates
  |                             |
  |                             +--> (optional) HybridScorer alpha blend
  v
Top-k context chunks (ranked, highest first)
  |
  v
[LLM: OpenRouter or Ollama, T=0.0]
  |
  v
Answer + metrics (retrieval + RAGAS)
```

## Installation

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the env template and add your keys (never commit `.env`):
   ```bash
   cp .env.example .env
   ```
   Set `OPENROUTER_API_KEY` for the default [OpenRouter](https://openrouter.ai) backend. Models live under `generation` in `configs/default.yaml`. For local [Ollama](https://ollama.com) instead, set `generation.backend` to `ollama` and optionally `OLLAMA_BASE_URL`, then pull models (for example `mistral:7b-instruct` / `llama3:8b-instruct`).

## Build Index From Custom Corpus

Use `RAGPipeline` and pass corpus documents in format:
`{"document_id": str, "source_title": str, "content": str}`.

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline("configs/default.yaml")
corpus = [
    {"document_id": "doc1", "source_title": "Title 1", "content": "Document text..."},
]
pipeline.build_index(corpus)
```

## Run Experimental Conditions

```bash
python experiments/run_experiment.py --dataset hotpotqa --condition standard_rag --generator mistral --top_k 5 --config configs/default.yaml
python experiments/run_experiment.py --dataset hotpotqa --condition large_pool --generator mistral --top_k 5 --config configs/default.yaml
python experiments/run_experiment.py --dataset hotpotqa --condition llm_listwise --generator mistral --top_k 5 --config configs/default.yaml
python experiments/run_experiment.py --dataset hotpotqa --condition cross_encoder_pure --generator mistral --top_k 5 --config configs/default.yaml
python experiments/run_experiment.py --dataset hotpotqa --condition cross_encoder_hybrid --generator mistral --top_k 5 --alpha 0.3 --config configs/default.yaml
```

## Run Alpha Ablation

```bash
python experiments/ablation_alpha.py
```

Outputs `results/ablation_alpha.json`.

## Run Significance Tests

```bash
python experiments/significance_test.py
```

Uses Wilcoxon signed-rank tests with Bonferroni correction (`0.05/6 = 0.00833`), outputs `results/significance_tests.json`.

## Output Format

Main experiment outputs:
`results/{dataset}_{condition}_{generator}_top{k}.json`

Structure:
- `config`
- `retrieval_metrics` (`recall@5`, `recall@10`, `precision@5`, `mrr`)
- `generation_metrics` (`answer_faithfulness`, `answer_relevance`)
- `per_query_results`
- `latency_stats` (`mean_ms`, `p95_ms`, `throughput_qps`)

## Known Limitations

- Cross-encoder reranking over 25 candidates can be latency-heavy.
- RAGAS evaluation depends on judge availability (`OPENROUTER_API_KEY` / `OPENAI_API_KEY` in `.env`, or a local Ollama judge).
- Gold chunk mapping is an approximation when support facts are title/paragraph-level but chunking creates multiple chunks.
