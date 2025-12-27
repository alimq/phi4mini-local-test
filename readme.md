# phi4mini-local-test

Small local project that turns RSS feeds into a digest and lets you query the same data through a simple RAG UI.

## Try it (live)

- Summarizer UI: http://46.224.80.50/summarizer/
- RAG UI: http://46.224.80.50/rag/

## Benchmark note (retrieval)

In this repo’s AI-news retrieval benchmark, BM25 scored higher than GraphRAG on these retrieval metrics under this setup.

BM25 vs GraphRAG (mean over labeled queries; delta = GraphRAG - BM25):
- recall@10: 0.6417 vs 0.5033 (delta -0.1383)
- recall@20: 0.6500 vs 0.5033 (delta -0.1467)
- MRR: 0.5372 vs 0.4639 (delta -0.0733)
- nDCG@10: 0.5659 vs 0.4611 (delta -0.1048)
- precision@10: 0.1100 vs 0.0833 (delta -0.0267)

Run context (from `eval/graphrag_eval/RUN.md`):
- `--topk 20`
- The corpus file `corpus.jsonl` is not committed

Query set used: `eval/graphrag_eval/queries_labeled.json` (30 labeled queries).

## Benchmark: GraphRAG vs BM25 on AI news

This repo includes a retrieval benchmark comparing GraphRAG vs BM25.

Files:
- `eval/graphrag_eval/report.md`
- `eval/graphrag_eval/report.json`
- `eval/graphrag_eval/queries_labeled.json`
- `eval/graphrag_eval/queries_answerable.json`

Key facts from the benchmark files:
- Query count: 30 labeled queries (`queries_labeled.json`).
- TopK used: 20 (`report.md`).
- GraphRAG mapping notes: `{"used_doc_ids": false, "extracted_from_text": false, "doc_level": false}` (`report.json`).

Summary metrics (from `report.md`):

| Metric | BM25 | GraphRAG | Delta (G-B) | Relative | 95% CI BM25 | 95% CI GraphRAG |
|---|---:|---:|---:|---:|---:|---:|
| recall@10 | 0.6417 | 0.5033 | -0.1383 | -21.56% | [0.4831, 0.8167] | [0.3367, 0.6667] |
| recall@20 | 0.6500 | 0.5033 | -0.1467 | -22.56% | [0.4833, 0.8333] | [0.3367, 0.6667] |
| mrr | 0.5372 | 0.4639 | -0.0733 | -13.65% | [0.3744, 0.7050] | [0.3000, 0.6361] |
| ndcg@10 | 0.5659 | 0.4611 | -0.1048 | -18.52% | [0.4067, 0.7344] | [0.3009, 0.6198] |
| precision@10 | 0.1100 | 0.0833 | -0.0267 | -24.24% | [0.0700, 0.1600] | [0.0500, 0.1234] |

Per-query excerpts (from `report.md`, top 3 worst for GraphRAG):

- Query: What did researchers say about Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face?
  - Gold IDs: 6818
  - BM25 top10: 6805, 3871, 6817, 6818, 6838, 6508, 3776, 6810, 6814, 6811 (first gold rank: 4)
  - GraphRAG top10: 6805, 6817, 6810, 6811, 6814, 6808, 3871, 6508 (first gold rank: None)

- Query: What is the update regarding Rocket Money x Hugging Face: Scaling Volatile ML Models in Production​?
  - Gold IDs: 7105, 7103
  - BM25 top10: 7098, 7105, 7106, 7103, 6955, 7486, 7102, 6234, 7321, 6155 (first gold rank: 2)
  - GraphRAG top10: 7098, 6182, 3631, 7469, 2804, 5970, 5971, 5028 (first gold rank: None)

- Query: What is A Knapsack Public Key Cryptosystem Based on Arithmetic in Finite Fields (1988) [pdf] about?
  - Gold IDs: 12252
  - BM25 top10: 12252, 8087, 8056, 11418, 12354, 9284, 11111, 8683, 2533, 3162 (first gold rank: 1)
  - GraphRAG top10: 12245, 12023, 11848, 11604, 11946, 12014, 11616, 12016 (first gold rank: None)

Example queries from the benchmark set (from `queries_labeled.json`):

- "What is Powerful ASR + diarization + speculative decoding with Hugging Face Inference Endpoints about?"
- "What is the significance of Train 400x faster Static Embedding Models with Sentence Transformers in AI?"
- "What did researchers say about Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face?"

Results for those exact queries (from `report.json`):

- "Powerful ASR + diarization + speculative decoding..."
  - BM25: recall@20 1.0, MRR 1.0
  - GraphRAG: recall@20 0.5, MRR 1.0
- "Train 400x faster Static Embedding Models..."
  - BM25: recall@20 1.0, MRR 1.0
  - GraphRAG: recall@20 1.0, MRR 1.0
- "Welcome Mixtral - a SOTA Mixture of Experts..."
  - BM25: recall@20 1.0, MRR 0.25
  - GraphRAG: recall@20 0.0, MRR 0.0

## Brief project context

- Summarizer: renders the latest digest as a small website.
- Global RAG: questions over the full news corpus, with sources.
- User-docs RAG: upload files and ask questions over them.
