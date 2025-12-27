# GraphRAG vs BM25 Evaluation Run

Command used:

```
python eval_graphrag_vs_bm25.py \
  --corpus corpus.jsonl \
  --queries queries_labeled.json \
  --topk 20 \
  --timeout 180 \
  --graphrag_http_url "http://127.0.0.1:8000/debug/retrieval" \
  --out_dir results_real
```

Notes:
- GraphRAG endpoint: http://127.0.0.1:8000/debug/retrieval
- The corpus (`corpus.jsonl`) is intentionally excluded from version control.
