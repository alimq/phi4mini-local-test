# phi4mini-local-test

Local, always-on news summarizer + RAG stack built around phi4-mini and Ollama. It scrapes RSS feeds, generates a digest, and keeps a lightweight SQLite RAG index you can query through a simple web UI.

## What the website does

There are two lightweight web experiences that make the output easy to browse:

- Summarizer view: renders the latest digest markdown as a readable page (with a raw markdown download).
- Global RAG: ask questions over everything that has been ingested into the outbox (with sources).
- User-docs RAG: upload your own files and ask questions over them in a per-user namespace.

If you only want one page to show a friend, point them at the Summarizer or the Global RAG UI. They complement each other: the Summarizer gives a daily/weekly narrative, and the RAG UI answers ad-hoc questions with sources.

## RAG focus (how it works)

This repo keeps RAG intentionally simple and local:

- Storage: SQLite (`rag.sqlite`) plus FTS5 for fast keyword retrieval.
- Embeddings: optional embedding rerank using Ollama (`nomic-embed-text` by default).
- Graph signals: optional GraphRAG-style entity expansion (toggle via `PHI4MINI_GRAPH_ENABLED`).
- Outputs: answers + source links for the global news corpus, or answers + local file paths for user uploads.

The result is a fast, low-dependency RAG system that runs on a small server without external vector DBs.

## Pipeline overview

The 24/7 loop is:

1) Scrape RSS feeds into `/var/lib/phi4mini/outbox.jsonl`
2) Summarize new items into `/var/lib/phi4mini/digest.md`
3) Maintain the global RAG index in `/var/lib/phi4mini/rag.sqlite`

The wrapper script for this is `run_loop.sh`.

## Quickstart

Prereqs:
- Ollama running locally
- `phi4-mini:latest` model pulled
- `nomic-embed-text` pulled (for embeddings; optional but recommended)

Setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run a one-off digest:

```bash
python3 -u feed_into_phi.py \
  --days 90 --last-n 300 --max-items 120 \
  --snippet-chars 260 --max-prompt-chars 6500 \
  --num-predict-chunk 220 --num-predict-final 900
```

Run the continuous loop:

```bash
bash run_loop.sh
```

## Web UIs

Summarizer (HTML digest):

```bash
python3 web/serve_digest.py
```

Or, if you prefer FastAPI:

```bash
uvicorn summarizer_web:app --host 0.0.0.0 --port 8088
```

Global RAG (news corpus):

```bash
uvicorn rag_server:app --host 0.0.0.0 --port 8090
```

User-docs RAG (upload your own files):

```bash
uvicorn rag_user_server:app --host 0.0.0.0 --port 8091
```

## Useful endpoints

Global RAG:
- `GET /` UI
- `GET /api/ask?q=...` JSON response
- `GET /debug/retrieval?q=...` retrieval metadata

User-docs RAG:
- `GET /` upload + ask UI
- `POST /upload` upload file
- `GET /api/ask?user_id=...&q=...` JSON response

## Configuration (common env vars)

- `OLLAMA_GENERATE_URL` (default `http://127.0.0.1:11434/api/generate`)
- `OLLAMA_EMBED_URL` (default `http://127.0.0.1:11434/api/embeddings`)
- `OLLAMA_GEN_MODEL` (default `phi4-mini:latest`)
- `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`)
- `PHI4MINI_RAG_DB` (default `/var/lib/phi4mini/rag.sqlite`)
- `PHI4MINI_DIGEST_MD` (default `/var/lib/phi4mini/digest.md`)

## Historical notes

- Optional backfill from sitemaps is documented in `BACKFILL.md`.
- GraphRAG vs BM25 eval (Dec 2025): BM25 outperformed GraphRAG on the test set. See `eval/graphrag_eval/report.md`.

## Data locations

Runtime data lives in `/var/lib/phi4mini`. The repo only contains code and docs; generated artifacts should not be committed.
