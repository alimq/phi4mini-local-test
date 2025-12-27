#!/usr/bin/env bash
set -euo pipefail

PY=/opt/phi4mini/.venv/bin/python
ROOT=/opt/phi4mini
DATA=/var/lib/phi4mini

mkdir -p "$DATA"

LOG="$DATA/run.log"
touch "$LOG"

# Send all stdout/stderr to run.log so you can tail it in real time.
exec >>"$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

export PYTHONUNBUFFERED=1

# 24/7 loop:
#   1) Scrape feeds -> /var/lib/phi4mini/outbox.jsonl
#   2) Summarize new items -> /var/lib/phi4mini/digest.md
#   3) Use the old 15-minute "sleep" window to maintain the global RAG index.

while true; do
  log "scraper start"
  # 1) Scrape feeds -> append new items to outbox.jsonl, update seen.sqlite
  PYTHONPATH="$ROOT" $PY -u -m app.copied_scraper \
    --feeds-file $ROOT/config/feeds.txt \
    --out $DATA/outbox.jsonl \
    --db $DATA/seen.sqlite \
    --max-items 20 \
    --log-level INFO \
    || { log "[warn] scraper failed; sleeping 120s"; sleep 120; continue; }
  log "scraper end"

  # 2) Generate digest (append per run) using local Ollama model
  log "digest start"
  cd "$DATA"
  PYTHONPATH="$ROOT" $PY -u -m app.feed_into_phi \
    --outbox $DATA/outbox.jsonl \
    --state-db $DATA/seen.sqlite \
    --cursor-key outbox_offset_bytes \
    --batch-items 120 \
    --days 0 \
    --last-n 0 \
    --max-items 120 \
    --model phi4-mini:latest \
    --num-ctx 4096 \
    --first-token-timeout 600 \
    --read-timeout 7200 \
    --log-level INFO \
    || { log "[warn] feed_into_phi failed; sleeping 120s"; sleep 120; continue; }
  log "digest end"

  # 3) Maintain the global RAG index for up to 900 seconds.
  #    If embeddings are unavailable, the indexer still keeps SQLite FTS up-to-date.
  log "rag_index start budget_s=900"
  PYTHONPATH="$ROOT" $PY -u -m app.rag_index_outbox \
    --outbox $DATA/outbox.jsonl \
    --db $DATA/rag.sqlite \
    --state-db $DATA/rag.sqlite \
    --cursor-key outbox_offset_bytes \
    --max-seconds 900 \
    --log-level INFO \
    || { log "[warn] rag_index_outbox failed; sleeping 900s"; sleep 900; }
  log "rag_index end"
done
