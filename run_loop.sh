#!/usr/bin/env bash
set -euo pipefail

PY=/opt/phi4mini/.venv/bin/python
CODE=/opt/phi4mini
DATA=/var/lib/phi4mini

mkdir -p "$DATA"

LOG="$DATA/run.log"
touch "$LOG"

# Send all stdout/stderr to run.log so you can tail it in real time.
exec >>"$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

export PYTHONUNBUFFERED=1

while true; do
  log "scraper start"
  # 1) Scrape feeds -> append new items to outbox, update seen.sqlite
  $PY -u $CODE/copied_scraper.py \
    --feeds-file $DATA/feeds.txt \
    --out $DATA/outbox.jsonl \
    --db $DATA/seen.sqlite \
    --max-items 20 \
    --log-level INFO \
    || { log "[warn] scraper failed; sleeping 120s"; sleep 120; continue; }
  log "scraper end"

  # 2) Generate digest (append per run) using local Ollama model
  log "digest start"
  cd "$DATA"
  $PY -u $CODE/feed_into_phi.py \
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

  # 3) Sleep 15 minutes
  log "sleep 900s"
  sleep 900
done
