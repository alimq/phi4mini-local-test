#!/usr/bin/env bash
set -euo pipefail

# ---- config (override via environment if you want) ----
DATA_DIR="${DATA_DIR:-/var/lib/phi4mini}"
DB="${DB:-$DATA_DIR/seen.sqlite}"
LOG="${LOG:-$DATA_DIR/backfill.log}"

# Prefer the project venv, fallback to system python3 if venv missing.
if [[ -x "/opt/phi4mini/.venv/bin/python" ]]; then
  PY="/opt/phi4mini/.venv/bin/python"
else
  PY="${PY:-python3}"
fi

# Cap: if NEW backlog >= this, skip discovery until it drops.
DISCOVER_MAX_NEW="${DISCOVER_MAX_NEW:-5000}"

# How often to run discovery (seconds)
DISCOVER_INTERVAL_S="${DISCOVER_INTERVAL_S:-3600}"   # 1 hour

# How often to run the worker loop (seconds)
WORK_INTERVAL_S="${WORK_INTERVAL_S:-60}"            # 1 minute

# Worker batch size per iteration (keep modest; it's 24/7)
WORK_BATCH="${WORK_BATCH:-30}"

# Additional fetch/extract limits (match your scripts' flags)
MAX_BYTES="${MAX_BYTES:-2500000}"                   # 2.5MB
MAX_CHARS="${MAX_CHARS:-50000}"                     # clamp huge pages

umask 022
mkdir -p "$DATA_DIR"

log() {
  echo "[$(date -u +%FT%TZ)] [backfill_loop] $*"
}

get_new_count() {
  sqlite3 "$DB" 'select count(*) from url_queue where status="NEW";' 2>/dev/null || echo 0
}

# ---- main loop ----
next_discover_epoch=0

log "starting (DATA_DIR=$DATA_DIR DB=$DB DISCOVER_MAX_NEW=$DISCOVER_MAX_NEW DISCOVER_INTERVAL_S=$DISCOVER_INTERVAL_S WORK_INTERVAL_S=$WORK_INTERVAL_S WORK_BATCH=$WORK_BATCH PY=$PY)"

while true; do
  now_epoch="$(date +%s)"

  # Discovery (hourly by default), but only if backlog is below cap
  if [[ "$now_epoch" -ge "$next_discover_epoch" ]]; then
    new_count="$(get_new_count)"
    if [[ "$new_count" -ge "$DISCOVER_MAX_NEW" ]]; then
      log "INFO: backlog NEW=$new_count >= cap=$DISCOVER_MAX_NEW, skipping discovery this cycle"
    else
      log "INFO: discovery start (NEW=$new_count cap=$DISCOVER_MAX_NEW)"
      if "$PY" /opt/phi4mini/backfill_discover.py \
          --data-dir "$DATA_DIR" \
          --db "$DB"; then
        log "INFO: discovery done"
      else
        log "WARN: discovery failed (continuing)"
      fi
    fi
    next_discover_epoch="$((now_epoch + DISCOVER_INTERVAL_S))"
  fi

  # Worker (runs every minute by default)
  log "INFO: worker start (batch=$WORK_BATCH)"
  if "$PY" /opt/phi4mini/backfill_worker.py \
        --data-dir "$DATA_DIR" \
        --db "$DB" \
        --outbox "$DATA_DIR/outbox.jsonl" \
        --batch "$WORK_BATCH" \
        --max-bytes "$MAX_BYTES" \
        --max-chars "$MAX_CHARS"; then
    :
  else
    log "WARN: worker failed (continuing)"
  fi

  # Quick status snapshot
  # (kept short so log stays readable)
  if command -v sqlite3 >/dev/null 2>&1; then
    status_line="$(sqlite3 "$DB" 'select group_concat(status||"="||cnt," ") from (select status, count(*) cnt from url_queue group by status);' 2>/dev/null || true)"
    [[ -n "${status_line:-}" ]] && log "INFO: queue ${status_line}"
  fi

  sleep "$WORK_INTERVAL_S"
done
