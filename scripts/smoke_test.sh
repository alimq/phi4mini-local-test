#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1/rag/api/ask}"
QUERIES=(
  "what is rag"
  "advantage of ollama"
  "advantages of running llm locally"
  "rag indexing best practices"
  "embedding model mismatch"
  "chunking overlap guidance"
  "reranking techniques"
  "vector store sqlite fts"
  "llm quantization benefits"
  "local inference latency"
  "retrieval augmented generation evals"
  "rag pipeline monitoring"
  "context window limits"
  "document ingestion pipeline"
  "ollama timeout troubleshooting"
  "inference cpu vs gpu"
  "semantic search vs bm25"
  "ollama embeddings"
  "rag failure modes"
  "local model deployment"
)

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

printf "Base URL: %s\n" "$BASE_URL"

for q in "${QUERIES[@]}"; do
  tmp_body="$(mktemp)"
  tmp_hdr="$(mktemp)"

  result=$(curl -s -D "$tmp_hdr" -o "$tmp_body" \
    -w "%{time_total} %{http_code}" \
    --get --data-urlencode "q=$q" \
    "$BASE_URL")

  total_time=$(echo "$result" | awk '{print $1}')
  status=$(echo "$result" | awk '{print $2}')
  request_id=$(awk -v RS='\r\n' 'tolower($0) ~ /^x-request-id:/ {print $2}' "$tmp_hdr" | tr -d ' ')

  num_returned=$(python3 - <<PY
import json
import sys
try:
    data = json.load(open("$tmp_body", "r", encoding="utf-8"))
    print(data.get("num_returned", ""))
except Exception:
    print("")
PY
)

  echo "status=$status t_total_s=${total_time:-"-"} request_id=${request_id:-"-"} num_returned=${num_returned:-"-"} query=\"$q\""

  rm -f "$tmp_body" "$tmp_hdr"
  sleep 0.2
done
