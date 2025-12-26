# RAG Endpoint Runbook

## Quick Health Checks
- App health: `curl -s http://127.0.0.1:8002/health`
- Metrics: `curl -s http://127.0.0.1:8002/metrics`
- Nginx via proxy: `curl -s http://127.0.0.1/rag/health`
- Retrieval debug: `curl -s "http://127.0.0.1:8002/debug/retrieval?q=advantages%20of%20ollama"`

## Smoke Test
Run a fixed set of queries and report status + latency:

```bash
BASE_URL=http://127.0.0.1/rag/api/ask ./scripts/smoke_test.sh
```

## Ingest Audit
Verify index contents and embedding metadata:

```bash
./scripts/ingest_audit.py --db /var/lib/phi4mini/rag.sqlite
```

## GraphRAG Build
Build a lightweight entity graph from existing chunks:

```bash
./scripts/build_graph.py --db /var/lib/phi4mini/rag.sqlite --reset
```

## Request Correlation
The API returns `x-request-id` on every response. To log it in nginx access logs:

```
log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                '$status $body_bytes_sent "$http_referer" '
                '"$http_user_agent" request_id=$request_id';
```

## Nginx Proxy Settings
Use `nginx_snippet_rag.conf` for timeouts and buffering. Then:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Tunables (env vars)
- `PHI4MINI_TOP_K`, `PHI4MINI_CANDIDATE_K`, `PHI4MINI_RELAXED_TOP_K`
- `PHI4MINI_SCORE_THRESHOLD`
- `PHI4MINI_EMBED_TIMEOUT_S`, `PHI4MINI_GEN_TIMEOUT_S`, `PHI4MINI_RETRIEVE_TIMEOUT_S`
- `PHI4MINI_MAX_CONCURRENCY`
- `PHI4MINI_INDEX_NAME`, `PHI4MINI_INDEX_VERSION`
- `PHI4MINI_GRAPH_ENABLED`, `PHI4MINI_GRAPH_ENTITY_LIMIT`, `PHI4MINI_GRAPH_EXPAND_LIMIT`, `PHI4MINI_GRAPH_CHUNK_LIMIT`

## Common Failure Modes
- Empty retrieval: run ingest audit, confirm embeddings exist, and verify embedding model matches between ingest and query.
- Timeouts: increase nginx timeouts and `PHI4MINI_GEN_TIMEOUT_S` after measuring /metrics p95.
