# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python scripts drive the pipeline: scraping (`copied_scraper.py`), summarization (`feed_into_phi.py`), RAG indexing/serving (`rag_index_outbox.py`, `rag_server.py`, `rag_user_server.py`), and utilities (`rag_db.py`, `summarizer_web.py`).
- Historical backfill lives in `backfill_discover.py`, `backfill_worker.py`, and the wrapper loop `backfill_loop.sh`.
- Web UI lives under `web/` (`web/serve_digest.py`, `web/requirements-web.txt`).
- Ops scripts and docs: `run_loop.sh`, `setup_web.sh`, `BACKFILL.md`, `DEPLOY_NOTES.md`, `readme.md`.

## Build, Test, and Development Commands
- Install Python deps (local dev): `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.
- Run a one-off digest (example from `readme.md`):
  `python3 -u feed_into_phi.py --days 90 --last-n 300 --max-items 120 --snippet-chars 260 --max-prompt-chars 6500`.
- Run the continuous loop: `bash run_loop.sh` (scrape -> digest -> RAG index; writes to `/var/lib/phi4mini`).
- Run backfill continuously: `bash backfill_loop.sh` (uses `/var/lib/phi4mini/seen.sqlite`).
- Local web preview: `python3 web/serve_digest.py` (serves `/var/lib/phi4mini/digest.md` on port 8088 by default). For full nginx setup, see `setup_web.sh`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and snake_case; match existing lightweight, script-style modules.
- Prefer clear, explicit CLI flags (argparse) and small helper functions; keep logging at INFO/WARN levels.
- Shell scripts use `set -euo pipefail` and uppercase env vars for configuration.

## Testing Guidelines
- No automated test suite is present. Validate changes by running a short scrape/digest cycle and checking outputs in `/var/lib/phi4mini/` (`outbox.jsonl`, `digest.md`, `seen.sqlite`).

## Commit & Pull Request Guidelines
- No Git history is available in this bundle, so no established commit conventions were found.
- When contributing, use imperative subjects (e.g., "Add backfill sitemap rules") and include a brief summary of changes plus commands run.

## Configuration & Ops Notes
- Runtime data lives under `/var/lib/phi4mini`; do not commit generated artifacts.
- Systemd units and nginx config are referenced in `DEPLOY_NOTES.md` and `web/README_WEB.md`; web installs use a self-signed cert by default.
