# Historical backfill beyond RSS (continuous)

RSS feeds usually expose only a rolling window. This backfill adds a **second path**
that steadily finds and processes older URLs, without disturbing your existing RSS loop.

## What it does

### 1) Discovery (low impact, bounded)

It continuously discovers historical URLs in two ways:

**A. Sitemaps (most sites)**
- For each domain, reads `robots.txt` and extracts any `Sitemap:` entries (when present)
- Falls back to common sitemap locations (`/sitemap.xml`, `/sitemap_index.xml`, etc.)
- Parses sitemap indexes + urlsets and enqueues URLs into SQLite `url_queue` (idempotent)

**B. arXiv API (deep history / cursor-based)**
- Uses the arXiv API to page through older items (ascending by submitted date)
- Stores cursors per category in SQLite (`arxiv_state`)
- Enqueues `https://arxiv.org/abs/...` pages for the worker to fetch and extract

### 2) Worker (fetch + extract)

- Claims URLs from `url_queue` (oldest-first)
- **Respects robots.txt by default** (best-effort); can be disabled with `--no-robots`
- Best-effort Crawl-delay + per-domain throttle
- Downloads (bounded by `--max-bytes`)
- Extracts main content via `trafilatura`
- Caps extracted text (`--max-chars`) to prevent “whale” pages
- Appends normalized records into `/var/lib/phi4mini/outbox.jsonl`
- Tracks idempotency in SQLite `seen` (`id = url:sha256(url)`)

This keeps your pipeline *moving forward* through history even when RSS stops.

## Always-on sources (built in)

Even if your `feeds.txt` is small, discovery also includes a curated set of AI-heavy domains
by default (you can disable this).

Built-in sitemap seeds include:
- openai.com (blog)
- anthropic.com (news / research)
- deepmind.google (DeepMind site)
- microsoft.com (Microsoft Research blog)
- developer.nvidia.com + www.nvidia.com (NVIDIA blogs)
- pytorch.org (PyTorch blog)
- huggingface.co (Hugging Face blog)
- paperswithcode.com (papers + blog)

Additional built-in AI/ML sources (sitemap-based) include:
- ai.meta.com (Meta AI blog)
- cohere.com (blog + research)
- stability.ai (blog/news)
- deeplearning.ai (The Batch + blog)
- blog.tensorflow.org (TensorFlow blog)
- research.ibm.com (IBM Research blog)
- aws.amazon.com/blogs/machine-learning (AWS ML blog)
- cloud.google.com (AI/ML blog sections)
- research.google/blog (Google Research blog)
- databricks.com/blog (Databricks blog)
- wandb.ai/fully-connected (Weights & Biases)
- scale.com/blog (Scale AI)
- perplexity.ai/blog (Perplexity)
- assemblyai.com/blog (AssemblyAI)
- together.ai/blog (Together AI)
- ai21.com/blog (AI21 Labs)
- mistral.ai (news/blog)
- adept.ai/blog (Adept)
- cerebras.net/blog (Cerebras)
- groq.com/blog (Groq)
- run.ai/blog (Run:AI)

Built-in arXiv categories:
- cs.AI, cs.LG, stat.ML, cs.CL, cs.IR

## Running continuously

A dedicated systemd unit is included: `phi4mini-backfill.service`.

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now phi4mini-backfill.service
sudo systemctl status phi4mini-backfill.service
```

Logs:
- `/var/lib/phi4mini/backfill.log`

Watch progress live:

```bash
tail -f /var/lib/phi4mini/backfill.log
```

To watch it live:

```bash
sudo tail -f /var/lib/phi4mini/backfill.log
```

Or via systemd/journald:

```bash
sudo journalctl -u phi4mini-backfill.service -f
```

## Tuning

You can override defaults by adding environment variables in:
`/etc/systemd/system/phi4mini-backfill.service`

Common knobs:
- `DISCOVER_EVERY_SECONDS` (default 21600 = 6h)
- `DISCOVER_MAX_NEW` (default 2000 per discover run)
- `ARXIV_MAX_NEW` (default 500 per discover run)
- `WORKER_LIMIT` (default 40 per worker cycle)
- `WORKER_CYCLE_SLEEP` (default 60s)
- `IDLE_SLEEP` (default 300s)

Worker etiquette:
- `WORKER_SLEEP` baseline sleep per fetch (default 1.0s)
- `DOMAIN_MIN_GAP` minimum gap per domain (default 1.0s)

## Optional: add/remove sources

### Add extra sites
Create `/var/lib/phi4mini/backfill_seeds.txt` with one URL per line, e.g.:

```text
https://your-favorite-lab.org/
https://example.com/blog/
```

### Disable built-in sources
Run discovery with `--no-default-sources` (the systemd unit uses built-ins by default).

## Optional per-domain rules

Create `/var/lib/phi4mini/backfill_rules.json` to constrain discovery (recommended for very large sites).

Example:

```json
{
  "default": {
    "exclude": ["\\?utm_", "/tag/", "/category/"],
    "max_new_per_run": 500
  },
  "microsoft.com": {
    "include": ["^https?://www\\.microsoft\\.com/en-us/research/blog/"],
    "max_new_per_run": 800
  }
}
```

If the file is missing, backfill runs with reasonable (built-in) defaults.
