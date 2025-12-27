#!/usr/bin/env python3
"""
backfill_discover.py

Continuously discover historical article URLs beyond RSS.

Discovery modes
1) Sitemaps (most websites)
   - robots.txt "Sitemap:" entries (preferred)
   - common sitemap locations (fallback)

2) arXiv API (deep history)
   - Cursor-based walking through older papers for selected categories.

Idempotency
- url_queue.url is PRIMARY KEY, so discovery is safe to run 24/7.

Safety
- Host-only filtering.
- Regex include/exclude rules per domain (defaults provided; user can override via JSON file).
- Budgets to avoid runaway discovery per run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import logging
import re
import sqlite3
import sys
import time
from typing import Iterable, Optional, Tuple, List, Dict
from urllib.parse import urlparse, urljoin, urlencode, quote_plus

import requests
import xml.etree.ElementTree as ET

import feedparser


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("phi4mini.backfill.discover")
    if logger.handlers:
        return logger
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(lvl)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] [backfill_discover] %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    logging.Formatter.converter = time.gmtime
    return logger

ASSET_RE = re.compile(r"\.(?:jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mp3|wav|zip|gz|tar|tgz|bz2|7z|rar|woff2?|ttf)$", re.I)

# ----------------------------
# Default always-on AI sources
# ----------------------------
# These are *seeds* to discover sitemaps and constrain rules; discovery is still respectful and
# bounded, and fetching obeys robots in the worker.
DEFAULT_SITEMAP_SOURCES: list[str] = [
    "https://openai.com/",              # OpenAI (robots includes sitemap) citeturn0search0
    "https://www.anthropic.com/",       # Anthropic (sitemap varies; will fallback) citeturn2search2
    "https://deepmind.google/",         # Google DeepMind
    "https://www.microsoft.com/",       # Microsoft (robots includes sitemap indexes) citeturn3search3
    "https://developer.nvidia.com/",    # NVIDIA Developer (robots includes sitemap index) citeturn2search3
    "https://www.nvidia.com/",          # NVIDIA main site (robots includes sitemap index) citeturn2search3
    "https://pytorch.org/",             # PyTorch blog lives here (sitemap varies)
    "https://huggingface.co/",          # Hugging Face blog (sitemap varies)
    "https://paperswithcode.com/",      # Papers with Code

    # ---- Added: more AI/ML historical sources (sitemap-based) ----
    "https://ai.meta.com/",             # Meta AI blog
    "https://cohere.com/",              # Cohere blog
    "https://stability.ai/",            # Stability AI updates
    "https://www.deeplearning.ai/",      # DeepLearning.AI (The Batch + blog)
    "https://blog.tensorflow.org/",      # TensorFlow blog
    "https://research.ibm.com/",         # IBM Research blog
    "https://aws.amazon.com/",           # AWS ML blog (under /blogs/machine-learning)
    "https://cloud.google.com/",         # Google Cloud AI/ML blog
    "https://research.google/",          # Google Research blog
    "https://www.databricks.com/",       # Databricks blog
    "https://wandb.ai/",                 # Weights & Biases (Fully Connected)
    "https://www.scale.com/",            # Scale AI blog
    "https://www.perplexity.ai/",        # Perplexity blog
    "https://www.assemblyai.com/",       # AssemblyAI blog
    "https://www.together.ai/",          # Together AI blog
    "https://www.ai21.com/",             # AI21 Labs blog
    "https://mistral.ai/",               # Mistral blog
    "https://www.adept.ai/",             # Adept blog
    "https://www.cerebras.net/",         # Cerebras blog
    "https://groq.com/",                 # Groq blog
    "https://www.run.ai/",               # Run:AI blog
]

DEFAULT_ARXIV_CATS: list[str] = [
    "cs.AI",
    "cs.LG",
    "stat.ML",
    "cs.CL",
    "cs.IR",
]

# Default rules to keep discovery focused on "articles" rather than whole sites.
# Users can override/extend via --rules JSON (same schema as earlier versions).
DEFAULT_RULES: dict = {
    "default": {
        "exclude": [
            r"\?.*utm_",
            r"\?ref=",
            r"#",
            r"/search",
            r"/tag/",
            r"/tags/",
            r"/category/",
            r"/categories/",
            r"/topics/",
            r"/topic/",
            r"/page/\d+/?$",
            r"/author/",
            r"/users/",
            r"/login",
            r"/signup",
            r"/privacy",
            r"/terms",
        ],
        "max_new_per_run": 2000,
    },
    "openai.com": {
        "include": [r"^https?://openai\.com/blog/"],
        "exclude": [r"/careers", r"/policies/"],
        "max_new_per_run": 2000,
    },
    "anthropic.com": {
        "include": [r"^https?://www\.anthropic\.com/(news|research|claude|transparency)"],
        "max_new_per_run": 1500,
    },
    "deepmind.google": {
        "include": [r"^https?://deepmind\.google/"],
        "exclude": [r"^https?://deepmind\.google/models/"],  # model cards can be heavy
        "max_new_per_run": 1200,
    },
    "microsoft.com": {
        "include": [r"^https?://www\.microsoft\.com/en-us/research/blog/"],
        "max_new_per_run": 1500,
    },
    "developer.nvidia.com": {
        "include": [r"^https?://developer\.nvidia\.com/blog/"],
        "max_new_per_run": 1500,
    },
    "nvidia.com": {
        "include": [r"^https?://www\.nvidia\.com/.*/blog/|^https?://www\.nvidia\.com/en-us/blog/"],
        "max_new_per_run": 1000,
    },
    "pytorch.org": {
        "include": [r"^https?://pytorch\.org/blog/"],
        "max_new_per_run": 1200,
    },
    "huggingface.co": {
        "include": [r"^https?://huggingface\.co/blog/"],
        "max_new_per_run": 1500,
    },
    "paperswithcode.com": {
        "include": [r"^https?://paperswithcode\.com/(paper|blog)"],
        "max_new_per_run": 1500,
    },

    # ---- Added sources rules (keep discovery focused) ----
    "ai.meta.com": {
        "include": [r"^https?://ai\.meta\.com/blog/"],
        "max_new_per_run": 1500,
    },
    "cohere.com": {
        "include": [r"^https?://cohere\.com/blog/", r"^https?://cohere\.com/research/"],
        "max_new_per_run": 1500,
    },
    "stability.ai": {
        "include": [r"^https?://stability\.ai/(blog|news|updates)"],
        "max_new_per_run": 1200,
    },
    "deeplearning.ai": {
        "include": [r"^https?://(www\.)?deeplearning\.ai/(the-batch|blog)/"],
        "max_new_per_run": 1500,
    },
    "blog.tensorflow.org": {
        "include": [r"^https?://blog\.tensorflow\.org/"],
        "max_new_per_run": 1200,
    },
    "research.ibm.com": {
        "include": [r"^https?://research\.ibm\.com/blog/"],
        "max_new_per_run": 1200,
    },
    "aws.amazon.com": {
        "include": [r"^https?://aws\.amazon\.com/blogs/machine-learning/"],
        "max_new_per_run": 1200,
    },
    "cloud.google.com": {
        "include": [r"^https?://cloud\.google\.com/blog/topics/ai-machine-learning", r"^https?://cloud\.google\.com/blog/products/ai-machine-learning"],
        "max_new_per_run": 1200,
    },
    "research.google": {
        "include": [r"^https?://research\.google/blog/"],
        "max_new_per_run": 1200,
    },
    "databricks.com": {
        "include": [r"^https?://(www\.)?databricks\.com/blog/"],
        "max_new_per_run": 1500,
    },
    "wandb.ai": {
        "include": [r"^https?://wandb\.ai/fully-connected/"],
        "max_new_per_run": 1500,
    },
    "scale.com": {
        "include": [r"^https?://(www\.)?scale\.com/blog/"],
        "max_new_per_run": 1200,
    },
    "perplexity.ai": {
        "include": [r"^https?://(www\.)?perplexity\.ai/blog/"],
        "max_new_per_run": 1200,
    },
    "assemblyai.com": {
        "include": [r"^https?://(www\.)?assemblyai\.com/blog/"],
        "max_new_per_run": 1200,
    },
    "together.ai": {
        "include": [r"^https?://(www\.)?together\.ai/blog/"],
        "max_new_per_run": 1200,
    },
    "ai21.com": {
        "include": [r"^https?://(www\.)?ai21\.com/blog/"],
        "max_new_per_run": 1200,
    },
    "mistral.ai": {
        "include": [r"^https?://mistral\.ai/(news|blog)/"],
        "max_new_per_run": 800,
    },
    "adept.ai": {
        "include": [r"^https?://(www\.)?adept\.ai/blog/"],
        "max_new_per_run": 800,
    },
    "cerebras.net": {
        "include": [r"^https?://(www\.)?cerebras\.net/blog/"],
        "max_new_per_run": 800,
    },
    "groq.com": {
        "include": [r"^https?://groq\.com/blog/"],
        "max_new_per_run": 800,
    },
    "run.ai": {
        "include": [r"^https?://(www\.)?run\.ai/blog/"],
        "max_new_per_run": 800,
    },
}

# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS url_queue (
            url TEXT PRIMARY KEY,
            source_domain TEXT NOT NULL,
            discovered_at_utc TEXT NOT NULL,
            lastmod TEXT,
            source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'NEW',
            tries INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            done_at_utc TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sitemap_state (
            sitemap_url TEXT PRIMARY KEY,
            etag TEXT,
            last_modified TEXT,
            last_crawl_utc TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS arxiv_state (
            cat TEXT PRIMARY KEY,
            start INTEGER NOT NULL,
            last_run_utc TEXT
        )
        """
    )
    conn.commit()


def load_rules(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[backfill_discover] rules file error: {e}", file=sys.stderr)
        return {}


def merged_rules(user_rules: dict) -> dict:
    # Shallow merge: domain-specific entries override defaults; default section merged too.
    out = dict(DEFAULT_RULES)
    if not user_rules:
        return out
    if "default" in user_rules:
        d0 = dict(out.get("default", {}))
        d1 = dict(user_rules.get("default", {}))
        # merge lists
        if "exclude" in d1 and "exclude" in d0:
            d0["exclude"] = list(dict.fromkeys(list(d0["exclude"]) + list(d1["exclude"])))
        if "include" in d1 and "include" in d0:
            d0["include"] = list(dict.fromkeys(list(d0["include"]) + list(d1["include"])))
        for k, v in d1.items():
            if k not in ("include", "exclude"):
                d0[k] = v
        out["default"] = d0
    for k, v in user_rules.items():
        if k == "default":
            continue
        out[k] = v
    return out


def domain_rules(rules: dict, domain: str) -> tuple[list[re.Pattern], list[re.Pattern], int]:
    d = rules.get(domain, {})
    dflt = rules.get("default", {})
    inc = [re.compile(p) for p in (d.get("include") or dflt.get("include") or [])]
    exc = [re.compile(p) for p in (d.get("exclude") or dflt.get("exclude") or [])]
    max_new = int(d.get("max_new_per_run") or dflt.get("max_new_per_run") or 2000)
    return inc, exc, max_new


def should_keep(url: str, domain: str, include: list[re.Pattern], exclude: list[re.Pattern]) -> bool:
    try:
        u = urlparse(url)
    except Exception:
        return False
    if u.scheme not in ("http", "https") or not u.netloc:
        return False
    # Host-only (including subdomains)
    if u.netloc.lower() != domain.lower() and not u.netloc.lower().endswith("." + domain.lower()):
        return False
    if ASSET_RE.search(u.path or ""):
        return False
    for p in exclude:
        if p.search(url):
            return False
    if include:
        return any(p.search(url) for p in include)
    return True


def get_robots_sitemaps(session: requests.Session, base: str, timeout: int) -> list[str]:
    sitemaps: list[str] = []
    robots = urljoin(base, "/robots.txt")
    try:
        r = session.get(robots, timeout=timeout)
        if r.status_code != 200:
            return []
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)
    except Exception:
        return []
    # de-dupe
    out = []
    seen = set()
    for sm in sitemaps:
        if sm not in seen:
            seen.add(sm)
            out.append(sm)
    return out


def get_common_sitemaps(base: str) -> list[str]:
    return [
        urljoin(base, "/sitemap.xml"),
        urljoin(base, "/sitemap_index.xml"),
        urljoin(base, "/sitemap.xml.gz"),
        urljoin(base, "/sitemap-index.xml"),
        urljoin(base, "/sitemap1.xml"),
    ]


def fetch_sitemap(session: requests.Session, conn: sqlite3.Connection, sitemap_url: str, timeout: int) -> Optional[bytes]:
    row = conn.execute(
        "SELECT etag, last_modified FROM sitemap_state WHERE sitemap_url=?",
        (sitemap_url,),
    ).fetchone()
    headers: dict[str, str] = {}
    if row:
        etag, last_modified = row
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

    r = session.get(sitemap_url, timeout=timeout, headers=headers, stream=True)
    if r.status_code == 304:
        conn.execute(
            "UPDATE sitemap_state SET last_crawl_utc=? WHERE sitemap_url=?",
            (utc_now_iso(), sitemap_url),
        )
        conn.commit()
        return None
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for sitemap {sitemap_url}")

    content = r.content
    etag = r.headers.get("ETag")
    lm = r.headers.get("Last-Modified")
    conn.execute(
        """
        INSERT INTO sitemap_state (sitemap_url, etag, last_modified, last_crawl_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(sitemap_url) DO UPDATE SET
            etag=excluded.etag,
            last_modified=excluded.last_modified,
            last_crawl_utc=excluded.last_crawl_utc
        """,
        (sitemap_url, etag, lm, utc_now_iso()),
    )
    conn.commit()
    return content


def parse_sitemap_bytes(b: bytes, sitemap_url: str) -> tuple[list[str], list[str]]:
    if sitemap_url.endswith(".gz") or b[:2] == b"\x1f\x8b":
        try:
            b = gzip.decompress(b)
        except Exception:
            pass

    try:
        root = ET.fromstring(b)
    except Exception as e:
        raise RuntimeError(f"XML parse error for {sitemap_url}: {e}")

    tag = root.tag.lower()
    if "}" in tag:
        tag = tag.split("}", 1)[1]

    nested: list[str] = []
    urls: list[str] = []
    if tag.endswith("sitemapindex"):
        for sm in root.findall(".//{*}sitemap"):
            loc = sm.find("{*}loc")
            if loc is not None and loc.text:
                nested.append(loc.text.strip())
    elif tag.endswith("urlset"):
        for u in root.findall(".//{*}url"):
            loc = u.find("{*}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    else:
        for loc in root.findall(".//{*}loc"):
            if loc.text:
                val = loc.text.strip()
                if val.endswith(".xml") or val.endswith(".xml.gz"):
                    nested.append(val)
                else:
                    urls.append(val)

    return nested, urls


def enqueue_urls(
    conn: sqlite3.Connection,
    urls: Iterable[str],
    domain: str,
    source: str,
    include: list[re.Pattern],
    exclude: list[re.Pattern],
    budget: int,
) -> int:
    inserted = 0
    now = utc_now_iso()
    for url in urls:
        if inserted >= budget:
            break
        if not should_keep(url, domain, include, exclude):
            continue
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO url_queue(url, source_domain, discovered_at_utc, source, status)
                VALUES (?, ?, ?, ?, 'NEW')
                """,
                (url, domain, now, source),
            )
            if conn.total_changes > 0:
                inserted += 1
        except Exception:
            continue
    conn.commit()
    return inserted


def discover_domain(
    session: requests.Session,
    conn: sqlite3.Connection,
    domain: str,
    timeout: int,
    rules: dict,
    max_total: int,
    max_sitemaps: int,
) -> int:
    include, exclude, max_new_for_domain = domain_rules(rules, domain)
    max_new = min(max_total, max_new_for_domain)
    base = f"https://{domain}"
    sitemaps = get_robots_sitemaps(session, base, timeout=timeout)
    if not sitemaps:
        sitemaps = get_common_sitemaps(base)

    q: list[str] = []
    seen_sm = set()
    for sm in sitemaps:
        if sm not in seen_sm:
            seen_sm.add(sm)
            q.append(sm)

    inserted_total = 0
    processed_sm = 0
    while q and inserted_total < max_new and processed_sm < max_sitemaps:
        sm_url = q.pop(0)
        processed_sm += 1
        try:
            blob = fetch_sitemap(session, conn, sm_url, timeout=timeout)
            if blob is None:
                continue
            nested, urls = parse_sitemap_bytes(blob, sm_url)

            remaining = max_new - inserted_total
            inserted_total += enqueue_urls(
                conn, urls, domain, source="sitemap", include=include, exclude=exclude, budget=remaining
            )
            for n in nested:
                if n not in seen_sm:
                    seen_sm.add(n)
                    q.append(n)
        except Exception as e:
            print(f"[backfill_discover] {domain}: sitemap error: {e}", file=sys.stderr)

    return inserted_total


def normalize_domain_from_url(u: str) -> Optional[str]:
    try:
        p = urlparse(u)
        if not p.netloc:
            return None
        d = p.netloc.lower()
        if d.startswith("www."):
            d = d[4:]
        return d
    except Exception:
        return None


def read_domains_from_feeds(feeds_file: str) -> list[str]:
    domains: list[str] = []
    seen = set()
    with open(feeds_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = normalize_domain_from_url(line)
            if d and d not in seen:
                seen.add(d)
                domains.append(d)
    return domains


def read_domains_from_seeds(seeds_file: str) -> list[str]:
    out: list[str] = []
    seen = set()
    if not seeds_file:
        return out
    try:
        with open(seeds_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                d = normalize_domain_from_url(line)
                if d and d not in seen:
                    seen.add(d)
                    out.append(d)
    except FileNotFoundError:
        return []
    return out


# ----------------------------
# arXiv API discovery (cursor-based)
# ----------------------------
ARXIV_API = "http://export.arxiv.org/api/query"


def arxiv_fetch(session: requests.Session, cat: str, start: int, max_results: int, timeout: int) -> list[str]:
    q = f"cat:{cat}"
    params = {
        "search_query": q,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    url = ARXIV_API + "?" + urlencode(params, quote_via=quote_plus)
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"arxiv_http_{r.status_code}")
    parsed = feedparser.parse(r.text)
    urls: list[str] = []
    for e in getattr(parsed, "entries", []) or []:
        link = (e.get("link") or "").strip()
        if link:
            urls.append(link)
    return urls


def arxiv_state_get(conn: sqlite3.Connection, cat: str) -> int:
    row = conn.execute("SELECT start FROM arxiv_state WHERE cat=?", (cat,)).fetchone()
    if not row:
        conn.execute("INSERT OR IGNORE INTO arxiv_state(cat, start, last_run_utc) VALUES (?, 0, ?)", (cat, utc_now_iso()))
        conn.commit()
        return 0
    return int(row[0])


def arxiv_state_set(conn: sqlite3.Connection, cat: str, start: int) -> None:
    conn.execute(
        "INSERT INTO arxiv_state(cat, start, last_run_utc) VALUES (?, ?, ?) "
        "ON CONFLICT(cat) DO UPDATE SET start=excluded.start, last_run_utc=excluded.last_run_utc",
        (cat, int(start), utc_now_iso()),
    )
    conn.commit()


def discover_arxiv(session: requests.Session, conn: sqlite3.Connection, cat: str, timeout: int, budget: int, batch: int) -> int:
    if budget <= 0:
        return 0
    start = arxiv_state_get(conn, cat)
    urls = arxiv_fetch(session, cat, start=start, max_results=batch, timeout=timeout)
    if not urls:
        return 0
    inserted = enqueue_urls(conn, urls, domain="arxiv.org", source=f"arxiv_api:{cat}", include=[], exclude=[], budget=budget)
    # Advance cursor by batch size (not inserted) to keep progressing through history.
    arxiv_state_set(conn, cat, start + batch)
    # arXiv asks for polite rate limits; keep it low.
    time.sleep(3.0)
    return inserted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds-file", required=True, help="path to feeds.txt (used to infer domains)")
    ap.add_argument("--db", required=True, help="sqlite db (shared with pipeline), e.g. /var/lib/phi4mini/seen.sqlite")
    ap.add_argument("--rules", default="", help="optional JSON rules file (domain include/exclude patterns)")
    ap.add_argument("--seeds-file", default="", help="optional seeds file (one URL per line) to add domains")
    ap.add_argument("--no-default-sources", action="store_true", help="do not include built-in AI sources")
    ap.add_argument("--max-new", type=int, default=1200, help="max new URLs to enqueue per run total (all modes)")
    ap.add_argument("--max-sitemaps", type=int, default=30, help="max sitemap files to fetch per domain per run")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--user-agent", default="phi4mini-backfill/1.1 (+local; respectful)", help="User-Agent")
    ap.add_argument("--log-level", default="INFO", help="logging level: DEBUG/INFO/WARNING/ERROR")

    ap.add_argument("--arxiv", action="store_true", help="enable arXiv API discovery (deep history)")
    ap.add_argument("--arxiv-cats", action="append", default=[], help="arXiv categories (repeatable), e.g. cs.AI")
    ap.add_argument("--arxiv-batch", type=int, default=100, help="arXiv API batch size per category per run")
    ap.add_argument("--arxiv-max-new", type=int, default=300, help="max new URLs enqueued from arXiv per run total")
    args = ap.parse_args()

    log = setup_logger(args.log_level)

    user_rules = load_rules(args.rules)
    rules = merged_rules(user_rules)

    conn = sqlite3.connect(args.db)
    ensure_db(conn)

    log.info("start feeds_file=%s db=%s max_new=%d", args.feeds_file, args.db, args.max_new)

    # domains from feeds + optional seeds + optional built-in list
    domains = read_domains_from_feeds(args.feeds_file)
    if args.seeds_file:
        domains += read_domains_from_seeds(args.seeds_file)
    if not args.no_default_sources:
        for u in DEFAULT_SITEMAP_SOURCES:
            d = normalize_domain_from_url(u)
            if d:
                domains.append(d)

    # de-dupe preserve order
    deduped: list[str] = []
    seen = set()
    for d in domains:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    domains = deduped

    if not domains and not args.arxiv:
        log.warning("no domains found (and arxiv disabled)")
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept": "application/xml,text/xml,*/*;q=0.8"})

    remaining = int(args.max_new)
    total_inserted = 0

    log.info("domains=%d arxiv=%s", len(domains), bool(args.arxiv))

    # 1) arXiv (deep history) — take a slice of the budget
    if args.arxiv:
        cats = args.arxiv_cats or list(DEFAULT_ARXIV_CATS)
        arxiv_budget = min(remaining, int(args.arxiv_max_new))
        for cat in cats:
            if arxiv_budget <= 0:
                break
            try:
                inserted = discover_arxiv(session, conn, cat=cat, timeout=args.timeout, budget=arxiv_budget, batch=args.arxiv_batch)
                log.info("arxiv cat=%s inserted=%d next_start=%d", cat, inserted, arxiv_state_get(conn, cat))
                total_inserted += inserted
                arxiv_budget -= inserted
                remaining = max(0, remaining - inserted)
            except Exception as e:
                log.warning("arxiv error cat=%s err=%s", cat, str(e))

    # 2) sitemaps
    for dom in domains:
        if remaining <= 0:
            break
        try:
            inserted = discover_domain(session, conn, dom, timeout=args.timeout, rules=rules, max_total=remaining, max_sitemaps=args.max_sitemaps)
            log.info("domain done dom=%s inserted=%d remaining=%d", dom, inserted, max(0, remaining - inserted))
            total_inserted += inserted
            remaining -= inserted
        except Exception as e:
            log.warning("domain error dom=%s err=%s", dom, str(e))
        time.sleep(1.0)

    log.info("summary enqueued_new_urls=%d domains=%d arxiv=%s", total_inserted, len(domains), bool(args.arxiv))
    print(f"[backfill_discover] enqueued_new_urls={total_inserted} domains={len(domains)} arxiv={bool(args.arxiv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
