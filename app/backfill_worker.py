#!/usr/bin/env python3
"""
backfill_worker.py

Consumes url_queue (discovered via backfill_discover.py), fetches pages,
extracts main text (trafilatura), and appends normalized records to outbox.jsonl.

Idempotency:
- url_queue.url is PRIMARY KEY (discovery doesn't duplicate)
- seen.id prevents writing same URL multiple times into outbox

Safety / etiquette:
- Bounded download size + optional extracted-text clamp
- Optional robots.txt enforcement (default: ON)
- Best-effort Crawl-delay handling + per-domain throttle
- Skips PDFs by default (they tend to be huge and noisy)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Optional, Tuple, Dict
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import requests
import trafilatura


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("phi4mini.backfill.worker")
    if logger.handlers:
        return logger
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(lvl)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] [backfill_worker] %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    logging.Formatter.converter = time.gmtime
    return logger


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            first_seen_utc TEXT NOT NULL
        )
        """
    )
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
    conn.commit()


def claim_next(conn: sqlite3.Connection) -> Optional[Tuple[str, str]]:
    row = conn.execute(
        """
        SELECT url, source_domain
        FROM url_queue
        WHERE status='NEW'
        ORDER BY discovered_at_utc ASC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    url, domain = row
    conn.execute("UPDATE url_queue SET status='FETCHING' WHERE url=? AND status='NEW'", (url,))
    conn.commit()
    row2 = conn.execute("SELECT status FROM url_queue WHERE url=?", (url,)).fetchone()
    if not row2 or row2[0] != "FETCHING":
        return None
    return url, domain


def is_pdf(url: str, content_type: str) -> bool:
    if "application/pdf" in (content_type or "").lower():
        return True
    return url.lower().split("?", 1)[0].endswith(".pdf")


def clamp_text(text: str, max_chars: int) -> tuple[str, bool, int]:
    if max_chars <= 0:
        return text, False, len(text)
    orig = len(text)
    if orig <= max_chars:
        return text, False, orig
    # keep head + tail to preserve intro & conclusions
    tail_len = min(5000, max_chars // 3)
    head_len = max_chars - tail_len
    out = text[:head_len] + "\n\n[...TRUNCATED...]\n\n" + text[-tail_len:]
    return out, True, orig


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# Robots.txt (best effort)
# ----------------------------
class RobotsCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[RobotFileParser, Optional[float], float]] = {}
        # netloc -> (rp, crawl_delay, fetched_at_monotonic)

    def _parse_crawl_delay(self, robots_txt: str, user_agent: str) -> Optional[float]:
        # Very small parser: prefers exact UA section, else '*' section.
        ua = user_agent.lower()
        lines = [ln.strip() for ln in robots_txt.splitlines()]
        sections: list[tuple[list[str], list[str]]] = []
        cur_uas: list[str] = []
        cur_rules: list[str] = []
        for ln in lines:
            if not ln or ln.startswith("#"):
                continue
            if ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "user-agent":
                if cur_uas or cur_rules:
                    sections.append((cur_uas, cur_rules))
                cur_uas = [v.lower()]
                cur_rules = []
            else:
                cur_rules.append(f"{k}:{v}")
        if cur_uas or cur_rules:
            sections.append((cur_uas, cur_rules))

        def find_delay(match_ua: str) -> Optional[float]:
            for uas, rules in sections:
                if match_ua in uas:
                    for r in rules:
                        if r.startswith("crawl-delay:"):
                            try:
                                return float(r.split(":", 1)[1].strip())
                            except Exception:
                                return None
            return None

        # Try exact UA first, then '*'
        return find_delay(ua) or find_delay("*")

    def get(self, session: requests.Session, base_url: str, user_agent: str, timeout: int, max_age_s: int) -> Tuple[RobotFileParser, Optional[float]]:
        p = urlparse(base_url)
        netloc = p.netloc.lower()
        now = time.monotonic()
        cached = self._cache.get(netloc)
        if cached and (now - cached[2]) < max_age_s:
            return cached[0], cached[1]

        robots_url = urljoin(base_url, "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)

        crawl_delay = None
        try:
            r = session.get(robots_url, timeout=timeout, headers={"User-Agent": user_agent})
            if 200 <= r.status_code < 300:
                rp.parse(r.text.splitlines())
                crawl_delay = self._parse_crawl_delay(r.text, user_agent=user_agent)
            else:
                rp.parse([])  # unknown -> allow
        except Exception:
            rp.parse([])  # unknown -> allow

        self._cache[netloc] = (rp, crawl_delay, now)
        return rp, crawl_delay


def fetch_extract(
    session: requests.Session,
    url: str,
    timeout: int,
    max_bytes: int,
    max_chars: int,
    allow_pdf: bool,
) -> dict:
    r = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    ct = r.headers.get("Content-Type", "")
    final_url = r.url

    if is_pdf(final_url, ct) and not allow_pdf:
        raise RuntimeError("skip_pdf")

    chunks = []
    total = 0
    for chunk in r.iter_content(chunk_size=65536):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if max_bytes > 0 and total > max_bytes:
            raise RuntimeError(f"download_too_large>{max_bytes}")
    raw = b"".join(chunks)

    try:
        html = raw.decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    ) or ""

    meta = None
    try:
        meta = trafilatura.metadata.extract_metadata(html)  # type: ignore[attr-defined]
    except Exception:
        meta = None

    title = ""
    date = ""
    if meta:
        try:
            title = meta.title or ""
            date = meta.date or ""
        except Exception:
            pass

    extracted, truncated, orig_len = clamp_text(extracted, max_chars=max_chars)

    return {
        "url": final_url,
        "title": title.strip(),
        "published_utc": (date or "").strip(),
        "text": extracted.strip(),
        "text_truncated": truncated,
        "original_text_len": orig_len,
        "content_type": ct,
        "downloaded_bytes": len(raw),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sqlite db shared with pipeline, e.g. /var/lib/phi4mini/seen.sqlite")
    ap.add_argument("--outbox", required=True, help="outbox.jsonl to append to, e.g. /var/lib/phi4mini/outbox.jsonl")
    ap.add_argument("--limit", type=int, default=50, help="max URLs to process per run")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=1.0, help="baseline sleep between fetches (seconds)")
    ap.add_argument("--max-bytes", type=int, default=1_500_000, help="max bytes per download (0=unlimited)")
    ap.add_argument("--max-chars", type=int, default=50_000, help="max extracted chars per article (0=unlimited)")
    ap.add_argument("--allow-pdf", action="store_true", help="allow PDFs (not recommended)")
    ap.add_argument("--user-agent", default="phi4mini-backfill/1.1 (+local; respectful)", help="User-Agent")
    ap.add_argument("--no-robots", action="store_true", help="disable robots.txt checks (not recommended)")
    ap.add_argument("--robots-max-age", type=int, default=6 * 3600, help="cache robots.txt for N seconds")
    ap.add_argument("--domain-min-gap", type=float, default=1.0, help="minimum gap between requests to the same domain")
    ap.add_argument("--log-level", default="INFO", help="logging level: DEBUG/INFO/WARNING/ERROR")
    args = ap.parse_args()

    log = setup_logger(args.log_level)

    conn = sqlite3.connect(args.db)
    ensure_db(conn)

    session = requests.Session()
    session.headers.update({
        "User-Agent": args.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    robots = RobotsCache()
    last_fetch: Dict[str, float] = {}  # netloc -> monotonic timestamp

    processed = 0
    while processed < args.limit:
        claim = claim_next(conn)
        if not claim:
            break
        url, _source_domain = claim

        log.info("claim url=%s", url)

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        netloc = parsed.netloc.lower()

        # Respect robots.txt (best effort)
        if not args.no_robots and netloc:
            rp, crawl_delay = robots.get(
                session=session,
                base_url=base,
                user_agent=args.user_agent,
                timeout=min(args.timeout, 20),
                max_age_s=max(60, args.robots_max_age),
            )
            if not rp.can_fetch(args.user_agent, url):
                log.info("skip robots url=%s", url)
                conn.execute(
                    "UPDATE url_queue SET status='SKIPPED_ROBOTS', last_error=?, done_at_utc=? WHERE url=?",
                    ("disallowed_by_robots", utc_now_iso(), url),
                )
                conn.commit()
                processed += 1
                continue
        else:
            crawl_delay = None

        # Per-domain throttle (baseline gap + crawl-delay)
        now = time.monotonic()
        gap = max(args.domain_min_gap, args.sleep, float(crawl_delay or 0.0))
        last = last_fetch.get(netloc, 0.0)
        to_sleep = (last + gap) - now
        if to_sleep > 0:
            log.debug("throttle domain=%s sleep=%.2fs", netloc, to_sleep)
            time.sleep(to_sleep)

        seen_id = "url:" + sha256_hex(url)

        # idempotency: if seen, mark DONE and skip
        if conn.execute("SELECT 1 FROM seen WHERE id=?", (seen_id,)).fetchone():
            log.info("skip seen url=%s", url)
            conn.execute("UPDATE url_queue SET status='DONE', done_at_utc=? WHERE url=?", (utc_now_iso(), url))
            conn.commit()
            processed += 1
            continue

        try:
            t0 = time.monotonic()
            rec = fetch_extract(
                session,
                url,
                timeout=args.timeout,
                max_bytes=args.max_bytes,
                max_chars=args.max_chars,
                allow_pdf=args.allow_pdf,
            )
            text = rec.get("text", "")
            if not text:
                raise RuntimeError("empty_extract")

            dt_s = time.monotonic() - t0
            log.info(
                "fetched ok url=%s bytes=%s text_chars=%d truncated=%s secs=%.2f",
                rec.get("url") or url,
                rec.get("downloaded_bytes", "?"),
                len(text),
                bool(rec.get("text_truncated", False)),
                dt_s,
            )

            now_iso = utc_now_iso()
            out = {
                "id": seen_id,
                "feed_title": f"Backfill ({netloc or 'unknown'})",
                "feed_url": f"backfill:{netloc or 'unknown'}",
                "title": rec.get("title") or "",
                "url": rec.get("url") or url,
                "published_utc": rec.get("published_utc") or "",
                "fetched_at_utc": now_iso,
                "text": text,
                "text_truncated": rec.get("text_truncated", False),
                "original_text_len": rec.get("original_text_len", len(text)),
                "content_type": rec.get("content_type", ""),
                "source": "backfill",
            }

            conn.execute("INSERT OR IGNORE INTO seen(id, first_seen_utc) VALUES (?, ?)", (seen_id, now_iso))
            conn.execute("UPDATE url_queue SET status='DONE', done_at_utc=? WHERE url=?", (now_iso, url))
            conn.commit()

            append_jsonl(args.outbox, out)
            last_fetch[netloc] = time.monotonic()
            processed += 1
        except Exception as e:
            err = str(e)[:500]
            log.warning("fetched error url=%s err=%s", url, err)
            conn.execute("UPDATE url_queue SET tries=tries+1, last_error=? WHERE url=?", (err, url))
            tries_row = conn.execute("SELECT tries FROM url_queue WHERE url=?", (url,)).fetchone()
            tries = int(tries_row[0]) if tries_row else 1

            if str(e) == "skip_pdf":
                conn.execute("UPDATE url_queue SET status='SKIPPED_PDF', done_at_utc=? WHERE url=?", (utc_now_iso(), url))
            elif tries >= 5:
                conn.execute("UPDATE url_queue SET status='FAILED', done_at_utc=? WHERE url=?", (utc_now_iso(), url))
            else:
                conn.execute("UPDATE url_queue SET status='NEW' WHERE url=?", (url,))
            conn.commit()
            # small backoff
            time.sleep(min(10.0, max(args.sleep, 1.0)))

    log.info("summary processed=%d", processed)
    print(f"[backfill_worker] processed={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
