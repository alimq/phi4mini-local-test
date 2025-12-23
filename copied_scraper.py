#!/usr/bin/env python3
"""
rss2text.py

Fetch RSS/Atom feeds, extract usable text, and (if an entry has a link) fetch the
linked article and extract its main text for LLM input (e.g., phi4-mini).

This version fixes macOS SSL cert issues by using requests + certifi for ALL HTTPS
fetches (feeds and articles), then passing bytes to feedparser.

Outputs one JSON object per entry (JSONL).

Features:
- RSS/Atom parsing (feedparser)
- "seen" tracking in local SQLite (cron-friendly)
- If feed item contains summary/content, uses it
- If item has link, fetches article and extracts main text
- Best-effort robots.txt support + rate limiting
- Extraction: trafilatura (best) → readability-lxml → BeautifulSoup fallback

Install:
  python3 -m pip install -U feedparser requests certifi trafilatura
or:
  python3 -m pip install -U feedparser requests certifi readability-lxml beautifulsoup4 lxml
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import certifi
import feedparser
import requests

# Optional extractors
HAVE_TRAFILATURA = False
HAVE_READABILITY = False
HAVE_BS4 = False

try:
    import trafilatura  # type: ignore

    HAVE_TRAFILATURA = True
except Exception:
    pass

try:
    from readability import Document  # type: ignore

    HAVE_READABILITY = True
except Exception:
    pass

try:
    from bs4 import BeautifulSoup  # type: ignore

    HAVE_BS4 = True
except Exception:
    pass


USER_AGENT = "rss2text/1.1 (+local script; respectful polling; no redistribution)"


# ----------------------------
# Persistence: seen items DB
# ----------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            first_seen_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def is_seen(conn: sqlite3.Connection, item_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM seen WHERE id = ?", (item_id,))
    return cur.fetchone() is not None


def mark_seen(conn: sqlite3.Connection, item_id: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO seen (id, first_seen_utc) VALUES (?, ?)",
        (item_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


# ----------------------------
# Robots.txt (best effort)
# ----------------------------
_robot_cache = {}


def allowed_by_robots(
    url: str, session: requests.Session, user_agent: str = USER_AGENT, timeout: int = 10
) -> bool:
    """
    Best-effort robots check. If robots is unavailable, default allow.
    """
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base in _robot_cache:
            rp = _robot_cache[base]
        else:
            robots_url = f"{base}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)

            r = session.get(
                robots_url,
                headers={"User-Agent": user_agent},
                timeout=timeout,
                verify=certifi.where(),
            )
            if 200 <= r.status_code < 300:
                rp.parse(r.text.splitlines())
            else:
                rp.parse([])  # unknown -> allow
            _robot_cache[base] = rp

        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


# ----------------------------
# Text extraction
# ----------------------------
def normalize_whitespace(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def extract_text_from_html(html: str, url: str = "") -> str:
    # Best: trafilatura
    if HAVE_TRAFILATURA:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        if extracted and extracted.strip():
            return normalize_whitespace(extracted)

    # Next: readability-lxml + BeautifulSoup
    if HAVE_READABILITY and HAVE_BS4:
        doc = Document(html)
        main_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(main_html, "lxml")
        text = soup.get_text("\n")
        if text and text.strip():
            return normalize_whitespace(text)

    # Fallback: BeautifulSoup full-page text
    if HAVE_BS4:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav", "aside"]):
            tag.decompose()
        return normalize_whitespace(soup.get_text("\n"))

    # Last resort: raw
    return normalize_whitespace(html)


def fetch_article_text(
    url: str,
    session: requests.Session,
    timeout: int,
    max_bytes: int,
    respect_robots: bool,
) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, "no_url"

    if respect_robots and not allowed_by_robots(url, session=session):
        return None, "blocked_by_robots"

    try:
        r = session.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            verify=certifi.where(),
        )
        if r.status_code != 200:
            return None, f"http_{r.status_code}"

        content = r.content
        if len(content) > max_bytes:
            content = content[:max_bytes]

        encoding = r.encoding or "utf-8"
        html = content.decode(encoding, errors="replace")

        text = extract_text_from_html(html, url=url)
        if not text:
            return None, "extraction_empty"
        return text, None
    except Exception as e:
        return None, f"fetch_error:{type(e).__name__}"


# ----------------------------
# Feed parsing helpers
# ----------------------------
def stable_item_id(entry: dict) -> str:
    for key in ("id", "guid", "link"):
        v = entry.get(key)
        if v:
            return str(v)

    title = entry.get("title", "")
    link = entry.get("link", "")
    published = entry.get("published", "") or entry.get("updated", "")
    raw = f"{title}\n{link}\n{published}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def entry_text_from_feed(entry: dict) -> str:
    parts = []
    summary = entry.get("summary") or entry.get("description")
    if summary:
        parts.append(summary)

    content = entry.get("content")
    if isinstance(content, list):
        for c in content:
            val = c.get("value")
            if val:
                parts.append(val)

    return normalize_whitespace("\n\n".join(parts))


def fetch_and_parse_feed(
    feed_url: str, session: requests.Session, timeout: int
) -> feedparser.FeedParserDict:
    """
    Fetch feed XML with requests+certifi, then parse bytes with feedparser.
    This avoids Python SSL CA issues on macOS.
    """
    r = session.get(
        feed_url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        verify=certifi.where(),
    )
    r.raise_for_status()
    return feedparser.parse(r.content)


# ----------------------------
# Output
# ----------------------------
def write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", action="append", default=[], help="RSS/Atom feed URL (repeatable)")
    ap.add_argument("--feeds-file", default="", help="Text file with one feed URL per line")
    ap.add_argument("--out", default="outbox.jsonl", help="Output JSONL path")
    ap.add_argument("--db", default="seen.sqlite", help="SQLite DB path for seen items")
    ap.add_argument("--max-items", type=int, default=20, help="Max items per feed per run")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between article fetches")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bytes per article download")
    ap.add_argument("--no-robots", action="store_true", help="Disable robots.txt checks (not recommended)")
    ap.add_argument("--no-article-fetch", action="store_true", help="Do not fetch linked articles; use feed text only")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    feeds = list(args.feed)
    if args.feeds_file:
        with open(args.feeds_file, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    feeds.append(u)

    if not feeds:
        print("No feeds provided. Use --feed URL or --feeds-file feeds.txt", file=sys.stderr)
        return 2

    # Sanity: at least one extractor available
    if not (HAVE_TRAFILATURA or (HAVE_READABILITY and HAVE_BS4) or HAVE_BS4):
        print(
            "Warning: No HTML extractor installed. Install one of:\n"
            "  pip install trafilatura\n"
            "or:\n"
            "  pip install readability-lxml beautifulsoup4 lxml",
            file=sys.stderr,
        )

    conn = init_db(args.db)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    now_utc = datetime.now(timezone.utc).isoformat()
    respect_robots = not args.no_robots

    for feed_url in feeds:
        try:
            parsed = fetch_and_parse_feed(feed_url, session=session, timeout=args.timeout)
        except Exception as e:
            print(f"[ERROR] feed fetch failed: {feed_url} ({type(e).__name__}: {e})", file=sys.stderr)
            continue

        feed_title = (parsed.feed.get("title") or "").strip()
        entries = parsed.entries[: args.max_items]

        if args.debug:
            print(f"\nFEED: {feed_url}")
            print(f"  Title: {feed_title}")
            print(f"  Entries parsed: {len(parsed.entries)} (processing {len(entries)})")
            if getattr(parsed, "bozo", 0):
                print("  bozo_exception:", repr(getattr(parsed, "bozo_exception", None)))

        for entry in entries:
            item_id = stable_item_id(entry)
            if is_seen(conn, item_id):
                continue

            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            published = (entry.get("published") or entry.get("updated") or "").strip()

            feed_text = entry_text_from_feed(entry)
            article_text = None
            article_err = None

            if (not args.no_article_fetch) and link:
                if args.debug:
                    print(f"  ITEM: {title[:80]!r}")
                    print(f"    link={link}")
                article_text, article_err = fetch_article_text(
                    link,
                    session=session,
                    timeout=args.timeout,
                    max_bytes=args.max_bytes,
                    respect_robots=respect_robots,
                )
                if args.debug:
                    if article_err:
                        print(f"    article_fetch_error={article_err}")
                    else:
                        print(f"    article_text_chars={len(article_text or '')}")
                time.sleep(args.sleep)

            final_text = article_text or feed_text

            obj = {
                "fetched_at_utc": now_utc,
                "feed_url": feed_url,
                "feed_title": feed_title,
                "id": item_id,
                "title": title,
                "url": link,
                "published": published,
                "text_source": "article" if article_text else ("feed" if feed_text else "empty"),
                "article_fetch_error": article_err,
                "text": final_text,
            }

            write_jsonl(args.out, obj)
            mark_seen(conn, item_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
