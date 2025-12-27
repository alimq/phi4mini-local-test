#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, sys, time, sqlite3, logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dateutil import parser as dtparser

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("phi4mini.digest")
    if logger.handlers:
        return logger
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(lvl)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] [digest] %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    logging.Formatter.converter = time.gmtime
    return logger


# ----------------------------
# Cursor/state (SQLite)
# ----------------------------
def init_state_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT value FROM state WHERE key = ?", (key,))
    row = cur.fetchone()
    return None if row is None else str(row[0])


def state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()

FINAL_REPORT_TEMPLATE = """You are my research assistant.

Goal: Write a single, high-quality digest focused on:
- RAG (retrieval augmented generation), retrieval quality, reranking, chunking, evals
- Running SLM/LLMs locally, ollama, inference performance, quantization, CPU/GPU, deployment
- Practical experiments and actionable next steps

Inputs:
You will receive a set of short evidence snippets from recent articles. Some may be incomplete.

Required structure (use these headings):
# Long gist
(8–10 sentences, coherent and opinionated but careful)

# Trend map
(List 6–12 trends; each trend has 1–3 bullets and includes evidence links)

# Weekly watchlist
(5–10 items to monitor next week)

# Contradictions & open questions
(5–10 items; point out inconsistencies or uncertainties)

# Action plan
(5–10 action items with concrete experiments and pipeline improvements)

# Uncertainty notes
(Briefly note what’s missing due to limited context)

Evidence snippets (each has title, url, source, and a short text excerpt):
{EVIDENCE}
"""

CHUNK_TEMPLATE = """You are my research assistant.
Extract the most important facts, claims, or technical details from the snippets below.
Return a compact bullet list. Include links for each bullet (use the given URL).

Snippets:
{SNIPS}
"""


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clamp(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def parse_dt(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def iso_z(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    dt = dt.astimezone(timezone.utc)
    s = dt.isoformat()
    # Normalize +00:00 suffix to Z
    return s.replace("+00:00", "Z")



@dataclass
class Item:
    title: str
    url: str
    source: str
    published: Optional[datetime]
    fetched: Optional[datetime]
    text: str


def item_from_json(obj: dict) -> Item:
    title = obj.get("title") or ""
    url = obj.get("url") or obj.get("link") or ""
    source = obj.get("feed_title") or obj.get("feed_url") or obj.get("source") or ""
    published = parse_dt(obj.get("published") or "")
    fetched = parse_dt(
        obj.get("fetched_at_utc")
        or obj.get("fetched_at")
        or obj.get("fetched")
        or ""
    )
    text = obj.get("text") or obj.get("content") or ""
    text = strip_html(text)
    return Item(title=title, url=url, source=source, published=published, fetched=fetched, text=text)


def load_outbox_incremental(path: Path, start_offset: int, max_items: int) -> Tuple[List[Item], int]:
    """Read *new* JSONL entries from outbox.jsonl starting at byte offset.

    Returns (items, new_offset). If the file was truncated/rotated, start_offset is reset to 0.
    """
    items: List[Item] = []
    if not path.exists():
        return items, start_offset

    size = path.stat().st_size
    if start_offset < 0 or start_offset > size:
        start_offset = 0

    with path.open("rb") as bf:
        bf.seek(start_offset)
        while max_items <= 0 or len(items) < max_items:
            line = bf.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            items.append(item_from_json(obj))
        new_offset = bf.tell()

    return items, new_offset


def filter_items(items: List[Item], days: int) -> List[Item]:
    if days <= 0:
        return items
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: List[Item] = []
    for it in items:
        dt = it.published or it.fetched
        if dt is None:
            out.append(it)
        else:
            if dt >= cutoff:
                out.append(it)
    return out


def score_relevance(text: str) -> int:
    t = text.lower()
    keywords = [
        "rag", "retrieval", "rerank", "reranker", "embedding", "vector", "chunk", "chunking", "context window",
        "ollama", "llama.cpp", "quant", "quantization", "int4", "int8", "gguf", "inference", "cpu", "gpu",
        "throughput", "latency", "tokens/sec", "kvcache", "kv cache", "serving", "deployment", "agent",
        "graph rag", "graphrag", "neo4j",
    ]
    score = 0
    for kw in keywords:
        if kw in t:
            score += 1
    return score


def select_items(items: List[Item], last_n: int, max_items: int) -> List[Item]:
    if last_n > 0:
        items = items[-last_n:]
    if max_items > 0 and len(items) > max_items:
        scored = [(score_relevance(it.title + " " + it.text), i, it) for i, it in enumerate(items)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        items = [it for _, _, it in scored[:max_items]]
    return items


def make_snippets(items: List[Item], snippet_chars: int) -> List[str]:
    snips: List[str] = []
    for it in items:
        dt = it.published or it.fetched
        dt_s = dt.isoformat() if dt else "unknown-date"
        txt = clamp(it.text, snippet_chars)
        snips.append(f"- {it.title}\n  url: {it.url}\n  source: {it.source}\n  date: {dt_s}\n  text: {txt}")
    return snips


def chunk_list(xs: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        return [xs]
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]


def call_ollama_stream(
    prompt: str,
    model: str,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    debug: bool,
    connect_timeout_s: int,
    read_timeout_s: int,
    first_token_timeout_s: int,
    stream_to_path: Optional[Path] = None,
    stream_mode: str = "a",
    label: str = "",
) -> Tuple[dict, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        "keep_alive": keep_alive,
    }

    t0 = time.time()
    first_token_t = None
    out_text = ""

    f = None
    if stream_to_path is not None:
        f = stream_to_path.open(stream_mode, encoding="utf-8")

    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(connect_timeout_s, read_timeout_s),
        )
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if "response" in obj:
                token = obj["response"]
                if first_token_t is None and token:
                    first_token_t = time.time()
                out_text += token
                if f is not None:
                    f.write(token)
                    f.flush()

            if first_token_t is None:
                if time.time() - t0 > first_token_timeout_s:
                    raise TimeoutError(f"[{label}] timed out waiting for first token (> {first_token_timeout_s}s)")

            if obj.get("done"):
                break

    finally:
        if f is not None:
            f.flush()
            f.close()

    meta = {
        "label": label,
        "t_total_s": round(time.time() - t0, 3),
        "t_first_token_s": None if first_token_t is None else round(first_token_t - t0, 3),
        "len_out": len(out_text),
    }
    if debug:
        eprint(f"[debug] {meta}")
    return meta, out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbox", default="outbox.jsonl", help="path to outbox.jsonl")
    ap.add_argument("--state-db", default="", help="SQLite DB path for cursor/state (default: use outbox dir/state.sqlite)")
    ap.add_argument("--cursor-key", default="outbox_offset_bytes", help="state key to store outbox byte offset")
    ap.add_argument("--batch-items", type=int, default=120, help="max NEW outbox items to consume this run (0=all)")
    ap.add_argument("--days", type=int, default=0, help="optional filter: keep items published/fetched within N days (0=all)")
    ap.add_argument("--max-items", type=int, default=120, help="max items to include after relevance selection (0=all)")
    ap.add_argument("--last-n", type=int, default=0, help="only consider last N of this run's batch (0=all)")
    ap.add_argument("--snippet-chars", type=int, default=260, help="chars per item snippet")
    ap.add_argument("--max-prompt-chars", type=int, default=6500, help="cap evidence text sent to the model")
    ap.add_argument("--chunk-size", type=int, default=30, help="snippets per chunk for chunk pass")
    ap.add_argument("--model", default="phi4-mini:latest", help="ollama model name")
    ap.add_argument("--num-ctx", type=int, default=4096, help="context window")
    ap.add_argument("--num-predict-chunk", type=int, default=220, help="max tokens for chunk pass")
    ap.add_argument("--num-predict-final", type=int, default=900, help="max tokens for final report")
    ap.add_argument("--keep-alive", default="10m", help="ollama keep_alive value")
    ap.add_argument("--connect-timeout", type=int, default=10, help="requests connect timeout (sec)")
    ap.add_argument("--read-timeout", type=int, default=1800, help="requests read timeout between chunks (sec)")
    ap.add_argument("--first-token-timeout", type=int, default=45, help="timeout waiting for first token (sec)")
    ap.add_argument("--log-level", default="INFO", help="logging level: DEBUG/INFO/WARNING/ERROR")
    ap.add_argument("--debug", action="store_true", help="extra debug for ollama streaming (to stderr)")
    args = ap.parse_args()

    log = setup_logger(args.log_level)

    outbox_path = Path(args.outbox)

    # Cursor/state DB
    if args.state_db:
        state_db_path = Path(args.state_db)
    else:
        state_db_path = outbox_path.parent / "state.sqlite"
    conn = init_state_db(state_db_path)

    raw_offset = state_get(conn, args.cursor_key) or "0"
    try:
        start_offset = int(raw_offset)
    except Exception:
        start_offset = 0

    log.info("cursor load key=%s start_offset=%d", args.cursor_key, start_offset)

    # Load NEW items since last offset, then (optionally) filter/select.
    items, new_offset = load_outbox_incremental(outbox_path, start_offset=start_offset, max_items=args.batch_items)
    log.info("outbox read path=%s new_offset=%d batch_items=%d got_items=%d", str(outbox_path), new_offset, args.batch_items, len(items))
    before_f = len(items)
    items = filter_items(items, args.days)
    after_f = len(items)
    items = select_items(items, args.last_n, args.max_items)
    log.info("filters days=%d before=%d after_days=%d selected=%d", args.days, before_f, after_f, len(items))

    # Sort by datetime for better coherence.
    items.sort(key=lambda it: (it.published or it.fetched or datetime.min.replace(tzinfo=timezone.utc)))

    if not items:
        log.info("no new items; cursor unchanged")
        return 0

    snips = make_snippets(items, args.snippet_chars)
    chunks = chunk_list(snips, args.chunk_size)

    # Stage 1: chunk pass in-memory
    chunk_bullets: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = CHUNK_TEMPLATE.format(SNIPS="\n\n".join(chunk))
        if len(prompt) > args.max_prompt_chars:
            prompt = prompt[: args.max_prompt_chars]

        log.info("chunk start i=%d/%d prompt_chars=%d", i, len(chunks), len(prompt))

        _, out = call_ollama_stream(
            prompt=prompt,
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict_chunk,
            keep_alive=args.keep_alive,
            debug=args.debug,
            connect_timeout_s=args.connect_timeout,
            read_timeout_s=args.read_timeout,
            first_token_timeout_s=args.first_token_timeout,
            stream_to_path=None,
            stream_mode="a",
            label=f"chunk-{i}",
        )
        chunk_bullets.append(out.strip())
        log.info("chunk done i=%d/%d out_chars=%d", i, len(chunks), len(out or ""))

    # Stage 2: final report appended to digest.md in current working directory
    evidence = "\n\n".join([b for b in chunk_bullets if b])
    final_prompt = FINAL_REPORT_TEMPLATE.format(EVIDENCE=evidence)
    if len(final_prompt) > args.max_prompt_chars:
        final_prompt = final_prompt[: args.max_prompt_chars]

    digest_path = Path("digest.md")
    digest_path.parent.mkdir(parents=True, exist_ok=True)

    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H:%M:%S")

    # Detect whether today's header already exists near the end of the file
    tail = ""
    if digest_path.exists():
        try:
            with digest_path.open("rb") as bf:
                bf.seek(0, 2)
                size = bf.tell()
                bf.seek(max(0, size - 8192))
                tail = bf.read().decode("utf-8", errors="ignore")
        except Exception:
            tail = ""
    need_day_header = f"# {day}" not in tail
    with digest_path.open("a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        if need_day_header:
            f.write(f"# {day}\n\n")
        f.write(f"## Run {day} {ts}\n\n")
        cov_start = items[0].published or items[0].fetched
        cov_end = items[-1].published or items[-1].fetched
        f.write(f"**Coverage:** {iso_z(cov_start)} → {iso_z(cov_end)} (UTC) · **Items:** {len(items)}\n\n")

    _, _ = call_ollama_stream(
        prompt=final_prompt,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict_final,
        keep_alive=args.keep_alive,
        debug=args.debug,
        connect_timeout_s=args.connect_timeout,
        read_timeout_s=args.read_timeout,
        first_token_timeout_s=args.first_token_timeout,
        label="final",
        stream_to_path=digest_path,
        stream_mode="a",
    )

    with digest_path.open("a", encoding="utf-8") as f:
        f.write("\n\n")

    log.info("digest written path=%s", str(digest_path))

    # Only advance cursor after successful completion.
    state_set(conn, args.cursor_key, str(new_offset))
    log.info("cursor advance key=%s new_offset=%d", args.cursor_key, new_offset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
