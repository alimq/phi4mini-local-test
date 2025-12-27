#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from app.rag_db import (
    init_state_db,
    insert_chunk,
    get_chunk_id,
    setup_logger,
    state_get,
    state_set,
    update_chunk_fts,
)

OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}
SHORT_TOKEN_ALLOWLIST = {"rag", "llm", "ai", "nlp", "ml"}


def strip_html_basic(text: str) -> str:
    # Outbox text is already stripped in feed_into_phi.py, but keep a tiny fallback.
    return " ".join((text or "").replace("\n", " ").split()).strip()


def iter_jsonl_new(path: Path, start_offset: int, max_items: int) -> Tuple[List[Dict], int]:
    items: List[Dict] = []
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
            items.append(obj)
        new_offset = bf.tell()
    return items, new_offset


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Prefer paragraph-based chunking
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not paras:
        paras = [text]

    chunks: List[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "\n" + p
        else:
            chunks.append(cur)
            # overlap from end of previous
            if overlap > 0 and len(cur) > overlap:
                tail = cur[-overlap:]
                cur = tail + "\n" + p
            else:
                cur = p
    if cur:
        chunks.append(cur)

    # Hard-split any very long chunk
    out: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
        else:
            for i in range(0, len(c), max_chars):
                out.append(c[i : i + max_chars])
    return out


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]


def _keyword_tags(text: str, max_tags: int = 12) -> List[str]:
    tokens = [
        t
        for t in _tokenize(text)
        if t not in STOPWORDS and (len(t) >= 4 or t in SHORT_TOKEN_ALLOWLIST)
    ]
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:max_tags]]


def embed(text: str, model: str, timeout_s: int = 60) -> Optional[List[float]]:
    payload = {"model": model, "prompt": text}
    try:
        r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return None
        data = r.json()
        emb = data.get("embedding")
        if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
            return [float(x) for x in emb]
        return None
    except Exception:
        return None


def stable_doc_id(obj: Dict) -> str:
    # Prefer the feed's stable id; fall back to URL; last resort: hash title+text.
    if obj.get("id"):
        return str(obj.get("id"))
    if obj.get("url") or obj.get("link"):
        return str(obj.get("url") or obj.get("link"))
    raw = f"{obj.get('title','')}\n{obj.get('text','')}"
    return str(abs(hash(raw)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally index outbox.jsonl into rag.sqlite")
    ap.add_argument("--outbox", default="/var/lib/phi4mini/outbox.jsonl")
    ap.add_argument("--db", default="/var/lib/phi4mini/rag.sqlite", help="RAG sqlite path")
    ap.add_argument("--state-db", default="", help="cursor/state sqlite path (default: same as --db)")
    ap.add_argument("--cursor-key", default="outbox_offset_bytes")
    ap.add_argument("--batch-items", type=int, default=200, help="max NEW items to consume per inner loop")
    ap.add_argument("--max-seconds", type=int, default=900, help="stop after this many seconds (used for the 15min window)")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--skip-existing", action="store_true", help="skip chunks that already exist")
    ap.add_argument("--no-embed", action="store_true", help="do not compute embeddings")
    ap.add_argument("--update-fts", action="store_true", help="refresh FTS text for existing chunks")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    log = setup_logger("phi4mini.rag.index", args.log_level)

    outbox_path = Path(args.outbox)
    db_path = Path(args.db)
    state_db_path = Path(args.state_db) if args.state_db else db_path

    conn = init_state_db(db_path)
    state_conn = init_state_db(state_db_path)

    raw_offset = state_get(state_conn, args.cursor_key) or "0"
    try:
        offset = int(raw_offset)
    except Exception:
        offset = 0

    deadline = time.monotonic() + max(1, int(args.max_seconds))
    log.info(
        "start outbox=%s db=%s offset=%d budget_s=%d embed_model=%s skip_existing=%s no_embed=%s update_fts=%s",
        str(outbox_path),
        str(db_path),
        offset,
        args.max_seconds,
        args.embed_model,
        args.skip_existing,
        args.no_embed,
        args.update_fts,
    )

    advanced = False
    while time.monotonic() < deadline:
        items, new_offset = iter_jsonl_new(outbox_path, start_offset=offset, max_items=args.batch_items)
        if not items:
            # No new work; wait a little so we still fill the 15 min window.
            time_left = deadline - time.monotonic()
            if time_left <= 0:
                break
            time.sleep(min(5.0, max(0.2, time_left)))
            continue

        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        n_chunks = 0
        for obj in items:
            doc_id = stable_doc_id(obj)
            title = (obj.get("title") or "").strip()
            url = (obj.get("url") or obj.get("link") or "").strip()
            text = strip_html_basic(obj.get("text") or obj.get("content") or "")
            if not text:
                continue

            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                existing_chunk_id = None
                if args.skip_existing or args.update_fts:
                    existing_chunk_id = get_chunk_id(
                        conn,
                        doc_id=doc_id,
                        user_id="",
                        source_type="outbox",
                        chunk_index=idx,
                    )
                if existing_chunk_id is not None:
                    if args.update_fts:
                        tags = _keyword_tags(f"{title}\n{ch}")
                        fts_text = ch if not tags else f"{ch}\n\nTAGS: {' '.join(tags)}"
                        update_chunk_fts(conn, existing_chunk_id, fts_text)
                    if args.skip_existing or args.update_fts:
                        continue
                tags = _keyword_tags(f"{title}\n{ch}")
                fts_text = ch if not tags else f"{ch}\n\nTAGS: {' '.join(tags)}"
                emb = None
                if not args.no_embed:
                    emb = embed(ch[:2000], model=args.embed_model)  # cap embedding input
                insert_chunk(
                    conn,
                    doc_id=doc_id,
                    source_type="outbox",
                    user_id="",
                    source_url=url,
                    title=title,
                    chunk_index=idx,
                    text=ch,
                    fts_text=fts_text,
                    embedding=emb,
                    embedding_model=args.embed_model if emb is not None else None,
                    created_at=now_iso,
                )
                n_chunks += 1

        conn.commit()
        offset = new_offset
        state_set(state_conn, args.cursor_key, str(offset))
        advanced = True
        log.info("indexed items=%d chunks=%d new_offset=%d", len(items), n_chunks, offset)

    if advanced:
        log.info("done offset=%d", offset)
    else:
        log.info("done (no new items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
