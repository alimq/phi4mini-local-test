#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_db import (
    ensure_db,
    get_or_create_entity,
    init_conn,
    upsert_chunk_entity,
    upsert_entity_edge,
)

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


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]


def extract_entities(text: str, min_len: int) -> Counter:
    tokens = [
        t
        for t in tokenize(text)
        if t not in STOPWORDS and (len(t) >= min_len or t in SHORT_TOKEN_ALLOWLIST)
    ]
    return Counter(tokens)


def iter_chunks(conn: sqlite3.Connection, limit: int = 0) -> Iterable[Tuple[int, str, str]]:
    sql = "SELECT chunk_id, title, text FROM chunks"
    if limit > 0:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (int(limit),)).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    for r in rows:
        yield int(r[0]), str(r[1] or ""), str(r[2] or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a lightweight entity graph for GraphRAG")
    ap.add_argument("--db", default="/var/lib/phi4mini/rag.sqlite")
    ap.add_argument("--reset", action="store_true", help="clear existing graph tables before rebuild")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-entities", type=int, default=8, help="max entities per chunk")
    ap.add_argument("--limit", type=int, default=0, help="limit number of chunks to process")
    args = ap.parse_args()

    conn = init_conn(args.db)
    ensure_db(conn)

    if args.reset:
        conn.execute("DELETE FROM entity_edges")
        conn.execute("DELETE FROM chunk_entities")
        conn.execute("DELETE FROM entities")
        conn.commit()

    total_chunks = 0
    total_entities = 0
    for chunk_id, title, text in iter_chunks(conn, limit=args.limit):
        total_chunks += 1
        counts = extract_entities(f"{title}\n{text}", min_len=args.min_len)
        if not counts:
            continue
        # Keep top entities by frequency per chunk to avoid dense graphs
        top = counts.most_common(args.max_entities)
        entity_ids: List[int] = []
        for name, cnt in top:
            eid = get_or_create_entity(conn, name)
            upsert_chunk_entity(conn, chunk_id, eid, cnt)
            entity_ids.append(eid)
            total_entities += 1

        # Connect co-occurring entities (undirected via two directed edges)
        for i, src in enumerate(entity_ids):
            for dst in entity_ids[i + 1 :]:
                upsert_entity_edge(conn, src, dst, weight=1)
                upsert_entity_edge(conn, dst, src, weight=1)

        if total_chunks % 200 == 0:
            conn.commit()

    conn.commit()
    print(f"chunks_processed={total_chunks}")
    print(f"chunk_entities_added={total_entities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
