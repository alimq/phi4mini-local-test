#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Optional


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _first_embedding_dim(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
    if not row or row[0] is None:
        return None
    try:
        vec = json.loads(row[0].decode("utf-8"))
        return len(vec)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/var/lib/phi4mini/rag.sqlite", help="Path to rag.sqlite")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"missing_db path={db_path}")
        return 2

    conn = sqlite3.connect(str(db_path))
    total_docs = _scalar(conn, "SELECT COUNT(*) FROM docs")
    total_chunks = _scalar(conn, "SELECT COUNT(*) FROM chunks")
    total_fts = _scalar(conn, "SELECT COUNT(*) FROM chunks_fts")
    total_embeddings = _scalar(conn, "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    distinct_docs = _scalar(conn, "SELECT COUNT(DISTINCT doc_id) FROM chunks")
    total_entities = _scalar(conn, "SELECT COUNT(*) FROM entities")
    total_chunk_entities = _scalar(conn, "SELECT COUNT(*) FROM chunk_entities")
    total_edges = _scalar(conn, "SELECT COUNT(*) FROM entity_edges")

    models = conn.execute(
        "SELECT embedding_model, COUNT(*) FROM chunks WHERE embedding_model IS NOT NULL GROUP BY embedding_model"
    ).fetchall()
    dim = _first_embedding_dim(conn)

    print(f"db_path={db_path}")
    print(f"total_docs={total_docs}")
    print(f"distinct_docs_with_chunks={distinct_docs}")
    print(f"total_chunks={total_chunks}")
    print(f"total_chunks_fts={total_fts}")
    print(f"total_embeddings={total_embeddings}")
    print(f"graph_entities={total_entities}")
    print(f"graph_chunk_entities={total_chunk_entities}")
    print(f"graph_edges={total_edges}")
    print(f"embedding_dim={dim if dim is not None else 'unknown'}")
    if models:
        print("embedding_models=")
        for name, cnt in models:
            print(f"  - {name}: {cnt}")
    else:
        print("embedding_models=none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
