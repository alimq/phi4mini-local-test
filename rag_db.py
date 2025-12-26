#!/usr/bin/env python3
"""SQLite-backed store for (2) global RAG and (3) user-document RAG.

No external vector database is required.

Retrieval strategy:
- Candidate selection: SQLite FTS5 over chunk text.
- Optional rerank: cosine similarity over embeddings stored as float32 blobs.

The DB also stores a small `state` table for incremental cursors.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(lvl)
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    logging.Formatter.converter = time.gmtime
    return logger


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_conn(db_path: str) -> sqlite3.Connection:
    """Convenience for services that pass paths via env vars."""
    return connect(Path(db_path))


def ensure_db(conn: sqlite3.Connection) -> None:
    """Create schema if needed."""
    init_db(conn)


def init_state_db(db_path: Path) -> sqlite3.Connection:
    """Create/open DB (including `state` table) at the provided path."""
    conn = connect(db_path)
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
          doc_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL,
          source_url TEXT,
          title TEXT,
          meta_json TEXT,
          created_at_utc TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id TEXT NOT NULL,
          user_id TEXT NOT NULL DEFAULT '',
          source_type TEXT NOT NULL,
          source_url TEXT,
          title TEXT,
          chunk_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          embedding BLOB,
          embedding_model TEXT,
          created_at_utc TEXT NOT NULL,
          FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
        )
        """
    )

    # Full-text index over chunk text. Keep it simple; we insert manually.
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text,
          content='chunks',
          content_rowid='chunk_id'
        )
        """
    )

    # Simple entity graph tables for GraphRAG-style retrieval.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
          entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_entities (
          chunk_id INTEGER NOT NULL,
          entity_id INTEGER NOT NULL,
          count INTEGER NOT NULL DEFAULT 1,
          PRIMARY KEY (chunk_id, entity_id),
          FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
          FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_edges (
          src_entity_id INTEGER NOT NULL,
          dst_entity_id INTEGER NOT NULL,
          weight INTEGER NOT NULL DEFAULT 1,
          PRIMARY KEY (src_entity_id, dst_entity_id),
          FOREIGN KEY (src_entity_id) REFERENCES entities(entity_id),
          FOREIGN KEY (dst_entity_id) REFERENCES entities(entity_id)
        )
        """
    )

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(src_entity_id)")
    conn.commit()


def state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def upsert_doc(
    conn: sqlite3.Connection,
    doc_id: str,
    source_type: str,
    source_url: str,
    title: str,
    meta: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO docs(doc_id,source_type,source_url,title,meta_json,created_at_utc)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE SET
          source_type=excluded.source_type,
          source_url=excluded.source_url,
          title=excluded.title,
          meta_json=excluded.meta_json
        """,
        (doc_id, source_type, source_url, title, json.dumps(meta, ensure_ascii=False), utc_now_iso()),
    )
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    doc_id: str,
    user_id: Optional[str],
    source_type: str,
    source_url: str,
    title: str,
    chunk_index: int,
    text: str,
    fts_text: Optional[str] = None,
    embedding: Optional[Sequence[float]] = None,
    embedding_model: Optional[str] = None,
    created_at: Optional[str] = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chunks(doc_id,user_id,source_type,source_url,title,chunk_index,text,embedding,embedding_model,created_at_utc)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            doc_id,
            user_id or "",
            source_type,
            source_url,
            title,
            chunk_index,
            text,
            encode_embedding(embedding),
            (embedding_model or None),
            (created_at or utc_now_iso()),
        ),
    )
    chunk_id = int(cur.lastrowid)
    fts_payload = fts_text if fts_text is not None else text
    cur.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (chunk_id, fts_payload))
    conn.commit()
    return chunk_id


def get_chunk_id(
    conn: sqlite3.Connection,
    doc_id: str,
    user_id: Optional[str],
    source_type: str,
    chunk_index: int,
) -> Optional[int]:
    row = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE doc_id=? AND user_id=? AND source_type=? AND chunk_index=?
        LIMIT 1
        """,
        (doc_id, user_id or "", source_type, int(chunk_index)),
    ).fetchone()
    if not row:
        return None
    return int(row[0])


def update_chunk_fts(conn: sqlite3.Connection, chunk_id: int, fts_text: str) -> None:
    conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (int(chunk_id),))
    conn.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (int(chunk_id), fts_text))


def has_chunks_for_doc(conn: sqlite3.Connection, doc_id: str, user_id: str = "") -> bool:
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE doc_id=? AND user_id=? LIMIT 1", (doc_id, user_id or "")
    ).fetchone()
    return row is not None


@dataclass
class ChunkHit:
    chunk_id: int
    doc_id: str
    user_id: str
    source_type: str
    source_url: str
    title: str
    chunk_index: int
    text: str
    score: float


def fts_candidates(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 30,
    source_type: Optional[str] = None,
    user_id: Optional[str] = None,
    match_any: bool = False,
) -> List[ChunkHit]:
    # Use bm25 for a reasonable lexical score.
    where: List[str] = []
    params: List[object] = []
    if source_type:
        where.append("c.source_type=?")
        params.append(source_type)
    if user_id is not None:
        where.append("c.user_id=?")
        params.append(user_id)

    # FTS MATCH syntax: quote each token to avoid parse errors on punctuation.
    tokens = [t for t in query.split() if t.strip()]
    if not tokens:
        match = '""'
    elif match_any:
        match = " OR ".join([f'"{t}"' for t in tokens])
    else:
        match = " ".join([f'"{t}"' for t in tokens])
    where.append("chunks_fts MATCH ?")
    params.append(match)

    where_sql = "WHERE " + " AND ".join(where)

    sql = f"""
      SELECT c.chunk_id, c.doc_id, c.user_id, c.source_type, c.source_url, c.title, c.chunk_index, c.text,
             bm25(chunks_fts) AS bm25_score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      {where_sql}
      ORDER BY bm25_score
      LIMIT ?
    """

    params2 = params + [int(limit)]
    rows = conn.execute(sql, params2).fetchall()

    # bm25 smaller is better, convert to higher-is-better.
    out: List[ChunkHit] = []
    for r in rows:
        bm25_score = float(r[8]) if r[8] is not None else 0.0
        score = 1.0 / (1.0 + max(0.0, bm25_score))
        out.append(
            ChunkHit(
                chunk_id=int(r[0]),
                doc_id=str(r[1]),
                user_id=str(r[2]),
                source_type=str(r[3]),
                source_url=str(r[4] or ""),
                title=str(r[5] or ""),
                chunk_index=int(r[6]),
                text=str(r[7] or ""),
                score=score,
            )
        )
    return out


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    source_type: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 30,
    match_any: bool = False,
) -> List[ChunkHit]:
    """Backwards-compatible name used by the web servers."""
    return fts_candidates(
        conn=conn,
        query=query,
        limit=limit,
        source_type=source_type,
        user_id=user_id,
        match_any=match_any,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if np is not None:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    # pure-python fallback
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = (na ** 0.5) * (nb ** 0.5)
    return 0.0 if denom <= 0 else (dot / denom)


def decode_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    # Stored as JSON bytes for compatibility. If you want a tighter format,
    # switch to float32 and store dims.
    try:
        return json.loads(blob.decode("utf-8"))
    except Exception:
        return None


def encode_embedding(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    return json.dumps(list(vec), ensure_ascii=False).encode("utf-8")


def rerank_with_embeddings(
    query_embedding: Sequence[float],
    hits: List[ChunkHit],
    conn: sqlite3.Connection,
    top_k: int = 8,
) -> List[ChunkHit]:
    if not hits:
        return []

    # Load embeddings for all candidates in one query
    ids = [h.chunk_id for h in hits]
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({qmarks})", ids).fetchall()
    emb_map = {int(r[0]): decode_embedding(r[1]) for r in rows}

    reranked: List[Tuple[float, ChunkHit]] = []
    for h in hits:
        emb = emb_map.get(h.chunk_id)
        if emb is None:
            continue
        sim = _cosine(query_embedding, emb)
        reranked.append((sim, h))

    reranked.sort(key=lambda x: x[0], reverse=True)
    out = [h for _, h in reranked[:top_k]]
    if out:
        return out
    # If none had embeddings, fall back to lexical order.
    return hits[:top_k]


def get_or_create_entity(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO entities(name) VALUES(?)", (name,))
    row = cur.execute("SELECT entity_id FROM entities WHERE name=?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"failed to create entity: {name}")
    return int(row[0])


def get_entity_ids(conn: sqlite3.Connection, names: Sequence[str]) -> List[int]:
    if not names:
        return []
    qmarks = ",".join(["?"] * len(names))
    rows = conn.execute(f"SELECT entity_id FROM entities WHERE name IN ({qmarks})", list(names)).fetchall()
    return [int(r[0]) for r in rows]


def upsert_chunk_entity(conn: sqlite3.Connection, chunk_id: int, entity_id: int, count: int) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chunk_entities(chunk_id, entity_id, count)
        VALUES(?,?,?)
        ON CONFLICT(chunk_id, entity_id) DO UPDATE SET count=count+excluded.count
        """,
        (chunk_id, entity_id, count),
    )


def upsert_entity_edge(conn: sqlite3.Connection, src_id: int, dst_id: int, weight: int = 1) -> None:
    if src_id == dst_id:
        return
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO entity_edges(src_entity_id, dst_entity_id, weight)
        VALUES(?,?,?)
        ON CONFLICT(src_entity_id, dst_entity_id) DO UPDATE SET weight=weight+excluded.weight
        """,
        (src_id, dst_id, weight),
    )


def graph_expand_entities(conn: sqlite3.Connection, entity_ids: Sequence[int], limit: int = 20) -> List[Tuple[int, int]]:
    if not entity_ids:
        return []
    qmarks = ",".join(["?"] * len(entity_ids))
    rows = conn.execute(
        f"""
        SELECT dst_entity_id, SUM(weight) AS w
        FROM entity_edges
        WHERE src_entity_id IN ({qmarks})
        GROUP BY dst_entity_id
        ORDER BY w DESC
        LIMIT ?
        """,
        list(entity_ids) + [int(limit)],
    ).fetchall()
    return [(int(r[0]), int(r[1])) for r in rows]


def graph_chunks_for_entities(
    conn: sqlite3.Connection,
    entity_ids: Sequence[int],
    limit: int = 20,
    source_type: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[ChunkHit]:
    if not entity_ids:
        return []
    where: List[str] = []
    params: List[object] = []
    if source_type:
        where.append("c.source_type=?")
        params.append(source_type)
    if user_id is not None:
        where.append("c.user_id=?")
        params.append(user_id)
    qmarks = ",".join(["?"] * len(entity_ids))
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
      SELECT c.chunk_id, c.doc_id, c.user_id, c.source_type, c.source_url, c.title, c.chunk_index, c.text,
             SUM(ce.count) AS score_sum
      FROM chunk_entities ce
      JOIN chunks c ON c.chunk_id = ce.chunk_id
      WHERE ce.entity_id IN ({qmarks})
      {(" AND " + " AND ".join(where)) if where else ""}
      GROUP BY c.chunk_id
      ORDER BY score_sum DESC
      LIMIT ?
    """
    rows = conn.execute(sql, list(entity_ids) + params + [int(limit)]).fetchall()
    out: List[ChunkHit] = []
    for r in rows:
        score = float(r[8]) if r[8] is not None else 0.0
        out.append(
            ChunkHit(
                chunk_id=int(r[0]),
                doc_id=str(r[1]),
                user_id=str(r[2]),
                source_type=str(r[3]),
                source_url=str(r[4] or ""),
                title=str(r[5] or ""),
                chunk_index=int(r[6]),
                text=str(r[7] or ""),
                score=score,
            )
        )
    return out
