#!/usr/bin/env python3
"""
Minimal helper to build queries.json with gold ids.

Usage:
  python build_queries.py --corpus corpus.jsonl --out queries.json

Workflow:
  - Enter a query
  - See BM25 top-20 ids + short snippets
  - Enter comma-separated gold ids
  - Repeat until blank query
"""
import argparse
import json
import re
from typing import List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build queries.json with gold ids.")
    parser.add_argument("--corpus", required=True, help="JSONL corpus with id/text.")
    parser.add_argument("--out", default="queries.json", help="Output JSON file.")
    parser.add_argument("--topk", type=int, default=20, help="BM25 topK to show.")
    parser.add_argument("--snippet_chars", type=int, default=120, help="Snippet length.")
    args = parser.parse_args()

    corpus = read_jsonl(args.corpus)
    chunk_ids = [str(c.get("id")) for c in corpus]
    chunk_texts = [str(c.get("text", "")) for c in corpus]
    tokenized = [tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)

    queries = []
    print("Enter a query (blank to finish):")
    while True:
        query = input("> ").strip()
        if not query:
            break
        scores = bm25.get_scores(tokenize(query))
        top_idx = scores.argsort()[::-1][: args.topk]
        print("\nBM25 top results:")
        for i, idx in enumerate(top_idx, 1):
            cid = chunk_ids[idx]
            text = chunk_texts[idx].replace("\n", " ")
            snippet = text[: args.snippet_chars]
            print(f"{i:02d}. {cid} | {snippet}")
        gold = input("\nEnter gold ids (comma-separated): ").strip()
        gold_ids = [g.strip() for g in gold.split(",") if g.strip()]
        queries.append({"query": query, "gold_ids": gold_ids})
        print("")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {len(queries)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
