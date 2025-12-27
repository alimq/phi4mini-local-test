#!/usr/bin/env python3
"""
GraphRAG vs BM25 evaluation harness.

Usage:
  python eval_graphrag_vs_bm25.py \
    --corpus corpus.jsonl \
    --queries queries.json \
    --topk 20 \
    --graphrag_http_url http://127.0.0.1:PORT/search

  OR

  python eval_graphrag_vs_bm25.py \
    --corpus corpus.jsonl \
    --queries queries.json \
    --topk 20 \
    --graphrag_cli_template './my_graphrag_search --query "{query}" --topk {topk} --retrieval_only --json'

Notes:
- The evaluation does NOT call any LLMs.
- GraphRAG is treated as a black-box retriever returning ranked ids.
- For best accuracy, configure GraphRAG to return chunk ids directly.
- If GraphRAG returns doc ids, this script can map doc->chunks (default) or
  compute doc-level metrics with --doc_level.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", [])
    if not isinstance(data, list):
        raise ValueError("queries.json must be a JSON list or {queries:[...]}")
    return data


def first_gold_rank(ranked_ids: Sequence[str], gold_set: set) -> Optional[int]:
    for idx, rid in enumerate(ranked_ids, 1):
        if rid in gold_set:
            return idx
    return None


def recall_at_k(ranked_ids: Sequence[str], gold_set: set, k: int) -> float:
    if not gold_set:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in gold_set)
    return hits / float(len(gold_set))


def precision_at_k(ranked_ids: Sequence[str], gold_set: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in gold_set)
    return hits / float(k)


def mrr(ranked_ids: Sequence[str], gold_set: set) -> float:
    rank = first_gold_rank(ranked_ids, gold_set)
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def ndcg_at_k(ranked_ids: Sequence[str], gold_set: set, k: int) -> float:
    if not gold_set:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], 1):
        if rid in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG with binary relevance
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, seed: int = 13) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = []
    n = len(arr)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass
class CorpusIndex:
    chunk_ids: List[str]
    chunk_texts: List[str]
    doc_id_for_chunk: List[Optional[str]]
    bm25: BM25Okapi
    id_to_idx: Dict[str, int]
    doc_to_chunks: Dict[str, List[str]]
    doc_to_chunk_idxs: Dict[str, List[int]]


def build_corpus_index(corpus_path: str) -> CorpusIndex:
    corpus = read_jsonl(corpus_path)
    chunk_ids = []
    chunk_texts = []
    doc_ids = []
    id_to_idx = {}
    doc_to_chunks = defaultdict(list)
    doc_to_chunk_idxs = defaultdict(list)
    for i, item in enumerate(corpus):
        cid = str(item.get("id"))
        text = str(item.get("text", ""))
        doc_id = item.get("doc_id")
        chunk_ids.append(cid)
        chunk_texts.append(text)
        doc_ids.append(doc_id if doc_id is not None else None)
        id_to_idx[cid] = i
        if doc_id is not None:
            doc_id = str(doc_id)
            doc_to_chunks[doc_id].append(cid)
            doc_to_chunk_idxs[doc_id].append(i)
    tokenized = [tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    return CorpusIndex(
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        doc_id_for_chunk=doc_ids,
        bm25=bm25,
        id_to_idx=id_to_idx,
        doc_to_chunks=dict(doc_to_chunks),
        doc_to_chunk_idxs=dict(doc_to_chunk_idxs),
    )


def bm25_retrieve(index: CorpusIndex, query: str, topk: int) -> List[str]:
    tokens = tokenize(query)
    scores = index.bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:topk]
    return [index.chunk_ids[i] for i in top_idx]


def run_graphrag_http(url: str, query: str, topk: int, timeout: int) -> Dict[str, Any]:
    if "debug/retrieval" in url:
        resp = requests.get(url, params={"q": query}, timeout=timeout)
    else:
        payload = {"query": query, "topk": topk, "mode": "retrieval_only"}
        resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def normalize_raw_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and "top_chunks" in raw and "results" not in raw:
        raw["results"] = [
            {"id": str(item["chunk_id"]), "score": item.get("score", 0.0)}
            for item in raw.get("top_chunks", [])
            if isinstance(item, dict) and "chunk_id" in item
        ]
    return raw


def run_graphrag_cli(template: str, query: str, topk: int, timeout: int) -> Dict[str, Any]:
    cmd = template.format(query=query.replace('"', '\\"'), topk=topk)
    completed = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GraphRAG CLI failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def extract_ids_from_text(text: str, id_set: set) -> List[str]:
    if not text:
        return []
    candidates = re.findall(r"[\w:/\.\-]+", text)
    hits = []
    for tok in candidates:
        if tok in id_set:
            hits.append(tok)
    return hits


def map_results_to_chunk_ids(
    raw_results: List[Dict[str, Any]],
    index: CorpusIndex,
    topk: int,
    doc_level: bool,
    allow_doc_expand: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    id_set = set(index.chunk_ids)
    mapped = []
    notes = {"used_doc_ids": False, "extracted_from_text": False, "doc_level": doc_level}

    def add_id(rid: str):
        if rid not in mapped:
            mapped.append(rid)

    for item in raw_results:
        rid = item.get("id") or item.get("chunk_id") or item.get("doc_id")
        if rid is None:
            text = item.get("text") or item.get("citation") or item.get("content")
            ids = extract_ids_from_text(str(text), id_set)
            if ids:
                notes["extracted_from_text"] = True
                for cid in ids:
                    add_id(cid)
            continue
        rid = str(rid)
        if rid in id_set:
            add_id(rid)
        else:
            if doc_level:
                add_id(rid)
            elif allow_doc_expand and rid in index.doc_to_chunks:
                notes["used_doc_ids"] = True
                for cid in index.doc_to_chunks[rid]:
                    add_id(cid)
            else:
                text = item.get("text") or item.get("citation") or item.get("content")
                ids = extract_ids_from_text(str(text), id_set)
                if ids:
                    notes["extracted_from_text"] = True
                    for cid in ids:
                        add_id(cid)

    return mapped[:topk], notes


def compute_metrics(ranked_ids: List[str], gold_ids: List[str], k10: int, k20: int) -> Dict[str, float]:
    gold_set = set(gold_ids)
    return {
        "recall@10": recall_at_k(ranked_ids, gold_set, k10),
        "recall@20": recall_at_k(ranked_ids, gold_set, k20),
        "precision@10": precision_at_k(ranked_ids, gold_set, k10),
        "mrr": mrr(ranked_ids, gold_set),
        "ndcg@10": ndcg_at_k(ranked_ids, gold_set, k10),
    }


def summarize_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([d[k] for d in per_query])) for k in keys}


def compute_bootstrap(per_query: List[Dict[str, float]], n_resamples: int, seed: int) -> Dict[str, Dict[str, float]]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    out = {}
    for k in keys:
        values = [d[k] for d in per_query]
        lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        out[k] = {"low": lo, "high": hi}
    return out


def format_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def safe_rel_change(system: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if system > 0 else 0.0
    return (system - baseline) / baseline


def write_reports(
    out_dir: str,
    baseline_summary: Dict[str, float],
    system_summary: Dict[str, float],
    baseline_ci: Dict[str, Dict[str, float]],
    system_ci: Dict[str, Dict[str, float]],
    per_query: List[Dict[str, Any]],
    notes: Dict[str, Any],
    topk: int,
):
    os.makedirs(out_dir, exist_ok=True)
    report_json = {
        "summary": {
            "baseline": baseline_summary,
            "system": system_summary,
        },
        "confidence_intervals": {
            "baseline": baseline_ci,
            "system": system_ci,
        },
        "per_query": per_query,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # Markdown report
    md_lines = []
    md_lines.append("# GraphRAG vs BM25 Retrieval Evaluation")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("| Metric | BM25 | GraphRAG | Delta (G-B) | Relative | 95% CI BM25 | 95% CI GraphRAG |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric in ["recall@10", "recall@20", "mrr", "ndcg@10", "precision@10"]:
        b = baseline_summary.get(metric, 0.0)
        s = system_summary.get(metric, 0.0)
        d = s - b
        rel = safe_rel_change(s, b)
        bci = baseline_ci.get(metric, {"low": 0.0, "high": 0.0})
        sci = system_ci.get(metric, {"low": 0.0, "high": 0.0})
        md_lines.append(
            f"| {metric} | {b:.4f} | {s:.4f} | {d:+.4f} | {rel*100:.2f}% | "
            f"[{bci['low']:.4f}, {bci['high']:.4f}] | [{sci['low']:.4f}, {sci['high']:.4f}] |"
        )
    md_lines.append("")

    # Plain-English conclusion
    r20_delta = system_summary.get("recall@20", 0.0) - baseline_summary.get("recall@20", 0.0)
    mrr_delta = system_summary.get("mrr", 0.0) - baseline_summary.get("mrr", 0.0)
    r20_rel = safe_rel_change(system_summary.get("recall@20", 0.0), baseline_summary.get("recall@20", 0.0))
    mrr_rel = safe_rel_change(system_summary.get("mrr", 0.0), baseline_summary.get("mrr", 0.0))
    r20_ci = system_ci.get("recall@20", {"low": 0.0, "high": 0.0})
    mrr_ci = system_ci.get("mrr", {"low": 0.0, "high": 0.0})
    md_lines.append("## Conclusion")
    md_lines.append("")
    md_lines.append(
        f"On this test set, GraphRAG is {r20_rel*100:.2f}% "
        f"{'better' if r20_delta >= 0 else 'worse'} than BM25 on Recall@20 "
        f"(GraphRAG CI [{r20_ci['low']:.4f}, {r20_ci['high']:.4f}]), and "
        f"{mrr_rel*100:.2f}% "
        f"{'better' if mrr_delta >= 0 else 'worse'} on MRR "
        f"(GraphRAG CI [{mrr_ci['low']:.4f}, {mrr_ci['high']:.4f}])."
    )
    md_lines.append("")

    # Per-query breakdown
    md_lines.append("## Per-query Breakdown (top 10 deltas)")
    md_lines.append("")
    worst = sorted(per_query, key=lambda x: x["delta"]["recall@20"])[:10]
    best = sorted(per_query, key=lambda x: x["delta"]["recall@20"], reverse=True)[:10]

    def render_block(title: str, items: List[Dict[str, Any]]):
        md_lines.append(f"### {title}")
        md_lines.append("")
        for item in items:
            md_lines.append(f"- Query: {item['query']}")
            md_lines.append(f"  - Gold IDs: {', '.join(item['gold_ids'])}")
            md_lines.append(
                f"  - BM25 top{min(10, topk)}: {', '.join(item['bm25']['top_ids'][:10])} "
                f"(first gold rank: {item['bm25']['first_gold_rank']})"
            )
            md_lines.append(
                f"  - GraphRAG top{min(10, topk)}: {', '.join(item['system']['top_ids'][:10])} "
                f"(first gold rank: {item['system']['first_gold_rank']})"
            )
            md_lines.append("")

    render_block("Worst for GraphRAG (BM25 wins)", worst)
    render_block("Best for GraphRAG (GraphRAG wins)", best)

    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append(f"- TopK used: {topk}")
    md_lines.append(f"- GraphRAG mapping notes: {json.dumps(notes)}")
    md_lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vs BM25 retrieval.")
    parser.add_argument("--corpus", required=True, help="JSONL corpus with id/text.")
    parser.add_argument("--queries", required=True, help="JSON list of query objects.")
    parser.add_argument("--topk", type=int, default=20, help="TopK to retrieve.")
    parser.add_argument("--graphrag_http_url", default=None, help="GraphRAG HTTP endpoint.")
    parser.add_argument("--graphrag_cli_template", default=None, help="GraphRAG CLI template.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for GraphRAG calls.")
    parser.add_argument("--doc_level", action="store_true", help="Evaluate doc-level if GraphRAG returns doc ids.")
    parser.add_argument("--no_doc_expand", action="store_true", help="Do not expand doc ids to chunks.")
    parser.add_argument("--out_dir", default=".", help="Output directory for reports.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for bootstrap.")
    args = parser.parse_args()

    if not args.graphrag_http_url and not args.graphrag_cli_template:
        print("Error: Provide --graphrag_http_url or --graphrag_cli_template", file=sys.stderr)
        return 2

    index = build_corpus_index(args.corpus)
    queries = read_queries(args.queries)
    if not queries:
        print("No queries found.", file=sys.stderr)
        return 1

    per_query_results = []
    mapping_notes = {"used_doc_ids": False, "extracted_from_text": False, "doc_level": args.doc_level}

    for q in tqdm(queries, desc="Evaluating queries"):
        query = q.get("query")
        gold_ids = [str(g) for g in q.get("gold_ids", [])]
        if query is None:
            continue
        bm25_ids = bm25_retrieve(index, query, args.topk)
        if args.graphrag_http_url:
            raw = run_graphrag_http(args.graphrag_http_url, query, args.topk, args.timeout)
        else:
            raw = run_graphrag_cli(args.graphrag_cli_template, query, args.topk, args.timeout)
        raw = normalize_raw_results(raw)
        raw_results = raw.get("results") or raw.get("hits") or raw.get("documents") or []
        system_ids, notes = map_results_to_chunk_ids(
            raw_results,
            index,
            args.topk,
            doc_level=args.doc_level,
            allow_doc_expand=not args.no_doc_expand,
        )
        mapping_notes["used_doc_ids"] = mapping_notes["used_doc_ids"] or notes.get("used_doc_ids", False)
        mapping_notes["extracted_from_text"] = mapping_notes["extracted_from_text"] or notes.get("extracted_from_text", False)

        bm25_metrics = compute_metrics(bm25_ids, gold_ids, 10, 20)
        sys_metrics = compute_metrics(system_ids, gold_ids, 10, 20)
        delta = {k: sys_metrics[k] - bm25_metrics[k] for k in bm25_metrics.keys()}

        per_query_results.append(
            {
                "query": query,
                "gold_ids": gold_ids,
                "bm25": {
                    "top_ids": bm25_ids,
                    "first_gold_rank": first_gold_rank(bm25_ids, set(gold_ids)),
                    "metrics": bm25_metrics,
                },
                "system": {
                    "top_ids": system_ids,
                    "first_gold_rank": first_gold_rank(system_ids, set(gold_ids)),
                    "metrics": sys_metrics,
                },
                "delta": delta,
            }
        )

    baseline_per_query = [r["bm25"]["metrics"] for r in per_query_results]
    system_per_query = [r["system"]["metrics"] for r in per_query_results]
    baseline_summary = summarize_metrics(baseline_per_query)
    system_summary = summarize_metrics(system_per_query)
    baseline_ci = compute_bootstrap(baseline_per_query, args.bootstrap, args.seed)
    system_ci = compute_bootstrap(system_per_query, args.bootstrap, args.seed)

    write_reports(
        args.out_dir,
        baseline_summary,
        system_summary,
        baseline_ci,
        system_ci,
        per_query_results,
        mapping_notes,
        args.topk,
    )

    # Console summary
    print("GraphRAG vs BM25 evaluation complete.")
    print(f"BM25 Recall@20: {baseline_summary.get('recall@20', 0.0):.4f}")
    print(f"GraphRAG Recall@20: {system_summary.get('recall@20', 0.0):.4f}")
    print(f"BM25 MRR: {baseline_summary.get('mrr', 0.0):.4f}")
    print(f"GraphRAG MRR: {system_summary.get('mrr', 0.0):.4f}")
    print(f"Reports saved to {os.path.join(args.out_dir, 'report.md')} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
