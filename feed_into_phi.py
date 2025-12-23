#!/usr/bin/env python3
"""
feed_into_phi.py — All-of-the-above AI report (RAG + local LLM focus)

Outputs (in digest.md):
1) Longer gist (8–10 sentences) focused on RAG + running SLM/LLM locally
2) Trend map w/ evidence links
3) Weekly watchlist (what to track next week)
4) Contradictions & open questions
5) Action plan for you (experiments + pipeline improvements)
6) Uncertainty notes (what's unclear due to limited snippets)

Performance:
- Bounded by max-items, chunk size, and num_predict caps
- Final report streams to digest.md (so it’s never empty during final generation)
- run.log will show "starting final report…" and a clear done line

Deps:
  python3 -m pip install -U requests python-dateutil
"""

from __future__ import annotations

import argparse, json, re, sys, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dateutil import parser as dtparser


# -----------------------------
# Utilities
# -----------------------------
def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


def parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class Item:
    title: str
    url: str
    source: str
    published: Optional[datetime]
    fetched: Optional[datetime]
    text: str


def sort_key(it: Item) -> datetime:
    return it.published or it.fetched or datetime(1970, 1, 1, tzinfo=timezone.utc)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# -----------------------------
# Load / filter
# -----------------------------
def load_items(outbox_path: Path) -> List[Item]:
    items: List[Item] = []
    for line in outbox_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue

        items.append(
            Item(
                title=(o.get("title") or "").strip(),
                url=(o.get("url") or "").strip(),
                source=(o.get("feed_title") or o.get("feed_url") or "").strip(),
                published=parse_dt(o.get("published") or ""),
                fetched=parse_dt(o.get("fetched_at_utc") or ""),
                text=((o.get("text") or "").strip()),
            )
        )
    return items


def select_recent(items: List[Item], last_n: int, hours: Optional[int], days: Optional[int]) -> List[Item]:
    items_sorted = sorted(items, key=sort_key, reverse=True)
    if hours is not None or days is not None:
        now = datetime.now(timezone.utc)
        delta = timedelta(hours=hours or 0) + timedelta(days=days or 0)
        cutoff = now - delta
        filtered = [it for it in items_sorted if sort_key(it) >= cutoff]
        return filtered[:last_n] if last_n else filtered
    return items_sorted[:last_n]


def dedupe(items: List[Item]) -> List[Item]:
    seen = set()
    out: List[Item] = []
    for it in items:
        key = it.url.strip() if it.url else it.title.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# -----------------------------
# Relevance scoring: bias toward RAG + local inference/tooling
# -----------------------------
POS_STRONG = [
    # core LLM/AI
    "llm", "language model", "transformer", "diffusion", "multimodal",
    "alignment", "rlhf", "dpo", "sft", "lora", "qlora", "finetune", "fine-tune",
    "benchmark", "evaluation", "reasoning",
    # RAG / retrieval
    "rag", "retrieval augmented", "retrieval", "rerank", "reranker", "embedding", "embeddings",
    "vector database", "chunking", "semantic search",
    # local inference / infra
    "quantization", "int4", "int8", "fp8", "gguf", "ggml", "llama.cpp", "vllm", "tgi",
    "inference", "throughput", "latency", "cuda", "gpu", "nvidia", "metal",
    "ollama", "openwebui", "lm studio",
    # agents/tool use
    "agent", "tool use", "function calling", "workflow",
]
POS_WEAK = [
    "ai", "artificial intelligence", "machine learning", "deep learning", "neural",
    "nlp", "tokenizer", "tokenization", "arxiv", "pytorch", "jax", "tensorflow",
    "openai", "anthropic", "deepmind", "hugging face",
]

NEGATIVE = [
    # common noise
    "ebook", "e-book", "book", "course", "curriculum", "syllabus", "lecture",
    "podcast", "webinar", "sale", "discount", "deal", "job", "hiring", "internship",
]


def ai_score(it: Item) -> int:
    hay = norm(it.title + " " + it.source + " " + (it.text[:700] if it.text else ""))
    score = 0
    for k in POS_STRONG:
        if k in hay:
            score += 3
    for k in POS_WEAK:
        if k in hay:
            score += 1

    url = (it.url or "").lower()
    if "arxiv.org" in url:
        score += 2
    if "huggingface.co" in url:
        score += 2
    if "github.com" in url:
        score += 1

    for n in NEGATIVE:
        if n in hay:
            score -= 3

    # Extra boost if explicitly about RAG/local
    if any(x in hay for x in ["rag", "retrieval", "embedding", "ollama", "llama.cpp", "gguf", "quantization"]):
        score += 2

    return score


def is_relevant(it: Item) -> bool:
    url = (it.url or "").lower()
    s = ai_score(it)
    if "arxiv.org" in url or "huggingface.co" in url:
        return s >= 3
    return s >= 6


# -----------------------------
# Formatting and chunking
# -----------------------------
def format_items(items: List[Item], snippet_chars: int) -> str:
    blocks = []
    for it in items:
        if not it.title or not it.url:
            continue
        snip = norm(it.text)[:snippet_chars] if it.text else ""
        blocks.append(
            f"TITLE: {it.title}\n"
            f"URL: {it.url}\n"
            f"SOURCE: {it.source}\n"
            f"SNIPPET: {snip}\n\n---\n"
        )
    return "".join(blocks).strip()


def chunk_by_chars(s: str, max_chars: int) -> List[str]:
    if len(s) <= max_chars:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        start = end
    return chunks


# -----------------------------
# Ollama streaming + progress + stream-to-file
# -----------------------------
SYSTEM_RULES = """You are an analyst focused on RAG and running models locally.

Hard rules:
- No meta commentary, no apologies, no questions.
- Do NOT invent details beyond TITLE/SNIPPET.
- Prefer concrete, decision-useful synthesis (what changed, so what, what to do).
- If a detail is unclear, say 'unclear' briefly rather than guessing.
"""


def ollama_generate_stream(
    model: str,
    prompt: str,
    *,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    connect_timeout_s: int,
    read_timeout_s: int,
    first_token_timeout_s: int,
    keep_alive: str,
    debug: bool,
    label: str,
    stream_to_path: Optional[Path] = None,
    stream_mode: str = "w",
) -> Tuple[str, dict]:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "system": SYSTEM_RULES,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    timeout = (connect_timeout_s, read_timeout_s)

    parts: List[str] = []
    meta: dict = {}

    start = time.time()
    last_log = start
    got_first = False

    out_f = None
    if stream_to_path is not None:
        stream_to_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = stream_to_path.open(stream_mode, encoding="utf-8")

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                now = time.time()

                if debug and (now - last_log) >= 2.0:
                    if not got_first:
                        eprint(f"[waiting {label}] {now - start:5.1f}s (no tokens yet)")
                    else:
                        eprint(f"[progress {label}] {now - start:5.1f}s")
                    last_log = now

                if not got_first and (now - start) > first_token_timeout_s:
                    raise TimeoutError(f"Ollama did not stream first token within {first_token_timeout_s}s for {label}")

                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                chunk = obj.get("response") or ""
                if chunk:
                    got_first = True
                    parts.append(chunk)
                    if out_f is not None:
                        out_f.write(chunk)
                        out_f.flush()

                meta = obj

        text = "".join(parts)
        if debug:
            eprint(f"[done {label}] {time.time() - start:5.1f}s chars={len(text)} done={meta.get('done')}")
        return text, meta

    finally:
        if out_f is not None:
            out_f.close()


def stderr_stats(meta: dict, label: str) -> None:
    eprint(json.dumps({
        "label": label,
        "prompt_tokens": meta.get("prompt_eval_count"),
        "output_tokens": meta.get("eval_count"),
        "total_s": (meta.get("total_duration") or 0) / 1e9 if meta.get("total_duration") else None,
    }, ensure_ascii=False))


# -----------------------------
# Prompts
# -----------------------------
# Stage 1: extract compact evidence bullets per chunk (kept small & structured)
CHUNK_EVIDENCE_TEMPLATE = """From ITEMS, extract decision-useful EVIDENCE bullets focused on RAG + local LLM.

Rules:
- Output ONLY markdown bullets.
- Exactly 6–10 bullets.
- Each bullet must:
  - start with a tag in brackets: [RAG] [LOCAL] [EVAL] [AGENTS] [MODELS] [TOOLING] [OTHER]
  - be <= 20 words
  - include exactly ONE URL in parentheses at the end
- Use only TITLE/SNIPPET. If unclear, say 'unclear' (brief).

ITEMS_BEGIN
{items}
ITEMS_END
"""

# Stage 2: generate the full report (all-of-the-above)
FINAL_REPORT_TEMPLATE = """You are given EVIDENCE bullets about recent AI progress.

User focus: building RAG; running SLM/LLM locally.

Write a report that is maximally useful for decisions and next actions.

Hard rules:
- Do NOT invent details beyond EVIDENCE.
- If evidence is thin, mark uncertainty.
- Keep it concise but substantive.

Output EXACTLY these sections in this order:

## Direction of AI (8–10 sentences)
Write 8–10 sentences. Each sentence should be specific (not generic). Mention RAG/local when relevant.
Include 3–6 supporting URLs total across the section (not every sentence).

## Trend map with evidence
Make 5–7 trend buckets. For each bucket:
- 1 sentence summary
- 2–4 evidence bullets copied/adapted from EVIDENCE (keep URLs)

## Weekly watchlist (next 7 days)
Give 6 items:
- what to track
- why it matters for RAG/local
- one supporting URL if available; else say 'unclear'

## Contradictions & open questions
List 5–8 bullets describing:
- where evidence points in different directions, OR
- what is unclear / missing to decide
Each bullet should include a URL when possible.

## Action plan for you (RAG + local)
Give:
- 6 concrete experiments you can do this week (each 1–2 lines)
- 6 engineering upgrades to your pipeline (scraper/summarizer/RAG ingestion) (each 1–2 lines)
Be practical: chunking strategy, embeddings, rerankers, eval sets, quantization, latency.

## Uncertainty notes
List 4–6 bullets of uncertainty caused by limited snippets or missing context.

EVIDENCE_BEGIN
{evidence}
EVIDENCE_END
"""


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbox", default="outbox.jsonl")
    ap.add_argument("--model", default="phi4-mini")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--hours", type=int, default=None)
    ap.add_argument("--last-n", type=int, default=300)
    ap.add_argument("--max-items", type=int, default=120, help="cap for bounded runtime")
    ap.add_argument("--snippet-chars", type=int, default=260)
    ap.add_argument("--max-prompt-chars", type=int, default=6500)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict-chunk", type=int, default=220)
    ap.add_argument("--num-predict-final", type=int, default=900)
    ap.add_argument("--connect-timeout", type=int, default=10)
    ap.add_argument("--read-timeout", type=int, default=1800)
    ap.add_argument("--first-token-timeout", type=int, default=45)
    ap.add_argument("--keep-alive", default="10m")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outbox_path = Path(args.outbox)
    if not outbox_path.exists():
        eprint(f"Missing outbox file: {outbox_path}")
        return 2

    items = load_items(outbox_path)
    items = select_recent(items, last_n=args.last_n, hours=args.hours, days=args.days)
    items = dedupe(items)

    scored = [(ai_score(it), it) for it in items if it.title and it.url]
    scored.sort(key=lambda x: (x[0], sort_key(x[1])), reverse=True)
    filtered = [it for s, it in scored if is_relevant(it)][: args.max_items]

    if args.debug:
        eprint(f"[debug] loaded_recent={len(items)} filtered={len(filtered)} max_items={args.max_items}")
        if filtered:
            eprint("[debug] sample_kept:")
            for it in filtered[:10]:
                eprint(f"  - ({ai_score(it)}) {it.title}")

    if not filtered:
        eprint("No relevant items after filtering. Increase --days/--last-n or loosen thresholds in code.")
        return 1

    items_blob = format_items(filtered, snippet_chars=args.snippet_chars)
    chunks = chunk_by_chars(items_blob, args.max_prompt_chars)

    if args.debug:
        eprint(f"[debug] items_blob_chars={len(items_blob)} chunks={len(chunks)} max_prompt_chars={args.max_prompt_chars}")

    # Stage 1: evidence bullets per chunk
    evidence_parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        prompt = CHUNK_EVIDENCE_TEMPLATE.format(items=ch)
        resp, meta = ollama_generate_stream(
            args.model, prompt,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict_chunk,
            connect_timeout_s=args.connect_timeout,
            read_timeout_s=args.read_timeout,
            first_token_timeout_s=args.first_token_timeout,
            keep_alive=args.keep_alive,
            debug=args.debug,
            label=f"evidence {i}/{len(chunks)}",
        )
        resp = resp.strip()
        if resp:
            evidence_parts.append(resp)
        if args.debug:
            stderr_stats(meta, label=f"evidence {i}/{len(chunks)}")

    evidence_blob = "\n".join(evidence_parts).strip()
    if not evidence_blob:
        eprint("No evidence produced.")
        return 1

    # Stage 2: final report streamed to digest.md
    eprint("[debug] starting final report…")
    digest_path = Path("digest.md")
    digest_path.write_text("", encoding="utf-8")

    final_prompt = FINAL_REPORT_TEMPLATE.format(evidence=evidence_blob)
    final_text, meta = ollama_generate_stream(
        args.model, final_prompt,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict_final,
        connect_timeout_s=args.connect_timeout,
        read_timeout_s=args.read_timeout,
        first_token_timeout_s=args.first_token_timeout,
        keep_alive=args.keep_alive,
        debug=args.debug,
        label="final",
        stream_to_path=digest_path,
        stream_mode="w",
    )

    if args.debug:
        stderr_stats(meta, label="final")
        eprint("[debug] finished (report written to digest.md)")

    print(final_text.strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
