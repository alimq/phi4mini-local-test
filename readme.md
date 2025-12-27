# phi4mini-local-test

This is a small local project that turns RSS feeds into a digest and lets you query the same data through a simple RAG UI. The UI shows the summary and a Q&A page with sources.

## The important result: BM25 beats RAG on AI news

This repo includes a real benchmark on domain-specific data (AI news). The result is clear: BM25 outperforms GraphRAG in this setup. This is a big deal if you are building RAG for narrow, proper-noun-heavy news data. A classic keyword system wins.

What we measured (real labeled run):

- recall@20: GraphRAG 0.5033 vs BM25 0.6500
- MRR: GraphRAG 0.4639 vs BM25 0.5372

Why this matters:

- AI news is dense with names and titles. BM25 handles this well.
- GraphRAG adds complexity, but did not help in this domain.
- If you care about accurate retrieval, start with BM25 and prove RAG helps before shipping it.

Where the benchmark lives:

- `eval/graphrag_eval/report.md`

## Brief project context

- Summarizer: renders the latest digest as a small website.
- Global RAG: questions over the full news corpus, with sources.
- User-docs RAG: upload files and ask questions over them.

That is it. The rest of this repo exists to support the benchmark and the digest pipeline.

## Example questions from the benchmark set

These are real queries from `queries_labeled.json` in `eval/graphrag_eval`:

- "What is Powerful ASR + diarization + speculative decoding with Hugging Face Inference Endpoints about?"
- "What is the significance of Train 400x faster Static Embedding Models with Sentence Transformers in AI?"
- "What did researchers say about Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face?"
