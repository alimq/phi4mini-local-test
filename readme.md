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

## Example questions to test retrieval

Try queries that are heavy on proper nouns and titles. This is where BM25 shines.

- "What did Anthropic ship last week related to Claude?"
- "Which OpenAI paper mentioned distillation in December?"
- "What are the key claims in the new Gemini model update?"
