#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

import markdown2

DIGEST_PATH = Path(os.environ.get("PHI4MINI_DIGEST", "/var/lib/phi4mini/digest.md"))

app = FastAPI(title="Phi4Mini Summarizer", docs_url=None, redoc_url=None)


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:980px;margin:30px auto;padding:0 16px;}}
    .muted{{color:#555;}}
    .card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0;}}
    pre,code{{background:#f6f6f6;border-radius:8px;}}
    pre{{padding:12px;overflow:auto;}}
    a{{color:#0b5; text-decoration:none;}}
    a:hover{{text-decoration:underline;}}
  </style>
</head>
<body>
{body}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    if not DIGEST_PATH.exists():
        body = f"<h1>Summarizer</h1><p class='muted'>No digest found at <code>{DIGEST_PATH}</code> yet.</p>"
        return _html_page("Summarizer", body)

    md = DIGEST_PATH.read_text(encoding="utf-8", errors="replace")
    html = markdown2.markdown(md, extras=["fenced-code-blocks", "tables"])
    body = f"""
    <h1>Summarizer</h1>
    <p class='muted'>Live view of <code>{DIGEST_PATH}</code></p>
    <p><a href='/raw'>Download raw markdown</a></p>
    <div class='card'>{html}</div>
    """
    return _html_page("Summarizer", body)


@app.get("/raw", response_class=PlainTextResponse)
def raw() -> str:
    if not DIGEST_PATH.exists():
        return ""
    return DIGEST_PATH.read_text(encoding="utf-8", errors="replace")
