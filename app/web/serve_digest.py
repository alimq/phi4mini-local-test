#!/usr/bin/env python3
"""Minimal HTTP server that renders /var/lib/phi4mini/digest.md as nice HTML.

Routes:
- GET /            -> latest report
- GET /runs        -> index (grouped by month/day)
- GET /run/<id>    -> a specific report (id like YYYYMMDD-HHMMSS)
- GET /raw         -> raw markdown
- GET /healthz     -> ok
- GET /static/style.css -> basic styling

Designed to sit behind nginx on :80/:443.
"""

from __future__ import annotations

import html
import os
import re
import socketserver
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DIGEST_MD = Path(os.environ.get("PHI4MINI_DIGEST_MD", "/var/lib/phi4mini/digest.md"))
TITLE = os.environ.get("PHI4MINI_DIGEST_TITLE", "Phi4-mini Digest")
REFRESH_SECS = int(os.environ.get("PHI4MINI_DIGEST_REFRESH_SECS", "60"))

# Optional dependency; we fall back to <pre> if missing.
try:
    import markdown as _md  # type: ignore

    def render_markdown(md_text: str) -> str:
        # Keep extensions conservative; works with common digest formats.
        return _md.markdown(
            md_text,
            extensions=[
                "fenced_code",
                "tables",
                "toc",
                "sane_lists",
                "smarty",
                # codehilite enables pygments if installed; otherwise it still works.
                "codehilite",
            ],
            output_format="html5",
        )

    MARKDOWN_OK = True
except Exception:

    def render_markdown(md_text: str) -> str:
        return "<pre>" + html.escape(md_text) + "</pre>"

    MARKDOWN_OK = False


def _http_date(ts: float) -> str:
    return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(ts))


RUN_LINE_RE = re.compile(r"^## Run (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\s*$", re.M)


@dataclass
class Run:
    id: str               # YYYYMMDD-HHMMSS
    day: str              # YYYY-MM-DD
    time: str             # HH:MM:SS
    dt: datetime          # naive, treated as UTC-ish
    coverage: str         # already human-ish markdown line, may be empty
    md: str               # markdown segment for this run (starts at '## Run ...')


def _run_id(day: str, hhmmss: str) -> str:
    return f"{day.replace('-', '')}-{hhmmss.replace(':', '')}"


def parse_runs(md_text: str) -> List[Run]:
    matches = list(RUN_LINE_RE.finditer(md_text))
    if not matches:
        return []

    runs: List[Run] = []
    for i, m in enumerate(matches):
        day, hhmmss = m.group(1), m.group(2)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        seg = md_text[start:end].strip() + "\n"

        # Try to extract a coverage line if present (we write it right after the Run header).
        # e.g. "**Coverage:** 2025-12-24T...Z → ... (UTC) · **Items:** 115"
        cov = ""
        seg_lines = seg.splitlines()
        if len(seg_lines) >= 2 and "Coverage" in seg_lines[1]:
            cov = seg_lines[1].strip()

        try:
            dt = datetime.strptime(f"{day} {hhmmss}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.min

        runs.append(
            Run(
                id=_run_id(day, hhmmss),
                day=day,
                time=hhmmss,
                dt=dt,
                coverage=cov,
                md=seg,
            )
        )

    runs.sort(key=lambda r: r.dt)
    return runs


def group_runs(runs: List[Run]) -> Dict[str, Dict[str, List[Run]]]:
    """Return {YYYY-MM: {YYYY-MM-DD: [runs...]}} in descending order when iterated."""
    out: Dict[str, Dict[str, List[Run]]] = {}
    for r in runs:
        month = r.day[:7]
        out.setdefault(month, {}).setdefault(r.day, []).append(r)
    # Sort runs within day newest-first for nicer nav.
    for month in out:
        for day in out[month]:
            out[month][day].sort(key=lambda x: x.dt, reverse=True)
    return out


def nav_header(title: str, mtime: Optional[float]) -> str:
    updated = "" if mtime is None else html.escape(_http_date(mtime))
    links = (
        "<a href='/'>latest</a> · "
        "<a href='/runs'>runs</a> · "
        "<a href='/raw'>raw</a>"
    )
    meta = f"Updated: <span>{updated}</span> · {links}" if updated else links
    return (
        "<header>"
        f"  <h1>{html.escape(title)}</h1>"
        f"  <div class='meta'>{meta}</div>"
        "</header>"
    )


def page_html(title: str, body_html: str, updated: float | None) -> str:
    updated_meta = "" if updated is None else _http_date(updated)
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <meta http-equiv='refresh' content='{REFRESH_SECS}'>
  <title>{html.escape(title)}</title>
  <link rel='stylesheet' href='/static/style.css'>
  <meta name='robots' content='noindex,nofollow'>
  <meta name='last-updated' content='{html.escape(updated_meta)}'>
</head>
<body>
  <div class='wrap'>
    {body_html}
  </div>
</body>
</html>"""


STYLE_CSS = """
:root { color-scheme: light dark; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; line-height: 1.55; }
.wrap { max-width: 980px; margin: 0 auto; padding: 24px 18px 64px; }
header { display: flex; gap: 16px; justify-content: space-between; align-items: baseline; flex-wrap: wrap; border-bottom: 1px solid rgba(127,127,127,.35); padding-bottom: 12px; margin-bottom: 18px; }
h1 { font-size: 28px; margin: 0; letter-spacing: -0.02em; }
.meta { font-size: 14px; opacity: .78; }
.meta a { color: inherit; }
.content h1, .content h2, .content h3 { margin-top: 28px; }
.content h2 { border-bottom: 1px solid rgba(127,127,127,.25); padding-bottom: 6px; }
.content pre, .content code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.content pre { padding: 12px 14px; border-radius: 10px; overflow-x: auto; background: rgba(127,127,127,.12); }
.content code { background: rgba(127,127,127,.12); padding: 2px 5px; border-radius: 6px; }
.content blockquote { margin: 14px 0; padding: 8px 12px; border-left: 4px solid rgba(127,127,127,.35); opacity: .9; }
.content table { border-collapse: collapse; width: 100%; overflow-x: auto; display: block; }
.content th, .content td { border: 1px solid rgba(127,127,127,.25); padding: 8px 10px; }
.content a { text-decoration: none; }
.content a:hover { text-decoration: underline; }
.warn { padding: 10px 12px; border-radius: 10px; background: rgba(255, 200, 0, .15); border: 1px solid rgba(255,200,0,.35); margin-bottom: 14px; }
.muted { opacity: .7; }

.pager { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 10px 0 18px; }
.pager a { color: inherit; text-decoration:none; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(127,127,127,.25); }
.pager a:hover { text-decoration: none; border-color: rgba(127,127,127,.55); }
.pager .dim { opacity: .6; border-style: dashed; }

.index h2 { margin-top: 22px; }
.index .day { margin: 10px 0 18px; padding-left: 0; list-style: none; }
.index .day li { margin: 6px 0; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
.badge { font-size: 12px; opacity: .85; padding: 2px 8px; border: 1px solid rgba(127,127,127,.25); border-radius: 999px; }
"""


class DigestHandler(BaseHTTPRequestHandler):
    server_version = "phi4mini-digest/1.1"

    # Simple in-process cache to avoid re-parsing/re-rendering on every request.
    _cache_mtime: float | None = None
    _cache_md: str | None = None
    _cache_runs: List[Run] | None = None

    def log_message(self, fmt: str, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _ensure_cache(self) -> Tuple[Optional[float], str, List[Run]]:
        if not DIGEST_MD.exists():
            return None, "", []
        st = DIGEST_MD.stat()
        mtime = st.st_mtime
        if self._cache_mtime != mtime or self._cache_md is None or self._cache_runs is None:
            md_text = DIGEST_MD.read_text(encoding="utf-8", errors="replace")
            runs = parse_runs(md_text)
            self._cache_mtime = mtime
            self._cache_md = md_text
            self._cache_runs = runs
        return self._cache_mtime, self._cache_md or "", self._cache_runs or []

    def do_GET(self):
        raw_path = self.path
        path, _, qs = raw_path.partition("?")
        path = path or "/"
        q = urllib.parse.parse_qs(qs)

        if path == "/healthz":
            self._send(200, "text/plain; charset=utf-8", "ok\n")
            return

        if path == "/static/style.css":
            self._send(200, "text/css; charset=utf-8", STYLE_CSS)
            return

        if path == "/raw":
            if not DIGEST_MD.exists():
                self._send(404, "text/plain; charset=utf-8", f"missing: {DIGEST_MD}\n")
                return
            self._send_file(DIGEST_MD, "text/markdown; charset=utf-8")
            return

        mtime, md_text, runs = self._ensure_cache()

        if not DIGEST_MD.exists():
            body = (
                nav_header(TITLE, None)
                + f"<p class='muted'>No digest yet. Waiting for {html.escape(str(DIGEST_MD))}.</p>"
            )
            self._send(200, "text/html; charset=utf-8", page_html(TITLE, body, updated=None))
            return

        warn = "" if MARKDOWN_OK else (
            "<div class='warn'>Python package <code>markdown</code> is not installed; "
            "showing plain text. Run <code>/opt/phi4mini/ops/setup_web.sh</code>.</div>"
        )

        # Latest (default)
        if path in ("/", ""):
            if not runs:
                rendered = render_markdown(md_text)
                body = warn + nav_header(TITLE, mtime) + f"<main class='content'>{rendered}</main>"
                self._send(200, "text/html; charset=utf-8", page_html(TITLE, body, updated=mtime))
                return
            latest = runs[-1]
            self._send(200, "text/html; charset=utf-8", self._render_run_page(latest, runs, mtime, warn))
            return

        # Index
        if path == "/runs":
            body = warn + nav_header(TITLE, mtime) + self._render_index(runs)
            self._send(200, "text/html; charset=utf-8", page_html(TITLE, body, updated=mtime))
            return

        # /run/<id>
        if path.startswith("/run/"):
            run_id = path[len("/run/") :].strip("/")
            run = next((r for r in runs if r.id == run_id), None)
            if run is None:
                self._send(404, "text/plain; charset=utf-8", "run not found\n")
                return
            self._send(200, "text/html; charset=utf-8", self._render_run_page(run, runs, mtime, warn))
            return

        # /day/<YYYY-MM-DD> -> show only that day's runs (index view)
        if path.startswith("/day/"):
            day = path[len("/day/") :].strip("/")
            day_runs = [r for r in runs if r.day == day]
            if not day_runs:
                self._send(404, "text/plain; charset=utf-8", "day not found\n")
                return
            # Render a small day index (newest first)
            day_runs.sort(key=lambda r: r.dt, reverse=True)
            items = []
            for r in day_runs:
                cov = f" <span class='badge'>{html.escape(r.coverage)}</span>" if r.coverage else ""
                items.append(f"<li><a href='/run/{html.escape(r.id)}'>{html.escape(r.time)}</a>{cov}</li>")
            body = (
                warn
                + nav_header(TITLE, mtime)
                + f"<div class='index'><h2>{html.escape(day)}</h2><ul class='day'>{''.join(items)}</ul></div>"
            )
            self._send(200, "text/html; charset=utf-8", page_html(TITLE, body, updated=mtime))
            return

        self._send(404, "text/plain; charset=utf-8", "not found\n")

    def _render_index(self, runs: List[Run]) -> str:
        if not runs:
            return "<p class='muted'>No runs detected yet.</p>"

        grouped = group_runs(runs)
        # Sort months newest-first
        months = sorted(grouped.keys(), reverse=True)

        parts = ["<div class='index'>"]
        for month in months:
            parts.append(f"<h2>{html.escape(month)}</h2>")
            # days newest-first
            days = sorted(grouped[month].keys(), reverse=True)
            for day in days:
                parts.append(f"<h3><a href='/day/{html.escape(day)}'>{html.escape(day)}</a></h3>")
                parts.append("<ul class='day'>")
                for r in grouped[month][day]:
                    cov = f" <span class='badge'>{html.escape(r.coverage)}</span>" if r.coverage else ""
                    parts.append(
                        f"<li><a href='/run/{html.escape(r.id)}'>{html.escape(r.time)}</a>{cov}</li>"
                    )
                parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _render_run_page(self, run: Run, runs: List[Run], mtime: Optional[float], warn: str) -> str:
        idx = next((i for i, r in enumerate(runs) if r.id == run.id), -1)
        prev_id = runs[idx - 1].id if idx > 0 else ""
        next_id = runs[idx + 1].id if 0 <= idx < len(runs) - 1 else ""

        pager = ["<div class='pager'>"]
        pager.append("<a href='/runs'>All runs</a>")
        if prev_id:
            pager.append(f"<a href='/run/{html.escape(prev_id)}'>← Prev</a>")
        else:
            pager.append("<span class='dim'>← Prev</span>")
        if next_id:
            pager.append(f"<a href='/run/{html.escape(next_id)}'>Next →</a>")
        else:
            pager.append("<span class='dim'>Next →</span>")
        pager.append("</div>")

        # Render just this run's markdown.
        rendered = render_markdown(run.md)
        subtitle = f"{run.day} {run.time}"
        body = (
            warn
            + nav_header(TITLE, mtime)
            + "".join(pager)
            + f"<h2 class='muted'>{html.escape(subtitle)}</h2>"
            + f"<main class='content'>{rendered}</main>"
        )
        return page_html(TITLE, body, updated=mtime)

    def _send(self, code: int, content_type: str, body: str):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Last-Modified", _http_date(path.stat().st_mtime))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    host = os.environ.get("PHI4MINI_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("PHI4MINI_WEB_PORT", "8088"))

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    with ThreadingHTTPServer((host, port), DigestHandler) as httpd:
        sys.stderr.write(f"Serving {DIGEST_MD} on http://{host}:{port}/ (pagination enabled)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
