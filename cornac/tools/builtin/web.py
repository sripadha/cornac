"""Web tools: web_search, web_fetch.

These need no Workspace (they touch the network, not the filesystem), so they're plain
@tool functions. web_search uses DuckDuckGo via the `ddgs` library — free, no API key,
so anyone can reproduce the benchmark. web_fetch retrieves a URL and strips it down to
readable text.

Network calls fail in many ways (timeouts, rate limits, dead links). We catch those and
return them as readable strings; the registry would also catch a raw exception, but
returning a clear message gives the model something it can actually act on.
"""

from __future__ import annotations

import re

import httpx

from cornac.tools.base import tool

FETCH_TIMEOUT = 20.0
MAX_FETCH_CHARS = 20_000


@tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) and return the top results as title / url / snippet."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 — network/library errors -> readable result
        return f"Error: web search failed ({type(exc).__name__}: {exc})"

    if not results:
        return f"(no results for {query!r})"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "") or r.get("url", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return "\n".join(lines)


@tool()
def web_fetch(url: str) -> str:
    """Fetch a web page by URL and return its readable text content (HTML tags stripped)."""
    try:
        resp = httpx.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True,
                         headers={"User-Agent": "cornac-agent/0.1"})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Error: fetch failed ({type(exc).__name__}: {exc})"

    text = resp.text
    # Crude HTML -> text: drop script/style, strip tags, collapse whitespace. Good
    # enough for an agent to read; a heavier dependency (e.g. trafilatura) is overkill.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + "\n... [truncated]"
    return text or "(page had no readable text)"
