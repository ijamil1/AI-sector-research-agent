"""Custom tools for source-backed AI sector research."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _format_tavily_results(query: str, payload: dict[str, Any]) -> str:
    """Format Tavily search payloads into compact source-backed notes.

    Args:
        query: Original search query.
        payload: Tavily JSON response.

    Returns:
        Human-readable search result text for the model.
    """
    answer = payload.get("answer")
    results = payload.get("results") or []

    sections = [f"Search query: {query}"]
    if answer:
        sections.append(f"Summary: {answer}")

    if not results:
        sections.append("No search results returned.")
        return "\n\n".join(sections)

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or "No URL"
        content = result.get("content") or result.get("raw_content") or "No content snippet."
        score = result.get("score")
        score_text = f" score={score}" if score is not None else ""
        sections.append(
            f"Result {index}:{score_text}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(sections)


def _search_tavily(query: str, max_results: int) -> dict[str, Any]:
    """Call Tavily's search API.

    Args:
        query: Search query.
        max_results: Maximum number of results to request.

    Returns:
        Tavily JSON response.

    Raises:
        RuntimeError: If `TAVILY_API_KEY` is not configured.
        httpx.HTTPError: If the Tavily request fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        msg = "TAVILY_API_KEY is required for web_search."
        raise RuntimeError(msg)

    response = httpx.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        },
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()


@tool(parse_docstring=True)
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for source-backed information.

    Args:
        query: Search query to execute.
        max_results: Maximum number of results to return.

    Returns:
        Compact source-backed search results with URLs.
    """
    if max_results < 1:
        return "Error: max_results must be at least 1."
    if max_results > 10:
        return "Error: max_results must be 10 or less."

    try:
        payload = _search_tavily(query, max_results)
    except RuntimeError as exc:
        return f"Error: {exc}"
    except httpx.HTTPError as exc:
        return f"Error: web search request failed: {exc}"

    return _format_tavily_results(query, payload)


@tool(parse_docstring=True)
def think(reflection: str) -> str:
    """Record a strategic research reflection.

    Args:
        reflection: Assessment of findings, gaps, evidence quality, and next steps.

    Returns:
        Confirmation that the reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"
