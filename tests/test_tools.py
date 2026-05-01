"""Tests for research tools."""

from research_agent.tools import _format_tavily_results, web_search


def test_web_search_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = web_search.invoke({"query": "AI chips market"})

    assert result == "Error: TAVILY_API_KEY is required for web_search."


def test_format_tavily_results_includes_sources() -> None:
    payload = {
        "answer": "AI infrastructure spending is growing.",
        "results": [
            {
                "title": "AI Infrastructure Report",
                "url": "https://example.com/report",
                "content": "Cloud and accelerator demand remain key drivers.",
                "score": 0.91,
            }
        ],
    }

    result = _format_tavily_results("AI infrastructure market", payload)

    assert "Search query: AI infrastructure market" in result
    assert "Summary: AI infrastructure spending is growing." in result
    assert "Title: AI Infrastructure Report" in result
    assert "URL: https://example.com/report" in result
    assert "Cloud and accelerator demand remain key drivers." in result
