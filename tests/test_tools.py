"""Tests for research tools."""

from research_agent.tools import _format_tavily_results, think, web_search


def test_think_is_deterministic() -> None:
    result = think.invoke({"reflection": "Need stronger source coverage."})

    assert result == "Reflection recorded: Need stronger source coverage."


def test_web_search_rejects_invalid_max_results() -> None:
    result = web_search.invoke({"query": "AI chips market", "max_results": 0})

    assert result == "Error: max_results must be at least 1."


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
