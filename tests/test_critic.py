"""Tests for the critic-review node's JSON-parsing logic.

These tests avoid the network and the LLM by exercising the parser
directly. The integration with an LLM is covered by
``test_graph_integration.py``.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.nodes import _parse_critic_response


def test_parses_clean_json():
    raw = (
        '{"verdict": "supported", '
        '"summary": "All claims traced to evidence.", '
        '"flags": []}'
    )
    review = _parse_critic_response(raw)
    assert review.verdict == "supported"
    assert review.summary == "All claims traced to evidence."
    assert review.flags == []


def test_parses_json_with_flags():
    raw = (
        '{"verdict": "partially_supported", '
        '"summary": "One overclaim.", '
        '"flags": [{"severity": "error", "claim": "pathogenic", '
        '"concern": "ACMG call was VUS", "suggestion": "relabel"}]}'
    )
    review = _parse_critic_response(raw)
    assert review.verdict == "partially_supported"
    assert len(review.flags) == 1
    f = review.flags[0]
    assert f.severity == "error"
    assert f.claim == "pathogenic"


def test_extracts_json_embedded_in_prose():
    """LLMs sometimes wrap JSON in explanatory text. We should cope."""
    raw = (
        "Here is my review:\n\n"
        '{"verdict": "supported", "summary": "Fine.", "flags": []}\n\n'
        "Hope this helps!"
    )
    review = _parse_critic_response(raw)
    assert review.verdict == "supported"


def test_degrades_on_malformed_json():
    raw = "this is not JSON at all"
    review = _parse_critic_response(raw)
    assert review.verdict == "partially_supported"
    assert "non-JSON" in review.summary


def test_degrades_on_broken_json():
    raw = '{"verdict": "supported", "summary": "oops"'  # missing closing brace
    review = _parse_critic_response(raw)
    assert review.verdict == "partially_supported"


def test_coerces_unknown_verdict():
    raw = '{"verdict": "approved", "summary": "...", "flags": []}'
    review = _parse_critic_response(raw)
    assert review.verdict == "partially_supported"


def test_coerces_unknown_severity():
    raw = (
        '{"verdict": "unsupported", "summary": "X", '
        '"flags": [{"severity": "blocker", "claim": "c", '
        '"concern": "c2", "suggestion": "s"}]}'
    )
    review = _parse_critic_response(raw)
    assert review.flags[0].severity == "warn"


def test_truncates_very_long_fields():
    """Critic output should never let a malicious LLM blow up memory."""
    huge = "x" * 100_000
    raw = (
        f'{{"verdict": "unsupported", "summary": "{huge}", '
        f'"flags": [{{"severity": "error", "claim": "{huge}", '
        f'"concern": "{huge}", "suggestion": "{huge}"}}]}}'
    )
    review = _parse_critic_response(raw)
    assert len(review.summary) <= 1000
    assert len(review.flags[0].claim) <= 800
    assert len(review.flags[0].suggestion) <= 400
