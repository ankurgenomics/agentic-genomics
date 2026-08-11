"""Unit tests for ``reviewer2_client._review_one``'s dossier parsing.

Regression coverage for a real bug caught while building the second-opinion
feature: Reviewer2 can raise a MAJOR-severity conflict flag for reasons that
have nothing to do with the two engines disagreeing (e.g. ClinVar itself
having conflicting submitters). Only a ``classification_disagreement`` flag
at major/critical severity should ever set ``materially_disagrees`` — these
tests pin that down with canned MCP tool responses (no subprocess, no network).
"""

from __future__ import annotations

import json

import pytest

from agentic_genomics.agents.variant_interpreter.state import AnnotatedVariant, Variant
from agentic_genomics.agents.variant_interpreter.tools import reviewer2_client


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolResult:
    def __init__(self, payload: dict) -> None:
        self.content = [_TextBlock(json.dumps(payload))]


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_call_args: dict | None = None

    async def call_tool(self, _name: str, arguments: dict):
        self.last_call_args = arguments
        return _ToolResult(self._payload)


def _mk_variant(gene: str = "PALB2") -> AnnotatedVariant:
    return AnnotatedVariant(variant=Variant(chrom="chr16", pos=1, ref="C", alt="T", gene=gene))


@pytest.mark.asyncio
async def test_clinvar_submitter_conflict_alone_is_not_a_material_disagreement():
    """Both engines agree on 'Likely benign', but ClinVar itself is messy.

    That must surface as an informational conflict, not as
    ``materially_disagrees`` — the whole point of the flag is "the two
    independent engines reached different clinical actions".
    """
    payload = {
        "independent_classification": "Likely benign",
        "disagreement_score": 0.0,
        "conflicts": [
            {
                "type": "clinvar_submitter_conflict",
                "severity": "major",
                "message": "ClinVar reports conflicting interpretations among submitters.",
            }
        ],
        "summary": "Concordant with proposed call.",
    }
    session = _FakeSession(payload)
    result = await reviewer2_client._review_one(session, _mk_variant())

    assert result.available is True
    assert result.independent_classification == "Likely benign"
    assert result.materially_disagrees is False
    assert len(result.conflicts) == 1
    assert result.conflicts[0].type == "clinvar_submitter_conflict"


@pytest.mark.asyncio
async def test_classification_disagreement_at_critical_severity_is_material():
    payload = {
        "independent_classification": "Uncertain significance",
        "disagreement_score": 0.5,
        "conflicts": [
            {
                "type": "classification_disagreement",
                "severity": "critical",
                "message": "Proposed call 'Pathogenic' disagrees with the independent call "
                "'Uncertain significance' and crosses the clinical-actionability line.",
            }
        ],
        "summary": "Escalate for human review.",
    }
    session = _FakeSession(payload)
    result = await reviewer2_client._review_one(session, _mk_variant(gene="SCN1A"))

    assert result.materially_disagrees is True


@pytest.mark.asyncio
async def test_minor_tier_only_disagreement_is_not_material():
    """Pathogenic vs Likely pathogenic — same action band, MINOR severity."""
    payload = {
        "independent_classification": "Likely pathogenic",
        "disagreement_score": 0.25,
        "conflicts": [
            {
                "type": "classification_disagreement",
                "severity": "minor",
                "message": "Same clinical action band, management unchanged.",
            }
        ],
        "summary": "Adjacent tier.",
    }
    session = _FakeSession(payload)
    result = await reviewer2_client._review_one(session, _mk_variant(gene="STK11"))

    assert result.materially_disagrees is False


@pytest.mark.asyncio
async def test_review_one_degrades_on_malformed_response():
    class _BrokenSession:
        async def call_tool(self, _name: str, arguments: dict):
            raise RuntimeError("boom")

    result = await reviewer2_client._review_one(_BrokenSession(), _mk_variant())
    assert result.available is False
    assert "boom" in (result.error or "")
