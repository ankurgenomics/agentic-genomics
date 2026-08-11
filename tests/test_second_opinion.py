"""Unit tests for the ``second_opinion_review`` node (Reviewer2 via MCP).

These run fully offline: ``reviewer2_client.get_second_opinions`` is
monkeypatched so no subprocess or network call ever happens here. The point
is to verify the node's own contract — it must never crash the pipeline and
must correctly tally concordant/flagged/missing outcomes.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter import nodes as nodes_mod
from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    SecondOpinion,
    SecondOpinionConflict,
    Variant,
    VariantInterpreterState,
)


def _mk_state(*genes: str) -> VariantInterpreterState:
    variants = [
        AnnotatedVariant(variant=Variant(chrom="1", pos=i + 1, ref="A", alt="T", gene=g))
        for i, g in enumerate(genes)
    ]
    return VariantInterpreterState(vcf_path="unused.vcf", variants=variants)


def test_second_opinion_review_merges_results(monkeypatch):
    state = _mk_state("SCN1A", "BRCA2")
    scn1a_key = state.variants[0].variant.key
    brca2_key = state.variants[1].variant.key

    async def _fake_get_second_opinions(_variants):
        return {
            scn1a_key: SecondOpinion(
                available=True,
                independent_classification="Uncertain significance",
                materially_disagrees=True,
                conflicts=[
                    SecondOpinionConflict(
                        type="classification_disagreement",
                        severity="major",
                        message="Proposed call disagrees with the independent call.",
                    )
                ],
            ),
            brca2_key: SecondOpinion(
                available=True,
                independent_classification="Benign",
                materially_disagrees=False,
            ),
        }

    monkeypatch.setattr(
        nodes_mod.reviewer2_client, "get_second_opinions", _fake_get_second_opinions
    )

    update = nodes_mod.second_opinion_review(state)
    updated_variants = update["variants"]

    scn1a = next(av for av in updated_variants if av.variant.gene == "SCN1A")
    brca2 = next(av for av in updated_variants if av.variant.gene == "BRCA2")

    assert scn1a.second_opinion is not None
    assert scn1a.second_opinion.available is True
    assert scn1a.second_opinion.materially_disagrees is True
    assert brca2.second_opinion is not None
    assert brca2.second_opinion.materially_disagrees is False

    step = update["reasoning_trace"][-1]
    assert step["node"] == "second_opinion_review"
    assert step["evidence"]["concordant"] == 1
    assert step["evidence"]["flagged"] == 1
    assert step["evidence"]["missing"] == 0


def test_second_opinion_review_missing_entry_marked_unavailable(monkeypatch):
    """A variant with no key in the results dict must not crash — just unavailable."""
    state = _mk_state("TP53")

    async def _fake_get_second_opinions(_variants):
        return {}

    monkeypatch.setattr(
        nodes_mod.reviewer2_client, "get_second_opinions", _fake_get_second_opinions
    )

    update = nodes_mod.second_opinion_review(state)
    av = update["variants"][0]
    assert av.second_opinion is not None
    assert av.second_opinion.available is False
    assert update["reasoning_trace"][-1]["evidence"]["missing"] == 1


def test_second_opinion_review_degrades_gracefully_on_exception(monkeypatch):
    """If the MCP call itself raises (subprocess failure, etc.), the whole
    pipeline must keep going with every variant marked unavailable — never crash.
    """
    state = _mk_state("PALB2")

    async def _raising_get_second_opinions(_variants):
        raise RuntimeError("Reviewer2 MCP server failed to start")

    monkeypatch.setattr(
        nodes_mod.reviewer2_client, "get_second_opinions", _raising_get_second_opinions
    )

    update = nodes_mod.second_opinion_review(state)
    av = update["variants"][0]
    assert av.second_opinion is not None
    assert av.second_opinion.available is False
    assert "Reviewer2 MCP server failed to start" in (av.second_opinion.error or "")
    assert update["reasoning_trace"][-1]["node"] == "second_opinion_review"
