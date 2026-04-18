"""Fast unit tests for the deterministic ACMG-lite classifier.

These tests run offline and exercise boundary conditions. The LLM and
live-API tests are gated behind the ``network`` pytest marker.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    ClinicalEvidence,
    FunctionalScores,
    PopulationFrequency,
    Variant,
)
from agentic_genomics.agents.variant_interpreter.tools import acmg


def _mk(
    gnomad_af: float | None = None,
    gnomad_af_popmax: float | None = None,
    cadd: float | None = None,
    revel: float | None = None,
    spliceai: float | None = None,
    clinvar: str | None = None,
) -> AnnotatedVariant:
    return AnnotatedVariant(
        variant=Variant(chrom="1", pos=1, ref="A", alt="T"),
        population=PopulationFrequency(
            gnomad_af=gnomad_af, gnomad_af_popmax=gnomad_af_popmax
        ),
        functional=FunctionalScores(
            cadd_phred=cadd, revel=revel, spliceai_max=spliceai
        ),
        clinical=ClinicalEvidence(clinvar_significance=clinvar),
    )


def test_absent_from_gnomad_triggers_pm2():
    a = acmg.classify(_mk())
    assert "PM2_Supporting" in a.criteria_triggered


def test_common_variant_is_benign():
    a = acmg.classify(_mk(gnomad_af_popmax=0.06))
    assert "BA1" in a.criteria_triggered
    assert a.call == "Benign"


def test_moderately_common_triggers_bs1():
    a = acmg.classify(_mk(gnomad_af_popmax=0.02))
    assert "BS1" in a.criteria_triggered


def test_multiple_deleterious_predictors_trigger_pp3():
    a = acmg.classify(_mk(cadd=28, revel=0.85, spliceai=0.6))
    assert "PP3" in a.criteria_triggered


def test_multiple_benign_predictors_trigger_bp4():
    a = acmg.classify(_mk(cadd=5, revel=0.1, spliceai=0.01))
    assert "BP4" in a.criteria_triggered


def test_clinvar_pathogenic_triggers_pp5():
    a = acmg.classify(_mk(clinvar="Likely pathogenic"))
    assert "PP5" in a.criteria_triggered


def test_clinvar_conflicting_does_not_trigger_pp5_or_bp6():
    a = acmg.classify(_mk(clinvar="Pathogenic, Benign"))
    assert "PP5" not in a.criteria_triggered
    assert "BP6" not in a.criteria_triggered


def test_strong_pathogenic_stack_escalates_call():
    a = acmg.classify(
        _mk(cadd=30, revel=0.9, spliceai=0.8, clinvar="Likely pathogenic")
    )
    assert a.call in {"Likely Pathogenic", "Pathogenic"}
    assert a.rationale  # non-empty narrative
