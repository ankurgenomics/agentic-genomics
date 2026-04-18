"""Unit tests for the deterministic ACMG-lite classifier.

Covers:
  - frequency criteria (PM2_Supporting, BS1, BA1)
  - in-silico criteria (PP3, BP4)
  - ClinVar criteria (PP5, BP6, conflicting)
  - PVS1 (null consequence in a LoF-intolerant gene)
  - Richards et al. 2015 Table 5 combining rules

These tests run offline. Live-API tests are gated behind the ``network`` marker.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    ClinicalEvidence,
    FunctionalScores,
    PopulationFrequency,
    Variant,
)
from agentic_genomics.agents.variant_interpreter.tools import acmg_lite


def _mk(
    gnomad_af: float | None = None,
    gnomad_af_popmax: float | None = None,
    cadd: float | None = None,
    revel: float | None = None,
    spliceai: float | None = None,
    clinvar: str | None = None,
    consequence: str | None = None,
    pli: float | None = None,
    loeuf: float | None = None,
    gene: str | None = "FOO",
) -> AnnotatedVariant:
    return AnnotatedVariant(
        variant=Variant(
            chrom="1", pos=1, ref="A", alt="T",
            gene=gene, consequence=consequence,
        ),
        population=PopulationFrequency(
            gnomad_af=gnomad_af, gnomad_af_popmax=gnomad_af_popmax
        ),
        functional=FunctionalScores(
            cadd_phred=cadd, revel=revel, spliceai_max=spliceai,
            gnomad_pli=pli, gnomad_loeuf=loeuf,
        ),
        clinical=ClinicalEvidence(clinvar_significance=clinvar),
    )


# --- Frequency criteria -------------------------------------------------

def test_absent_from_gnomad_triggers_pm2():
    a = acmg_lite.classify(_mk())
    assert "PM2_Supporting" in a.criteria_triggered


def test_common_variant_is_benign():
    a = acmg_lite.classify(_mk(gnomad_af_popmax=0.06))
    assert "BA1" in a.criteria_triggered
    assert a.call == "Benign"


def test_moderately_common_triggers_bs1():
    a = acmg_lite.classify(_mk(gnomad_af_popmax=0.02))
    assert "BS1" in a.criteria_triggered


# --- In-silico criteria -------------------------------------------------

def test_multiple_deleterious_predictors_trigger_pp3():
    a = acmg_lite.classify(_mk(cadd=28, revel=0.85, spliceai=0.6))
    assert "PP3" in a.criteria_triggered


def test_multiple_benign_predictors_trigger_bp4():
    a = acmg_lite.classify(_mk(cadd=5, revel=0.1, spliceai=0.01))
    assert "BP4" in a.criteria_triggered


# --- ClinVar criteria ---------------------------------------------------

def test_clinvar_pathogenic_triggers_pp5():
    a = acmg_lite.classify(_mk(clinvar="Likely pathogenic"))
    assert "PP5" in a.criteria_triggered


def test_clinvar_conflicting_does_not_trigger_pp5_or_bp6():
    a = acmg_lite.classify(_mk(clinvar="Pathogenic, Benign"))
    assert "PP5" not in a.criteria_triggered
    assert "BP6" not in a.criteria_triggered


# --- PVS1 (null variant in LoF-intolerant gene) ------------------------

def test_pvs1_triggers_on_stop_gained_in_lof_intolerant_gene():
    a = acmg_lite.classify(
        _mk(consequence="stop_gained", pli=0.99, gene="SCN1A")
    )
    assert "PVS1" in a.criteria_triggered


def test_pvs1_triggers_on_frameshift_with_loeuf_evidence():
    a = acmg_lite.classify(
        _mk(consequence="frameshift_variant", loeuf=0.2, gene="TP53")
    )
    assert "PVS1" in a.criteria_triggered


def test_pvs1_triggers_on_canonical_splice_donor():
    a = acmg_lite.classify(
        _mk(consequence="splice_donor_variant", pli=0.95)
    )
    assert "PVS1" in a.criteria_triggered


def test_pvs1_does_not_trigger_on_missense():
    a = acmg_lite.classify(
        _mk(consequence="missense_variant", pli=0.99)
    )
    assert "PVS1" not in a.criteria_triggered


def test_pvs1_does_not_trigger_without_constraint_evidence():
    """Null variant but no pLI/LOEUF data → we can't assert LoF intolerance."""
    a = acmg_lite.classify(_mk(consequence="stop_gained"))
    assert "PVS1" not in a.criteria_triggered


def test_pvs1_does_not_trigger_when_gene_is_lof_tolerant():
    a = acmg_lite.classify(
        _mk(consequence="stop_gained", pli=0.1, loeuf=1.5)
    )
    assert "PVS1" not in a.criteria_triggered


# --- Combining rules (Richards 2015 Table 5) ---------------------------

def test_pvs1_alone_is_likely_pathogenic():
    """Per ClinGen SVI 2018, PVS1 alone can support LP — we surface it as LP."""
    a = acmg_lite.classify(
        _mk(consequence="stop_gained", pli=0.99)
    )
    assert a.call == "Likely Pathogenic"


def test_pvs1_plus_pm2_is_likely_pathogenic():
    a = acmg_lite.classify(
        _mk(consequence="stop_gained", pli=0.99)  # PVS1 + PM2 (absent)
    )
    # PVS1 + PM2 (Moderate) → Likely Pathogenic per Richards Table 5.
    assert a.call == "Likely Pathogenic"
    assert "PVS1" in a.criteria_triggered
    assert "PM2_Supporting" in a.criteria_triggered


def test_strong_pathogenic_stack_without_pvs1_is_at_least_lp():
    """PP3 + PP5 + PM2 = 1 Moderate + 2 Supporting, which doesn't quite reach LP.

    Without PVS1 and without a true Strong criterion, even three supporting
    lines don't cross the Likely-Pathogenic threshold under Richards 2015.
    The classifier should stay conservative and return VUS.
    """
    a = acmg_lite.classify(
        _mk(cadd=30, revel=0.9, spliceai=0.8, clinvar="Likely pathogenic")
    )
    # PP3, PP5, PM2 → M=1, SUP=2, VS=0, S=0 → no combining rule fires.
    assert a.call == "Uncertain Significance"
    assert "PP3" in a.criteria_triggered
    assert "PP5" in a.criteria_triggered
    assert "PM2_Supporting" in a.criteria_triggered


def test_two_supporting_benign_is_likely_benign():
    """BP4 + BP6 = 2 Supporting benign → Likely Benign."""
    a = acmg_lite.classify(
        _mk(gnomad_af_popmax=0.005,  # Not rare enough for PM2, not common enough for BS1
            cadd=5, revel=0.1, spliceai=0.01,  # BP4
            clinvar="Likely benign")  # BP6
    )
    assert a.call == "Likely Benign"
    assert "BP4" in a.criteria_triggered
    assert "BP6" in a.criteria_triggered


def test_rationale_surfaces_caveat():
    """Every rationale string must surface the acmg-lite caveat."""
    a = acmg_lite.classify(_mk(gnomad_af_popmax=0.06))
    assert "acmg-lite" in a.rationale
