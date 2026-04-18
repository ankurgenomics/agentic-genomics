"""Deterministic ACMG-style classifier.

This is **not** a certified implementation of ACMG/AMP 2015 or ClinGen
expert-panel refinements. It is a *pragmatic, transparent* rule engine
that encodes a small subset of well-defined criteria:

  - PM2_Supporting: absent / ultra-rare in gnomAD
  - BA1:            allele frequency >= 5% in gnomAD
  - BS1:            allele frequency >= 1% in gnomAD
  - PP3:            multiple in-silico predictors support deleterious
  - BP4:            multiple in-silico predictors support benign
  - PP5:            reputable source reports as pathogenic (ClinVar LP/P)
  - BP6:            reputable source reports as benign  (ClinVar LB/B)
  - (phenotype_match is used by the LLM, not here)

The output is a transparent list of triggered criteria + a final call,
which the LLM synthesiser later turns into a readable narrative.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.state import (
    ACMGAssessment,
    AnnotatedVariant,
)

_PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic"}
_BENIGN_TERMS = {"benign", "likely benign"}


def _significance_set(sig: str | None) -> set[str]:
    if not sig:
        return set()
    return {s.strip().lower() for s in sig.split(",") if s.strip()}


def _evaluate_frequency(v: AnnotatedVariant) -> list[str]:
    af = v.population.gnomad_af_popmax or v.population.gnomad_af
    if af is None:
        return ["PM2_Supporting"]  # absent from gnomAD
    triggered = []
    if af >= 0.05:
        triggered.append("BA1")
    elif af >= 0.01:
        triggered.append("BS1")
    elif af < 1e-4:
        triggered.append("PM2_Supporting")
    return triggered


def _evaluate_insilico(v: AnnotatedVariant) -> list[str]:
    f = v.functional
    deleterious = 0
    benign = 0

    if f.cadd_phred is not None:
        if f.cadd_phred >= 20:
            deleterious += 1
        elif f.cadd_phred < 10:
            benign += 1

    if f.revel is not None:
        if f.revel >= 0.7:
            deleterious += 1
        elif f.revel <= 0.2:
            benign += 1

    if f.spliceai_max is not None:
        if f.spliceai_max >= 0.5:
            deleterious += 1
        elif f.spliceai_max < 0.1:
            benign += 1

    if f.sift == "D":
        deleterious += 1
    if f.sift == "T":
        benign += 1
    if f.polyphen in {"D", "P"}:
        deleterious += 1
    if f.polyphen == "B":
        benign += 1

    triggered = []
    if deleterious >= 2:
        triggered.append("PP3")
    if benign >= 2:
        triggered.append("BP4")
    return triggered


def _evaluate_clinvar(v: AnnotatedVariant) -> list[str]:
    sigs = _significance_set(v.clinical.clinvar_significance)
    if not sigs:
        return []
    if sigs & _PATHOGENIC_TERMS and not (sigs & _BENIGN_TERMS):
        return ["PP5"]
    if sigs & _BENIGN_TERMS and not (sigs & _PATHOGENIC_TERMS):
        return ["BP6"]
    return []  # conflicting — no clean ClinVar contribution


def _final_call(criteria: list[str]) -> str:
    pathogenic_hits = sum(1 for c in criteria if c.startswith(("PS", "PM", "PP")))
    benign_hits = sum(1 for c in criteria if c.startswith(("BA", "BS", "BP")))

    if "BA1" in criteria:
        return "Benign"
    if benign_hits >= 2 and pathogenic_hits == 0:
        return "Likely Benign"
    if pathogenic_hits >= 3 and benign_hits == 0:
        return "Likely Pathogenic"
    if pathogenic_hits >= 4 and benign_hits == 0:
        return "Pathogenic"
    return "Uncertain Significance"


def classify(v: AnnotatedVariant) -> ACMGAssessment:
    """Run the deterministic ACMG-lite classifier."""
    criteria: list[str] = []
    criteria.extend(_evaluate_frequency(v))
    criteria.extend(_evaluate_insilico(v))
    criteria.extend(_evaluate_clinvar(v))

    call = _final_call(criteria)

    rationale_bits = []
    if "PM2_Supporting" in criteria:
        rationale_bits.append("absent/ultra-rare in gnomAD")
    if "BA1" in criteria:
        rationale_bits.append("common in gnomAD (AF ≥ 5%)")
    if "BS1" in criteria:
        rationale_bits.append("relatively common in gnomAD (AF ≥ 1%)")
    if "PP3" in criteria:
        rationale_bits.append("multiple in-silico predictors agree it is deleterious")
    if "BP4" in criteria:
        rationale_bits.append("multiple in-silico predictors agree it is benign")
    if "PP5" in criteria:
        rationale_bits.append("ClinVar reports (likely) pathogenic")
    if "BP6" in criteria:
        rationale_bits.append("ClinVar reports (likely) benign")

    rationale = "; ".join(rationale_bits) if rationale_bits else "insufficient evidence"

    return ACMGAssessment(
        call=call,
        criteria_triggered=criteria,
        rationale=rationale,
    )
