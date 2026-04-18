"""Deterministic ACMG-lite classifier.

.. warning::

    This is **not** a certified ACMG/AMP 2015 implementation. It is a
    pragmatic, *transparent* rule engine covering a subset of well-defined
    criteria — meant for demonstration of the agentic-AI pattern, not for
    clinical use. See ``LIMITATIONS.md`` for what is NOT implemented.

Criteria currently implemented (9 of 28)
----------------------------------------

Pathogenic evidence:
- ``PVS1``               : null variant (stop-gained, frameshift, canonical
                            splice) in a LoF-intolerant gene (gnomAD pLI ≥ 0.9
                            or LOEUF ≤ 0.35 when available).
- ``PM2_Supporting``     : absent / ultra-rare in gnomAD (popmax AF < 1e-4).
- ``PP3``                : multiple in-silico predictors support deleterious.
- ``PP5``                : ClinVar reports (likely) pathogenic, no conflicts.

Benign evidence:
- ``BA1``                : allele frequency ≥ 5% in gnomAD popmax.
- ``BS1``                : allele frequency ≥ 1% in gnomAD popmax.
- ``BP4``                : multiple in-silico predictors support benign.
- ``BP6``                : ClinVar reports (likely) benign, no conflicts.

Combining rules
---------------

Uses the **Richards et al. 2015** (Genet Med, Table 5) combining rules
mapped onto the subset of criteria we actually trigger:

  Pathogenic   = (PVS1 + ≥1 Strong)
               | (PVS1 + ≥2 Moderate)
               | (PVS1 + ≥1 Moderate + ≥1 Supporting)
               | (PVS1 + ≥2 Supporting)
               | (≥2 Strong)
               | (≥1 Strong + ≥3 Moderate)
               | (≥1 Strong + ≥2 Moderate + ≥2 Supporting)
               | (≥1 Strong + ≥1 Moderate + ≥4 Supporting)
  Likely Pathogenic =
                 (PVS1 + ≥1 Moderate)
               | (≥1 Strong + ≥1 Moderate)
               | (≥1 Strong + ≥2 Supporting)
               | (≥3 Moderate)
               | (≥2 Moderate + ≥2 Supporting)
               | (≥1 Moderate + ≥4 Supporting)
  Benign        = ``BA1`` | (≥2 Strong benign)
  Likely Benign = (≥1 Strong + ≥1 Supporting benign)
               | (≥2 Supporting benign)

Evidence weights used here (per Richards 2015, with PP5/BP6 marked Supporting
per the 2015 paper — note ClinGen SVI *deprecated* PP5/BP6 in 2018 in favour
of promoting ClinVar entries based on review status; we do not implement that
refinement, see ``LIMITATIONS.md``):

  PVS1            -> Very Strong (pathogenic)
  PS*             -> Strong      (pathogenic)   [none implemented]
  PM2, PM*        -> Moderate    (pathogenic)   [only PM2_Supporting used]
  PP3, PP5        -> Supporting  (pathogenic)
  BA1             -> Stand-alone (benign)
  BS1             -> Strong      (benign)
  BP4, BP6        -> Supporting  (benign)

The output is a transparent list of triggered criteria + a final call +
a narrative rationale, all of which the LLM synthesiser later reflects
into the human-readable report.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.state import (
    ACMGAssessment,
    AnnotatedVariant,
)

_PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic"}
_BENIGN_TERMS = {"benign", "likely benign"}

# Consequence strings that represent predicted loss-of-function.
# Follows Ensembl VEP terminology. We intentionally do NOT include
# "start_lost" or "stop_lost" in the Very-Strong bucket — those have
# their own (not-yet-implemented) rules (PM4 / PS1 variants).
_NULL_CONSEQUENCES = {
    "stop_gained",
    "frameshift_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    # pyVCF / VEP composite forms — any CSQ containing one of the above.
}

# Gating thresholds for "the gene tolerates loss-of-function poorly",
# i.e. haploinsufficiency is a plausible disease mechanism.
_PLI_LOF_INTOLERANT = 0.9
_LOEUF_LOF_INTOLERANT = 0.35

# Classifier disclaimer surfaced in every rationale.
_CAVEAT = (
    "[acmg-lite: transparent subset classifier — not ACMG/AMP 2015 certified]"
)


def _significance_set(sig: str | None) -> set[str]:
    if not sig:
        return set()
    return {s.strip().lower() for s in sig.split(",") if s.strip()}


def _is_null_consequence(consequence: str | None) -> bool:
    """True if the VEP consequence string implies loss-of-function."""
    if not consequence:
        return False
    tokens = {t.strip() for t in consequence.replace(",", "&").split("&")}
    return bool(tokens & _NULL_CONSEQUENCES)


def _is_lof_intolerant_gene(v: AnnotatedVariant) -> bool:
    """Apply gnomAD pLI / LOEUF gate for PVS1 eligibility.

    Returns True if there's evidence the gene is haploinsufficient. We
    require *some* constraint evidence — a gene with no constraint data
    is NOT automatically assumed to be LoF-intolerant.
    """
    pli = v.functional.gnomad_pli
    loeuf = v.functional.gnomad_loeuf
    if pli is not None and pli >= _PLI_LOF_INTOLERANT:
        return True
    return loeuf is not None and loeuf <= _LOEUF_LOF_INTOLERANT


def _evaluate_pvs1(v: AnnotatedVariant) -> list[str]:
    """PVS1: null variant in a LoF-intolerant gene.

    This is a deliberately conservative approximation of the ClinGen SVI
    PVS1 decision tree (Abou Tayoun et al., Hum Mutat 2018):

    - We check *null consequence* (stop-gained / frameshift / canonical splice).
    - We check *LoF intolerance* of the gene via gnomAD pLI or LOEUF.

    We do NOT implement the full SVI refinements:
    - NMD-escape rules (last exon / within 50 nt of last exon junction)
    - single-exon-gene guard
    - biologically-relevant transcript check
    - alternative splicing rescue

    So this PVS1 trigger should be read as "candidate PVS1; requires manual
    SVI flowchart confirmation."
    """
    if not _is_null_consequence(v.variant.consequence):
        return []
    if not _is_lof_intolerant_gene(v):
        return []
    return ["PVS1"]


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


# --- Evidence-weight bookkeeping -----------------------------------------
# Each criterion code maps to its evidence weight. This is the 2015
# taxonomy; the (deprecated-by-ClinGen-2018) PP5/BP6 are kept Supporting.

_WEIGHT: dict[str, tuple[str, str]] = {
    # (direction, strength)  direction in {"P","B"}  strength in {"VS","S","M","SUP","STANDALONE"}
    "PVS1": ("P", "VS"),
    "PM2_Supporting": ("P", "SUP"),  # we only ever trigger the Supporting form
    "PP3": ("P", "SUP"),
    "PP5": ("P", "SUP"),
    "BA1": ("B", "STANDALONE"),
    "BS1": ("B", "S"),
    "BP4": ("B", "SUP"),
    "BP6": ("B", "SUP"),
}


def _tally(criteria: list[str]) -> dict[str, int]:
    """Count triggered criteria by (direction, strength) bucket."""
    buckets = {
        "P_VS": 0, "P_S": 0, "P_M": 0, "P_SUP": 0,
        "B_STANDALONE": 0, "B_S": 0, "B_M": 0, "B_SUP": 0,
    }
    for c in criteria:
        weight = _WEIGHT.get(c)
        if not weight:
            continue
        direction, strength = weight
        buckets[f"{direction}_{strength}"] += 1
    return buckets


def _final_call(criteria: list[str]) -> str:
    """Apply Richards et al. 2015 Table 5 combining rules."""
    b = _tally(criteria)
    vs, s, m, sup = b["P_VS"], b["P_S"], b["P_M"], b["P_SUP"]
    b_sa, b_s, b_sup = b["B_STANDALONE"], b["B_S"], b["B_SUP"]

    # Benign first — BA1 trumps everything.
    if b_sa >= 1:
        return "Benign"
    # Two strong benign criteria → Benign.
    if b_s >= 2:
        return "Benign"
    # One strong + one supporting, or two supporting → Likely Benign eligible.
    likely_benign_eligible = (b_s >= 1 and b_sup >= 1) or b_sup >= 2

    # Pathogenic rules (per Richards 2015, Table 5, "Pathogenic" list).
    pathogenic_rules = [
        vs >= 1 and s >= 1,                       # PVS1 + 1 Strong
        vs >= 1 and m >= 2,                       # PVS1 + 2 Moderate
        vs >= 1 and m >= 1 and sup >= 1,          # PVS1 + 1 Moderate + 1 Supporting
        vs >= 1 and sup >= 2,                     # PVS1 + 2 Supporting
        s >= 2,                                   # 2 Strong
        s >= 1 and m >= 3,                        # 1 Strong + 3 Moderate
        s >= 1 and m >= 2 and sup >= 2,           # 1 Strong + 2 Moderate + 2 Supporting
        s >= 1 and m >= 1 and sup >= 4,           # 1 Strong + 1 Moderate + 4 Supporting
    ]
    # Only return Pathogenic if there are no conflicting benign criteria;
    # otherwise fall through to LP/VUS consideration.
    if any(pathogenic_rules) and not (b_s or b_sup):
        return "Pathogenic"

    # Likely Pathogenic rules (per Richards 2015, Table 5, "Likely Pathogenic" list).
    likely_pathogenic_rules = [
        vs >= 1 and m >= 1,                       # PVS1 + 1 Moderate
        s >= 1 and m >= 1,                        # 1 Strong + 1 Moderate
        s >= 1 and sup >= 2,                      # 1 Strong + 2 Supporting
        m >= 3,                                   # 3 Moderate
        m >= 2 and sup >= 2,                      # 2 Moderate + 2 Supporting
        m >= 1 and sup >= 4,                      # 1 Moderate + 4 Supporting
        # "PVS1 alone" is NOT enough in Richards 2015 but IS enough under the
        # ClinGen SVI 2018 PVS1 refinement (Abou Tayoun 2018). We surface this
        # as LP to be useful on real null variants; LIMITATIONS.md documents
        # that the full SVI flowchart isn't implemented.
        vs >= 1,
    ]
    if any(likely_pathogenic_rules) and not (b_s or b_sup):
        return "Likely Pathogenic"

    if likely_benign_eligible:
        return "Likely Benign"
    return "Uncertain Significance"


# --- Public API ----------------------------------------------------------

def classify(v: AnnotatedVariant) -> ACMGAssessment:
    """Run the deterministic ACMG-lite classifier on a single variant."""
    criteria: list[str] = []
    criteria.extend(_evaluate_pvs1(v))
    criteria.extend(_evaluate_frequency(v))
    criteria.extend(_evaluate_insilico(v))
    criteria.extend(_evaluate_clinvar(v))

    call = _final_call(criteria)

    rationale_bits: list[str] = []
    if "PVS1" in criteria:
        rationale_bits.append(
            "null consequence in a LoF-intolerant gene (candidate PVS1; "
            "SVI flowchart not fully applied)"
        )
    if "PM2_Supporting" in criteria:
        rationale_bits.append("absent/ultra-rare in gnomAD")
    if "BA1" in criteria:
        rationale_bits.append("common in gnomAD (popmax AF ≥ 5%)")
    if "BS1" in criteria:
        rationale_bits.append("relatively common in gnomAD (popmax AF ≥ 1%)")
    if "PP3" in criteria:
        rationale_bits.append("multiple in-silico predictors agree it is deleterious")
    if "BP4" in criteria:
        rationale_bits.append("multiple in-silico predictors agree it is benign")
    if "PP5" in criteria:
        rationale_bits.append("ClinVar reports (likely) pathogenic")
    if "BP6" in criteria:
        rationale_bits.append("ClinVar reports (likely) benign")

    body = "; ".join(rationale_bits) if rationale_bits else "insufficient evidence"
    rationale = f"{body}. {_CAVEAT}"

    return ACMGAssessment(
        call=call,
        criteria_triggered=criteria,
        rationale=rationale,
    )
