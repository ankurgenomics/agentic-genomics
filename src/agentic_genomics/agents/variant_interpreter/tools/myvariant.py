"""MyVariant.info client — the agent's swiss-army tool for variant facts.

MyVariant.info aggregates gnomAD, ClinVar, CADD, REVEL, SpliceAI, dbNSFP,
VEP consequences, and more into one free REST endpoint. Using it lets the
MVP avoid wiring up five separate APIs and keeps latency low enough for
an interactive demo.

Docs: https://docs.myvariant.info/
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_genomics.agents.variant_interpreter.state import (
    ClinicalEvidence,
    FunctionalScores,
    PopulationFrequency,
    Variant,
)
from agentic_genomics.core import cache

MYVARIANT_URL = "https://myvariant.info/v1/variant/{hgvs}"
ASSEMBLY = "hg38"  # change to "hg19" if your VCF is on GRCh37
TIMEOUT = 15.0


def _hgvs_from_variant(v: Variant) -> str:
    """Build an HGVS-style key that MyVariant.info accepts."""
    chrom = v.chrom if v.chrom.startswith("chr") else f"chr{v.chrom}"
    # SNVs and simple indels both work with this form.
    return f"{chrom}:g.{v.pos}{v.ref}>{v.alt}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_raw(hgvs: str) -> dict[str, Any] | None:
    """Hit MyVariant.info with sensible retries."""
    url = MYVARIANT_URL.format(hgvs=hgvs)
    params = {"assembly": ASSEMBLY}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def fetch_variant_record(variant: Variant) -> dict[str, Any] | None:
    """Return the raw MyVariant.info record, cached on disk."""
    hgvs = _hgvs_from_variant(variant)
    return cache.cached_call(
        namespace="myvariant",
        key=f"{ASSEMBLY}:{hgvs}",
        fn=lambda: _fetch_raw(hgvs),
    )


def extract_population(record: dict[str, Any] | None) -> PopulationFrequency:
    """Pull gnomAD frequencies out of a MyVariant record."""
    if not record:
        return PopulationFrequency()
    gnomad = record.get("gnomad_genome") or record.get("gnomad_exome") or {}
    af = gnomad.get("af", {}) if isinstance(gnomad, dict) else {}
    # MyVariant.info has returned "af" both as a nested dict
    # ({"af": {"af": 0.001, "af_popmax": ...}}) and, for some records, as a
    # flat float ({"af": 0.001}). Treating the flat form as "no data"
    # silently produced gnomad_af=None for a variant that IS common —
    # which then incorrectly triggers PM2_Supporting ("absent from
    # gnomAD") for it downstream in the ACMG classifier.
    if isinstance(af, dict):
        gnomad_af = af.get("af")
        gnomad_af_popmax = af.get("af_popmax")
    elif isinstance(af, (int, float)):
        gnomad_af = float(af)
        gnomad_af_popmax = None
    else:
        gnomad_af = gnomad_af_popmax = None
    return PopulationFrequency(
        gnomad_af=gnomad_af,
        gnomad_af_popmax=gnomad_af_popmax,
        gnomad_hom=gnomad.get("ac", {}).get("ac_hom") if isinstance(gnomad, dict) else None,
    )


def _consensus_pred(value: Any, deleterious: set[str]) -> str | None:
    """Collapse a dbNSFP prediction field to a single call.

    dbNSFP predictions (SIFT, PolyPhen2, ...) are computed per-transcript,
    and for variants overlapping multiple transcripts MyVariant.info
    returns a *list* of per-transcript calls (e.g. ``["T", "T", "D"]``)
    instead of a single string. Passing that list straight into
    ``FunctionalScores`` (a ``str | None`` field) crashes pydantic
    validation for any real-world multi-transcript variant. We
    conservatively surface a deleterious call if any transcript shows
    one, else fall back to the first available call.
    """
    if isinstance(value, list):
        for v in value:
            if v in deleterious:
                return v
        return value[0] if value else None
    return value if isinstance(value, str) else None


def extract_functional(record: dict[str, Any] | None) -> FunctionalScores:
    """Pull CADD / REVEL / SpliceAI / dbNSFP scores + gnomAD constraint."""
    if not record:
        return FunctionalScores()

    cadd = record.get("cadd", {}) or {}
    dbnsfp = record.get("dbnsfp", {}) or {}
    spliceai = dbnsfp.get("spliceai", {}) or {}

    spliceai_max = None
    if isinstance(spliceai, dict):
        raw_ds_values = [
            spliceai.get("ds_ag"),
            spliceai.get("ds_al"),
            spliceai.get("ds_dg"),
            spliceai.get("ds_dl"),
        ]
        ds_values: list[float] = [v for v in raw_ds_values if isinstance(v, (int, float))]
        spliceai_max = max(ds_values) if ds_values else None

    revel = dbnsfp.get("revel", {})
    revel_score = revel.get("score") if isinstance(revel, dict) else revel

    sift_pred = _consensus_pred(
        dbnsfp.get("sift", {}).get("pred") if isinstance(dbnsfp.get("sift"), dict) else None,
        deleterious={"D"},
    )
    polyphen_field = dbnsfp.get("polyphen2", {})
    polyphen_hdiv = polyphen_field.get("hdiv", {}) if isinstance(polyphen_field, dict) else {}
    polyphen_pred = _consensus_pred(
        polyphen_hdiv.get("pred") if isinstance(polyphen_hdiv, dict) else None,
        deleterious={"D", "P"},
    )

    # gnomAD constraint metrics. MyVariant aggregates these under
    # "gnomad_genome" / "gnomad_exome" > "constraint" or a top-level
    # "gnomad_constraint" field depending on release; we look in the
    # usual places and fall back gracefully.
    pli = loeuf = None
    for container in (
        record.get("gnomad_constraint"),
        (record.get("gnomad_exome") or {}).get("constraint")
        if isinstance(record.get("gnomad_exome"), dict) else None,
        (record.get("gnomad_genome") or {}).get("constraint")
        if isinstance(record.get("gnomad_genome"), dict) else None,
    ):
        if not isinstance(container, dict):
            continue
        if pli is None:
            candidate = container.get("pli") or container.get("pLI")
            if isinstance(candidate, (int, float)):
                pli = float(candidate)
        if loeuf is None:
            candidate = container.get("loeuf") or container.get("oe_lof_upper")
            if isinstance(candidate, (int, float)):
                loeuf = float(candidate)

    return FunctionalScores(
        cadd_phred=cadd.get("phred") if isinstance(cadd, dict) else None,
        revel=revel_score if isinstance(revel_score, (int, float)) else None,
        spliceai_max=spliceai_max,
        sift=sift_pred,
        polyphen=polyphen_pred,
        gnomad_pli=pli,
        gnomad_loeuf=loeuf,
    )


def extract_clinvar(record: dict[str, Any] | None) -> ClinicalEvidence:
    """Pull ClinVar clinical-significance evidence."""
    if not record:
        return ClinicalEvidence()
    clinvar = record.get("clinvar")
    if not clinvar:
        return ClinicalEvidence()

    rcv = clinvar.get("rcv") if isinstance(clinvar, dict) else None
    if isinstance(rcv, list) and rcv:
        significances = {
            r.get("clinical_significance") for r in rcv if isinstance(r, dict)
        }
        significances.discard(None)
        conflicts = len(significances) > 1
        review_status = rcv[0].get("review_status") if isinstance(rcv[0], dict) else None
        return ClinicalEvidence(
            clinvar_significance=", ".join(sorted(s for s in significances if s)) or None,
            clinvar_review_status=review_status,
            clinvar_submitters=len(rcv),
            clinvar_conflicts=conflicts,
        )
    if isinstance(rcv, dict):
        return ClinicalEvidence(
            clinvar_significance=rcv.get("clinical_significance"),
            clinvar_review_status=rcv.get("review_status"),
            clinvar_submitters=1,
            clinvar_conflicts=False,
        )
    return ClinicalEvidence()
