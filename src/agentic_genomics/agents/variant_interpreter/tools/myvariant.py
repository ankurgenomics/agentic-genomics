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
    return PopulationFrequency(
        gnomad_af=af.get("af") if isinstance(af, dict) else None,
        gnomad_af_popmax=af.get("af_popmax") if isinstance(af, dict) else None,
        gnomad_hom=gnomad.get("ac", {}).get("ac_hom") if isinstance(gnomad, dict) else None,
    )


def extract_functional(record: dict[str, Any] | None) -> FunctionalScores:
    """Pull CADD / REVEL / SpliceAI / dbNSFP scores."""
    if not record:
        return FunctionalScores()

    cadd = record.get("cadd", {}) or {}
    dbnsfp = record.get("dbnsfp", {}) or {}
    spliceai = dbnsfp.get("spliceai", {}) or {}

    spliceai_max = None
    if isinstance(spliceai, dict):
        ds_values = [
            spliceai.get("ds_ag"),
            spliceai.get("ds_al"),
            spliceai.get("ds_dg"),
            spliceai.get("ds_dl"),
        ]
        ds_values = [v for v in ds_values if isinstance(v, (int, float))]
        spliceai_max = max(ds_values) if ds_values else None

    revel = dbnsfp.get("revel", {})
    revel_score = revel.get("score") if isinstance(revel, dict) else revel

    return FunctionalScores(
        cadd_phred=cadd.get("phred") if isinstance(cadd, dict) else None,
        revel=revel_score if isinstance(revel_score, (int, float)) else None,
        spliceai_max=spliceai_max,
        sift=dbnsfp.get("sift", {}).get("pred") if isinstance(dbnsfp.get("sift"), dict) else None,
        polyphen=dbnsfp.get("polyphen2", {}).get("hdiv", {}).get("pred")
        if isinstance(dbnsfp.get("polyphen2"), dict)
        else None,
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
