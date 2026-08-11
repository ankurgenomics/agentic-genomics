"""gnomAD gene-constraint lookup (pLI / LOEUF) — fills a gap MyVariant.info leaves empty.

``myvariant.extract_functional`` looks for gnomAD constraint metrics under
``gnomad_constraint`` / ``gnomad_exome.constraint`` / ``gnomad_genome.constraint``
in the MyVariant.info record. As of this writing **none of those paths are
populated for any record in MyVariant.info's index** (`_exists_:gnomad_constraint`
etc. all return zero hits) — meaning PVS1, which gates on exactly these fields via
``acmg_lite._is_lof_intolerant_gene``, can never fire against live MyVariant.info
data, for any gene, ever. Since PVS1 is the *only* combining-rule path to a
Pathogenic/Likely-Pathogenic call in this v1 criteria set, that silently caps the
whole classifier at Uncertain Significance for every missense variant, no matter
how strong the ClinVar consensus.

This module queries gnomAD's own public GraphQL API directly for gene-level
constraint (the source MyVariant.info's field was presumably meant to mirror),
gated per gene rather than per variant so it's cheap and disk-cached.

Known limitation carried over from the ACMG-lite gene list, not fixed here: pLI /
LOEUF are population-constraint metrics tuned for early-onset, high-penetrance
disease genes. They are well-documented as unreliable for adult-onset cancer
predisposition genes (BRCA1/2, CHEK2, PALB2, ...), which show much weaker
constraint signal despite being clinically LoF-intolerant. Don't be surprised if
PVS1 doesn't fire for those genes even with this fix in place — that's gnomAD's
constraint metric being the wrong tool for that gene class, not a bug here.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_genomics.core import cache

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
TIMEOUT = 15.0

_QUERY = """
{
  gene(gene_symbol: "%s", reference_genome: GRCh38) {
    gnomad_constraint { pli oe_lof_upper }
  }
}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_constraint(gene_symbol: str) -> dict | None:
    """Return ``{"pli": float, "loeuf": float}`` for a gene, or None if unavailable."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            GNOMAD_API_URL,
            json={"query": _QUERY % gene_symbol},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()

    gene = (payload.get("data") or {}).get("gene")
    if not gene:
        return None
    constraint = gene.get("gnomad_constraint")
    if not constraint:
        return None
    pli = constraint.get("pli")
    loeuf = constraint.get("oe_lof_upper")
    if pli is None and loeuf is None:
        return None
    return {"pli": pli, "loeuf": loeuf}


def gene_constraint(gene_symbol: str | None) -> dict | None:
    """Cached wrapper around :func:`_fetch_constraint`.

    Degrades to ``None`` on any failure — a transient gnomAD API outage should
    cost this gene's PVS1 gating evidence, not crash the pipeline (same
    reasoning as ``hpo.gene_hpo_annotations``).
    """
    if not gene_symbol:
        return None
    try:
        return cache.cached_call(
            namespace="gnomad_constraint",
            key=gene_symbol.upper(),
            fn=lambda: _fetch_constraint(gene_symbol.upper()),
        )
    except Exception:
        return None
