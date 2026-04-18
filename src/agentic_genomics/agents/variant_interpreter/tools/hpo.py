"""HPO phenotype → gene association lookups.

Uses the JAX HPO Toolkit API (free, no key required):
    https://ontology.jax.org/api/docs

We resolve gene-to-phenotype overlap with the proband's HPO terms and
assign a qualitative match strength. The LLM later uses this match
alongside ClinVar/frequency/score evidence when ranking variants.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_genomics.agents.variant_interpreter.state import PhenotypeMatch
from agentic_genomics.core import cache

HPO_GENE_URL = "https://ontology.jax.org/api/network/annotation/{gene}"
TIMEOUT = 15.0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_gene_hpo(gene_symbol: str) -> dict | None:
    """Return HPO annotation data for a gene symbol, or None if not found."""
    url = HPO_GENE_URL.format(gene=gene_symbol)
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def gene_hpo_annotations(gene_symbol: str) -> dict | None:
    """Cached wrapper around :func:`_fetch_gene_hpo`."""
    if not gene_symbol:
        return None
    return cache.cached_call(
        namespace="hpo_gene",
        key=gene_symbol.upper(),
        fn=lambda: _fetch_gene_hpo(gene_symbol.upper()),
    )


def score_phenotype_match(
    gene_symbol: str | None, patient_hpo_terms: list[str]
) -> PhenotypeMatch:
    """Return a :class:`PhenotypeMatch` describing HPO overlap.

    Match strength heuristic:
      - **strong**: 2+ patient terms match the gene's HPO profile
      - **moderate**: exactly 1 match
      - **weak**: no direct match but the gene *has* HPO annotations
      - **none**: gene has no HPO annotations
    """
    if not gene_symbol or not patient_hpo_terms:
        return PhenotypeMatch(match_strength="none")

    data = gene_hpo_annotations(gene_symbol)
    if not data:
        return PhenotypeMatch(match_strength="none")

    gene_terms: set[str] = set()
    linked_diseases: list[str] = []

    phenotypes = data.get("phenotypes") or data.get("hpos") or []
    for p in phenotypes:
        if isinstance(p, dict):
            term_id = p.get("id") or p.get("termId")
            if term_id:
                gene_terms.add(term_id)
        elif isinstance(p, str):
            gene_terms.add(p)

    diseases = data.get("diseases") or []
    for d in diseases:
        if isinstance(d, dict):
            name = d.get("name") or d.get("disease_name")
            if name:
                linked_diseases.append(name)

    patient_set = {t.strip() for t in patient_hpo_terms if t.strip()}
    matched = sorted(patient_set & gene_terms)

    if len(matched) >= 2:
        strength = "strong"
    elif len(matched) == 1:
        strength = "moderate"
    elif gene_terms:
        strength = "weak"
    else:
        strength = "none"

    return PhenotypeMatch(
        matched_hpo_terms=matched,
        linked_diseases=linked_diseases[:10],
        match_strength=strength,
    )
