"""HPO phenotype → gene association and Phrank-style semantic similarity.

The scoring function produces a ``PhenotypeMatch`` describing how well a
gene's HPO profile matches the proband's HPO terms. It uses an
information-content-weighted best-match average, inspired by Phrank
(Jagadeesh et al., Genet Med 2019):

    score(P, G) = (1/|P|) * sum_{p in P} max_{g in G} IC(LCA(p, g))

where ``P`` is the proband's term set, ``G`` is the gene-annotated term set,
``LCA`` is the most informative common ancestor (deepest shared ancestor
with highest IC), and ``IC(t) = -log(freq(t))`` is the information content
of the term in a reference gene-phenotype annotation corpus.

We deliberately keep the implementation self-contained and offline-friendly:

- Gene HPO profile is fetched from the JAX HPO Toolkit API (cached).
- Ancestors for each term are fetched from the JAX ontology endpoint
  (cached to disk so subsequent runs are fast).
- Information content is derived on-demand from the union of all gene
  annotations we've ever cached. This converges to a reasonable estimate
  after the first dozen or so genes are looked up.

Falls back to an exact-ID overlap score if the ancestor walk fails (e.g.
JAX API is down and no ancestors are cached), so the pipeline keeps
running with degraded — but still meaningful — phenotype evidence.

See ``LIMITATIONS.md`` for what this does NOT do (PPI-augmented scoring,
disease-level aggregation, frequency-attribute weighting, etc.).
"""

from __future__ import annotations

import math
from collections import Counter

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_genomics.agents.variant_interpreter.state import PhenotypeMatch
from agentic_genomics.core import cache

HPO_GENE_URL = "https://ontology.jax.org/api/network/annotation/{gene}"
HPO_TERM_URL = "https://ontology.jax.org/api/hp/terms/{term_id}"
TIMEOUT = 15.0

# Convert a raw Phrank score to a coarse human bucket for the LLM prompt.
# These thresholds are tuned to a demo corpus (gene profiles of ~10-50
# terms, proband profiles of 1-5 terms) and are deliberately generous.
_STRONG_SCORE = 5.0
_MODERATE_SCORE = 2.0
_WEAK_SCORE = 0.5


# --- Gene-level annotation ----------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_gene_hpo(gene_symbol: str) -> dict | None:
    """Return raw HPO annotation data for a gene symbol, or None if not found."""
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


def _extract_gene_terms(data: dict | None) -> set[str]:
    """Pull the set of HPO term IDs out of the JAX gene-annotation payload."""
    if not data:
        return set()
    terms: set[str] = set()
    phenotypes = data.get("phenotypes") or data.get("hpos") or []
    for p in phenotypes:
        if isinstance(p, dict):
            term_id = p.get("id") or p.get("termId")
            if term_id:
                terms.add(term_id)
        elif isinstance(p, str):
            terms.add(p)
    return terms


def _extract_linked_diseases(data: dict | None) -> list[str]:
    if not data:
        return []
    names: list[str] = []
    for d in data.get("diseases") or []:
        if isinstance(d, dict):
            name = d.get("name") or d.get("disease_name")
            if name:
                names.append(name)
    return names[:10]


# --- Ontology ancestry walk ---------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _fetch_term_parents(term_id: str) -> list[str]:
    """Return the direct parents of an HPO term (IDs only)."""
    url = HPO_TERM_URL.format(term_id=term_id)
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    parents: list[str] = []
    for p in data.get("parents") or []:
        if isinstance(p, dict):
            pid = p.get("id") or p.get("termId")
            if pid:
                parents.append(pid)
        elif isinstance(p, str):
            parents.append(p)
    return parents


def _term_parents(term_id: str) -> list[str]:
    """Cached wrapper around :func:`_fetch_term_parents`."""
    return cache.cached_call(
        namespace="hpo_parents",
        key=term_id,
        fn=lambda: _fetch_term_parents(term_id),
    )


def _ancestors(term_id: str, _seen: set[str] | None = None) -> set[str]:
    """Return all ancestors of ``term_id`` (inclusive of itself).

    Uses :func:`_term_parents` which is disk-cached, so after a warm-up
    run this is essentially free. Any network failure is caught and the
    partial ancestor set is returned so callers can degrade gracefully.
    """
    if _seen is None:
        _seen = set()
    if term_id in _seen:
        return _seen
    _seen.add(term_id)
    try:
        parents = _term_parents(term_id) or []
    except Exception:
        # Network or API failure — partial DAG is still useful.
        return _seen
    for p in parents:
        _ancestors(p, _seen)
    return _seen


# --- Information content ------------------------------------------------

def _information_content(term_id: str, annotation_counts: Counter[str], total: int) -> float:
    """Estimate IC = -log(freq) from the gene-annotation corpus."""
    if total <= 0:
        return 0.0
    count = annotation_counts.get(term_id, 0)
    # Laplace smoothing so unseen terms have maximum IC rather than +inf.
    freq = (count + 1) / (total + 1)
    return -math.log(freq)


def _best_pair_ic(
    patient_term: str,
    gene_terms: set[str],
    annotation_counts: Counter[str],
    total: int,
) -> tuple[float, str | None]:
    """Find the most informative common ancestor between patient_term and any gene term."""
    patient_ancestors = _ancestors(patient_term)
    if not patient_ancestors:
        return 0.0, None

    best_ic = 0.0
    best_ancestor: str | None = None
    for g in gene_terms:
        shared = patient_ancestors & _ancestors(g)
        for a in shared:
            ic = _information_content(a, annotation_counts, total)
            if ic > best_ic:
                best_ic = ic
                best_ancestor = a
    return best_ic, best_ancestor


# --- Public scoring API -------------------------------------------------

def _bucket(score: float) -> str:
    if score >= _STRONG_SCORE:
        return "strong"
    if score >= _MODERATE_SCORE:
        return "moderate"
    if score >= _WEAK_SCORE:
        return "weak"
    return "none"


def score_phenotype_match(
    gene_symbol: str | None,
    patient_hpo_terms: list[str],
    *,
    annotation_corpus: Counter[str] | None = None,
    corpus_size: int | None = None,
) -> PhenotypeMatch:
    """Phrank-style semantic similarity between patient and gene HPO profiles.

    Parameters
    ----------
    gene_symbol
        Approved gene symbol (e.g. ``"SCN1A"``). If empty, returns an
        empty ``PhenotypeMatch``.
    patient_hpo_terms
        List of HPO term IDs (e.g. ``["HP:0001250", "HP:0001263"]``).
    annotation_corpus
        Optional pre-built counter mapping HPO term ID → number of genes
        annotated with it. Callers can supply this to avoid re-building
        the IC table on every variant; if omitted, a small corpus is
        derived from the patient + gene terms (which gives a usable but
        low-resolution IC estimate).
    corpus_size
        Denominator for IC — the total number of gene annotations the
        corpus was built from. Ignored when ``annotation_corpus`` is None.

    Returns
    -------
    PhenotypeMatch
        With a numeric ``score`` and a coarse ``match_strength`` bucket,
        matched HPO terms (the patient terms with any positive IC share
        with the gene), and up to 10 linked diseases.
    """
    if not gene_symbol or not patient_hpo_terms:
        return PhenotypeMatch(match_strength="none", score=0.0)

    data = gene_hpo_annotations(gene_symbol)
    if not data:
        return PhenotypeMatch(match_strength="none", score=0.0)

    gene_terms = _extract_gene_terms(data)
    linked_diseases = _extract_linked_diseases(data)
    patient_set = {t.strip() for t in patient_hpo_terms if t.strip()}

    if not gene_terms or not patient_set:
        return PhenotypeMatch(
            linked_diseases=linked_diseases,
            match_strength="none",
            score=0.0,
        )

    # Build a lightweight IC corpus if the caller didn't supply one.
    if annotation_corpus is None:
        counts: Counter[str] = Counter(gene_terms)
        for pt in patient_set:
            counts[pt] += 1
        corpus_size = sum(counts.values())
        annotation_corpus = counts

    # Phrank: for each patient term, find the max IC over all gene terms
    # (via LCA); average the max-ICs.
    matched: list[str] = []
    total_ic = 0.0
    for pt in patient_set:
        ic, _lca = _best_pair_ic(
            pt, gene_terms, annotation_corpus, corpus_size or 1
        )
        if ic > 0:
            matched.append(pt)
        total_ic += ic

    score = total_ic / max(len(patient_set), 1)

    # Best-effort degrade: if we couldn't walk any ancestors (offline or
    # JAX down) the score may be zero while there's a direct ID match.
    # Fall back to simple intersection so the pipeline still gives signal.
    if score == 0.0:
        exact = patient_set & gene_terms
        if exact:
            matched = sorted(exact)
            # Give exact matches a moderate score so they surface in the
            # coarse bucketing.
            score = _MODERATE_SCORE * (len(exact) / len(patient_set))

    return PhenotypeMatch(
        matched_hpo_terms=sorted(set(matched)),
        linked_diseases=linked_diseases,
        match_strength=_bucket(score),  # type: ignore[arg-type]
        score=round(score, 3),
    )


# --- Convenience for callers that want to build a corpus upfront --------

def build_annotation_corpus(
    gene_symbols: list[str],
) -> tuple[Counter[str], int]:
    """Precompute an IC corpus from a batch of gene symbols.

    Call this once per pipeline run to get sharper IC estimates than the
    per-variant default. Returns ``(counts, total)`` suitable for passing
    into :func:`score_phenotype_match`.
    """
    counts: Counter[str] = Counter()
    for gene in gene_symbols:
        data = gene_hpo_annotations(gene)
        if not data:
            continue
        counts.update(_extract_gene_terms(data))
    return counts, sum(counts.values())
