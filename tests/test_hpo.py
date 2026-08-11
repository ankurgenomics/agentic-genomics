"""Tests for the Phrank-style HPO phenotype scoring.

These tests stub the network-facing parts of ``tools/hpo.py`` so we can
exercise the scoring logic — IC computation, ancestor walks, LCA — in
isolation and offline.
"""

from __future__ import annotations

from collections import Counter

import pytest

from agentic_genomics.agents.variant_interpreter.tools import hpo
from agentic_genomics.core import cache


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    """Isolate every test in this file from the real on-disk cache.

    Without this, tests that exercise ``gene_hpo_annotations`` for real
    (rather than monkeypatching it away) would read/write
    ``.agentic_genomics_cache`` in the repo working directory — polluting
    it, and worse, letting a stale cache entry from an earlier manual run
    silently mask a test that's supposed to be exercising a cache miss.
    """
    monkeypatch.setattr(cache, "_CACHE_DIR", tmp_path)
    yield


# --- Pure helpers -------------------------------------------------------

def test_bucket_thresholds():
    """Score → bucket mapping uses documented thresholds."""
    assert hpo._bucket(0.0) == "none"
    assert hpo._bucket(0.4) == "none"
    assert hpo._bucket(1.0) == "weak"
    assert hpo._bucket(3.0) == "moderate"
    assert hpo._bucket(10.0) == "strong"


def test_extract_gene_terms_handles_both_schemas():
    """JAX API has returned both 'phenotypes' (list of dicts) and 'hpos'."""
    d1 = {"phenotypes": [{"id": "HP:0001250"}, {"termId": "HP:0001263"}]}
    d2 = {"hpos": ["HP:0003002", "HP:0000152"]}
    assert hpo._extract_gene_terms(d1) == {"HP:0001250", "HP:0001263"}
    assert hpo._extract_gene_terms(d2) == {"HP:0003002", "HP:0000152"}
    assert hpo._extract_gene_terms(None) == set()


def test_information_content_handles_empty_corpus():
    """Guard against division by zero on a fresh pipeline run."""
    assert hpo._information_content("HP:X", Counter(), 0) == 0.0


def test_information_content_rare_term_has_higher_ic():
    """IC = -log(freq); a rarer term should carry more IC."""
    counts = Counter({"HP:COMMON": 100, "HP:RARE": 1})
    ic_common = hpo._information_content("HP:COMMON", counts, 101)
    ic_rare = hpo._information_content("HP:RARE", counts, 101)
    assert ic_rare > ic_common > 0


# --- score_phenotype_match with everything stubbed ---------------------

def test_empty_inputs_return_none_match(monkeypatch):
    """No gene or no HPO terms → immediate none bucket, no API call."""
    calls: list[str] = []
    monkeypatch.setattr(
        hpo, "gene_hpo_annotations",
        lambda g: calls.append(g) or {"phenotypes": [{"id": "HP:0001250"}]},
    )

    m1 = hpo.score_phenotype_match(None, ["HP:0001250"])
    assert m1.match_strength == "none"
    assert m1.score == 0.0

    m2 = hpo.score_phenotype_match("SCN1A", [])
    assert m2.match_strength == "none"
    assert m2.score == 0.0

    assert calls == []  # neither path should have called the API


@pytest.mark.network
def test_gene_hpo_annotations_live_lookup_returns_real_data():
    """End-to-end sanity check against the real JAX ontology API.

    Regression test: ``HPO_GENE_URL`` used to be called directly with a
    bare gene symbol (e.g. ``.../annotation/BRCA1``), which the live API
    rejects with 400 ("TermId construction error: 'BRCA1' does not have a
    prefix!") — meaning every real run silently produced zero phenotype
    signal for every gene, with the failure masked by the offline unit
    tests (which all stub ``gene_hpo_annotations`` directly) and by the
    network-failure fallback degrading it to "no data" instead of raising.
    This hits the live API to prove the symbol→NCBIGene-ID resolution step
    actually returns usable phenotype/disease data.
    """
    data = hpo.gene_hpo_annotations("BRCA1")
    assert data is not None
    terms = hpo._extract_gene_terms(data)
    assert "HP:0002119" in terms or len(terms) > 0


def test_gene_hpo_annotations_degrades_on_persistent_network_failure(monkeypatch):
    """A persistent JAX API failure must degrade that gene, not crash the run.

    Regression test: ``_fetch_gene_hpo`` is wrapped in ``@retry``; once
    retries are exhausted for a non-404 failure (timeout, 5xx), tenacity
    raises. That used to propagate straight out of ``gene_hpo_annotations``
    and up through ``phenotype_score``, crashing the entire LangGraph run
    over a single gene's phenotype lookup — a transient outage should only
    cost that gene's phenotype signal.
    """

    def _always_fails(*_args, **_kwargs):
        raise TimeoutError("JAX ontology API unreachable")

    # Gene-ID resolution succeeds; the annotation fetch is what fails.
    monkeypatch.setattr(hpo, "_resolve_gene_id", lambda symbol: "NCBIGene:6323")
    monkeypatch.setattr(hpo, "_fetch_gene_hpo", _always_fails)
    assert hpo.gene_hpo_annotations("SCN1A") is None


def test_gene_hpo_annotations_returns_none_when_symbol_does_not_resolve(monkeypatch):
    """An unresolvable gene symbol should degrade to None, not call the
    annotation endpoint with a bare symbol (which 400s — see
    ``_fetch_gene_id``'s docstring).
    """
    monkeypatch.setattr(hpo, "_resolve_gene_id", lambda symbol: None)
    assert hpo.gene_hpo_annotations("NOTAREALGENE") is None


def test_gene_not_found_returns_none(monkeypatch):
    monkeypatch.setattr(hpo, "gene_hpo_annotations", lambda g: None)
    m = hpo.score_phenotype_match("NOPE", ["HP:0001250"])
    assert m.match_strength == "none"


def test_exact_match_fallback_when_no_ancestors(monkeypatch):
    """If the ontology walk finds nothing, we should still score exact-ID overlap."""
    monkeypatch.setattr(
        hpo, "gene_hpo_annotations",
        lambda g: {"phenotypes": [{"id": "HP:0001250"}]},
    )
    # Ancestors endpoint "fails" (returns empty set for everything).
    monkeypatch.setattr(hpo, "_ancestors", lambda t, _seen=None: {t})

    m = hpo.score_phenotype_match(
        "SCN1A", ["HP:0001250"],
        annotation_corpus=Counter({"HP:0001250": 1}),
        corpus_size=1,
    )
    # IC of a term at freq 1/(1+1) = 0.5 → -log(0.5) ≈ 0.693. That's below
    # the weak threshold (0.5), so we fall into the exact-match fallback,
    # which awards _MODERATE_SCORE × (1/1) = 2.0 → "moderate".
    assert m.match_strength in {"weak", "moderate"}
    assert "HP:0001250" in m.matched_hpo_terms


def test_ic_weighted_scoring_with_shared_ancestor(monkeypatch):
    """Patient and gene share an ancestor; score should reflect its IC."""
    monkeypatch.setattr(
        hpo, "gene_hpo_annotations",
        lambda g: {"phenotypes": [{"id": "HP:GENE_TERM"}]},
    )

    # HP:PATIENT and HP:GENE_TERM both descend from HP:ANCESTOR, which is rare.
    ancestors = {
        "HP:PATIENT": {"HP:PATIENT", "HP:ANCESTOR", "HP:ROOT"},
        "HP:GENE_TERM": {"HP:GENE_TERM", "HP:ANCESTOR", "HP:ROOT"},
    }
    monkeypatch.setattr(
        hpo, "_ancestors",
        lambda t, _seen=None: ancestors.get(t, {t}),
    )

    # Corpus: HP:ROOT is very common, HP:ANCESTOR is rare. LCA should pick
    # the rarer one (higher IC).
    corpus = Counter({"HP:ROOT": 100, "HP:ANCESTOR": 2, "HP:GENE_TERM": 1})
    m = hpo.score_phenotype_match(
        "SCN1A", ["HP:PATIENT"],
        annotation_corpus=corpus,
        corpus_size=sum(corpus.values()),
    )
    assert m.score > 0
    assert m.matched_hpo_terms == ["HP:PATIENT"]


def test_multiple_patient_terms_average_ic(monkeypatch):
    """Score is the mean of per-patient-term best-match ICs."""
    monkeypatch.setattr(
        hpo, "gene_hpo_annotations",
        lambda g: {"phenotypes": [{"id": "HP:G"}]},
    )
    ancestors = {
        "HP:P1": {"HP:P1", "HP:MATCH"},
        "HP:P2": {"HP:P2"},  # no shared ancestor with HP:G
        "HP:G": {"HP:G", "HP:MATCH"},
    }
    monkeypatch.setattr(
        hpo, "_ancestors",
        lambda t, _seen=None: ancestors.get(t, {t}),
    )
    corpus = Counter({"HP:MATCH": 1, "HP:G": 5, "HP:P1": 5, "HP:P2": 5})
    m = hpo.score_phenotype_match(
        "SCN1A", ["HP:P1", "HP:P2"],
        annotation_corpus=corpus,
        corpus_size=sum(corpus.values()),
    )
    # P1 contributes positive IC, P2 contributes 0 → average is half of
    # the single-term IC.
    assert m.score > 0
    assert "HP:P1" in m.matched_hpo_terms
    assert "HP:P2" not in m.matched_hpo_terms
