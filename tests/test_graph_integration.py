"""End-to-end integration test for the LangGraph variant interpreter.

This test exercises the full graph (ingest -> annotate -> frequency_filter ->
phenotype_match -> acmg_classify -> synthesize_report) using:

  - the real demo VCF shipped in ``data/samples/proband_demo.vcf``
  - stubbed MyVariant.info responses (no network)
  - a fake LLM that returns canned markdown (no API key required)

Its purpose is to catch regressions in graph wiring, state flow between nodes,
and schema drift between tools. It runs offline and in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_genomics.agents.variant_interpreter import graph as graph_mod
from agentic_genomics.agents.variant_interpreter import nodes as nodes_mod
from agentic_genomics.agents.variant_interpreter.state import (
    ClinicalEvidence,
    FunctionalScores,
    PopulationFrequency,
    Variant,
)

DEMO_VCF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "samples"
    / "proband_demo.vcf"
)


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Drop-in replacement for a LangChain chat model.

    Records the prompt it was called with so the test can assert the LLM
    was actually invoked with the serialised variant evidence, then returns
    a deterministic markdown report.
    """

    def __init__(self) -> None:
        self.last_messages: list[Any] | None = None
        self.call_count = 0

    def invoke(self, messages: list[Any]) -> _FakeLLMResponse:
        self.last_messages = messages
        self.call_count += 1
        return _FakeLLMResponse(
            content=(
                "# Variant interpretation report\n\n"
                "## Top candidates\n"
                "1. **BRCA1** — likely pathogenic; strong phenotype match.\n"
                "2. **BRCA2** — uncertain; moderate evidence.\n"
            )
        )


# Synthetic per-gene responses. Keys correspond to the 7 demo-VCF variants.
# Deliberately include a common variant (LDLR, AF=5%) to verify the frequency
# filter actually drops something.
_FAKE_RECORDS: dict[str, dict[str, Any]] = {
    "BRCA1": {
        "population": PopulationFrequency(gnomad_af=0.0001, gnomad_af_popmax=0.0002),
        "functional": FunctionalScores(cadd_phred=28.0, revel=0.82, spliceai_max=0.1),
        "clinical": ClinicalEvidence(clinvar_significance="Likely pathogenic"),
    },
    "BRCA2": {
        "population": PopulationFrequency(gnomad_af=None, gnomad_af_popmax=None),
        "functional": FunctionalScores(cadd_phred=22.0, revel=0.55),
        "clinical": ClinicalEvidence(),
    },
    "CFTR": {
        "population": PopulationFrequency(gnomad_af=0.005, gnomad_af_popmax=0.008),
        "functional": FunctionalScores(cadd_phred=26.0, revel=0.7),
        "clinical": ClinicalEvidence(clinvar_significance="Pathogenic"),
    },
    "LDLR": {
        # Common: should be dropped by frequency_filter (popmax > 1%).
        "population": PopulationFrequency(gnomad_af=0.04, gnomad_af_popmax=0.05),
        "functional": FunctionalScores(cadd_phred=5.0, revel=0.1),
        "clinical": ClinicalEvidence(),
    },
    "SCN1A": {
        "population": PopulationFrequency(gnomad_af=0.0001),
        "functional": FunctionalScores(cadd_phred=30.0, revel=0.9, spliceai_max=0.8),
        "clinical": ClinicalEvidence(clinvar_significance="Likely pathogenic"),
    },
    "TP53": {
        "population": PopulationFrequency(gnomad_af=0.0002),
        "functional": FunctionalScores(cadd_phred=27.0, revel=0.75),
        "clinical": ClinicalEvidence(),
    },
    "DMD": {
        "population": PopulationFrequency(gnomad_af=0.002),
        "functional": FunctionalScores(cadd_phred=8.0, revel=0.2),
        "clinical": ClinicalEvidence(),
    },
}


def _fake_fetch_variant_record(variant: Variant) -> dict:
    return {"_gene": variant.gene}


def _fake_extract_population(record: dict) -> PopulationFrequency:
    return _FAKE_RECORDS.get(
        record.get("_gene"), {}
    ).get("population", PopulationFrequency())


def _fake_extract_functional(record: dict) -> FunctionalScores:
    return _FAKE_RECORDS.get(
        record.get("_gene"), {}
    ).get("functional", FunctionalScores())


def _fake_extract_clinvar(record: dict) -> ClinicalEvidence:
    return _FAKE_RECORDS.get(
        record.get("_gene"), {}
    ).get("clinical", ClinicalEvidence())


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch out all network-touching bits so the graph runs offline."""
    monkeypatch.setattr(
        nodes_mod.myvariant, "fetch_variant_record", _fake_fetch_variant_record
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_population", _fake_extract_population
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_functional", _fake_extract_functional
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_clinvar", _fake_extract_clinvar
    )

    # HPO: return a deterministic moderate match for BRCA1/BRCA2, none otherwise.
    def _fake_score_phenotype_match(gene, hpo_terms):
        from agentic_genomics.agents.variant_interpreter.state import PhenotypeMatch

        if gene in {"BRCA1", "BRCA2"} and hpo_terms:
            return PhenotypeMatch(
                match_strength="strong",
                matched_hpo_terms=list(hpo_terms),
                linked_diseases=["Hereditary Breast and Ovarian Cancer"],
            )
        return PhenotypeMatch(match_strength="none")

    monkeypatch.setattr(
        nodes_mod.hpo, "score_phenotype_match", _fake_score_phenotype_match
    )

    fake_llm = _FakeLLM()
    monkeypatch.setattr(nodes_mod, "get_llm", lambda: fake_llm)
    return fake_llm


def test_full_graph_on_demo_vcf(patched_pipeline):
    """Run the full LangGraph on the real demo VCF with stubbed IO."""
    result = graph_mod.run_variant_interpreter(
        vcf_path=str(DEMO_VCF),
        hpo_terms=["HP:0003002"],  # breast carcinoma
        max_variants=50,
    )

    # 7 variants in the demo VCF; 1 (LDLR) is common and should be filtered out.
    assert len(result.variants) == 6, (
        f"expected 6 post-filter variants, got {len(result.variants)}"
    )
    kept_genes = {av.variant.gene for av in result.variants}
    assert "LDLR" not in kept_genes, "common LDLR variant should have been filtered"
    assert {"BRCA1", "BRCA2", "CFTR", "SCN1A", "TP53", "DMD"} == kept_genes

    # Every kept variant must have all downstream evidence populated.
    for av in result.variants:
        assert av.acmg is not None, f"ACMG missing for {av.variant.gene}"
        assert av.phenotype is not None
        # BRCA1/BRCA2 matched our phenotype stub; others didn't.
        if av.variant.gene in {"BRCA1", "BRCA2"}:
            assert av.phenotype.match_strength == "strong"

    # Report is non-empty markdown with the safety framing.
    assert result.report_markdown
    assert "Variant interpretation report" in result.report_markdown

    # The LLM was called exactly once and the prompt carried variant evidence.
    fake_llm = patched_pipeline
    assert fake_llm.call_count == 1
    human_msg = fake_llm.last_messages[-1].content
    assert "BRCA1" in human_msg
    assert "HP:0003002" in human_msg

    # Reasoning trace records all 6 node steps.
    trace_steps = [s["node"] for s in result.reasoning_trace]
    assert trace_steps == [
        "ingest_variants",
        "annotate_evidence",
        "frequency_filter",
        "phenotype_match",
        "acmg_classify",
        "synthesize_report",
    ]


def test_empty_vcf_short_circuits_llm(tmp_path, patched_pipeline):
    """If frequency filter drops every variant, the LLM should not be called."""
    empty_vcf = tmp_path / "empty.vcf"
    empty_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    result = graph_mod.run_variant_interpreter(
        vcf_path=str(empty_vcf), hpo_terms=[], max_variants=10
    )
    assert result.variants == []
    assert "No candidate variants" in result.report_markdown
    assert patched_pipeline.call_count == 0
