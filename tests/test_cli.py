"""Smoke tests for the ``agentic-genomics`` Typer CLI.

These tests exercise the command surface (help, version, interpret) without
touching the network or an LLM. The heavy lifting is patched at the node
level, mirroring ``tests/test_graph_integration.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_genomics.cli.main import app

runner = CliRunner()

DEMO_VCF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "samples"
    / "proband_demo.vcf"
)


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "agentic-genomics" in result.stdout


def test_cli_help_lists_interpret():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "interpret" in result.stdout


def test_cli_interpret_happy_path(tmp_path, monkeypatch):
    """Run `agentic-genomics interpret ...` with all IO stubbed."""
    from agentic_genomics.agents.variant_interpreter import nodes as nodes_mod
    from agentic_genomics.agents.variant_interpreter.state import (
        ClinicalEvidence,
        FunctionalScores,
        PhenotypeMatch,
        PopulationFrequency,
    )

    class _FakeLLM:
        def invoke(self, messages):
            class _R:
                content = "# Variant interpretation report\n\nStubbed output."

            return _R()

    monkeypatch.setattr(
        nodes_mod.myvariant, "fetch_variant_record", lambda v: {}
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_population", lambda r: PopulationFrequency()
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_functional", lambda r: FunctionalScores()
    )
    monkeypatch.setattr(
        nodes_mod.myvariant, "extract_clinvar", lambda r: ClinicalEvidence()
    )
    monkeypatch.setattr(
        nodes_mod.hpo,
        "score_phenotype_match",
        lambda gene, terms: PhenotypeMatch(match_strength="none"),
    )
    monkeypatch.setattr(nodes_mod, "get_llm", lambda: _FakeLLM())

    report_path = tmp_path / "reports" / "out.md"
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "interpret",
            "--vcf",
            str(DEMO_VCF),
            "--hpo",
            "HP:0001250,HP:0003002",
            "--out",
            str(report_path),
            "--trace-out",
            str(trace_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Report written" in result.stdout
    assert report_path.read_text().startswith("# Variant interpretation report")
    trace = json.loads(trace_path.read_text())
    assert [s["node"] for s in trace] == [
        "ingest_variants",
        "annotate_evidence",
        "frequency_filter",
        "phenotype_match",
        "acmg_classify",
        "synthesize_report",
    ]


def test_cli_rejects_missing_vcf(tmp_path):
    result = runner.invoke(
        app,
        [
            "interpret",
            "--vcf",
            str(tmp_path / "does_not_exist.vcf"),
        ],
    )
    assert result.exit_code != 0
