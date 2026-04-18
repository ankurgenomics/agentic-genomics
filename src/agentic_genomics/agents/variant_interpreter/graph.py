"""LangGraph wiring for GenomicsCopilot (variant interpretation agent)."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentic_genomics.agents.variant_interpreter.nodes import (
    acmg_classify,
    annotate_evidence,
    frequency_filter,
    ingest_variants,
    phenotype_match,
    synthesize_report,
)
from agentic_genomics.agents.variant_interpreter.state import VariantInterpreterState


def build_variant_interpreter():
    """Compile and return a LangGraph app for variant interpretation.

    The graph is intentionally linear. Agentic *reasoning* happens inside
    the LLM node; orchestration is deterministic so the pipeline is easy
    to reason about, test, and visualise.
    """
    graph = StateGraph(VariantInterpreterState)

    graph.add_node("ingest_variants", ingest_variants)
    graph.add_node("annotate_evidence", annotate_evidence)
    graph.add_node("frequency_filter", frequency_filter)
    graph.add_node("phenotype_match", phenotype_match)
    graph.add_node("acmg_classify", acmg_classify)
    graph.add_node("synthesize_report", synthesize_report)

    graph.set_entry_point("ingest_variants")
    graph.add_edge("ingest_variants", "annotate_evidence")
    graph.add_edge("annotate_evidence", "frequency_filter")
    graph.add_edge("frequency_filter", "phenotype_match")
    graph.add_edge("phenotype_match", "acmg_classify")
    graph.add_edge("acmg_classify", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    return graph.compile()


def run_variant_interpreter(
    vcf_path: str,
    hpo_terms: list[str] | None = None,
    max_variants: int = 50,
) -> VariantInterpreterState:
    """Convenience entry point used by CLI / notebook / Streamlit."""
    app = build_variant_interpreter()
    initial = VariantInterpreterState(
        vcf_path=vcf_path,
        hpo_terms=hpo_terms or [],
        max_variants=max_variants,
    )
    result = app.invoke(initial)
    # LangGraph returns a dict; rehydrate to our Pydantic model for convenience.
    return VariantInterpreterState.model_validate(result)
