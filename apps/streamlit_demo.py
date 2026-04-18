"""Streamlit demo for GenomicsCopilot.

Run with::

    streamlit run apps/streamlit_demo.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from agentic_genomics.agents.variant_interpreter import run_variant_interpreter

st.set_page_config(
    page_title="GenomicsCopilot · agentic-genomics",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GenomicsCopilot")
st.caption(
    "Autonomous variant interpretation with reasoning traces. "
    "Research / educational use only — not for clinical use."
)

with st.sidebar:
    st.header("Run configuration")
    sample_vcf_path = Path("data/samples/proband_demo.vcf")
    use_sample = st.checkbox(
        "Use bundled demo VCF (`data/samples/proband_demo.vcf`)",
        value=sample_vcf_path.exists(),
    )
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload a VCF", type=["vcf", "vcf.gz"])

    hpo_input = st.text_input(
        "HPO terms (comma-separated)",
        value="HP:0001250,HP:0001263",
        help="e.g. HP:0001250 (Seizure), HP:0001263 (Global developmental delay)",
    )
    max_variants = st.slider("Max variants", 5, 200, 25)

    st.divider()
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    if api_key_set:
        st.success("ANTHROPIC_API_KEY detected")
    else:
        st.error("Set ANTHROPIC_API_KEY in your environment (.env) before running.")

    run_button = st.button("Run GenomicsCopilot", type="primary", disabled=not api_key_set)


def _resolve_vcf_path() -> str | None:
    if use_sample and sample_vcf_path.exists():
        return str(sample_vcf_path)
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
        tmp.write(uploaded.getvalue())
        tmp.close()
        return tmp.name
    return None


if run_button:
    vcf_path = _resolve_vcf_path()
    if not vcf_path:
        st.warning("Please provide a VCF (upload one or enable the demo VCF).")
        st.stop()

    hpo_terms = [t.strip() for t in hpo_input.split(",") if t.strip()]
    with st.spinner("Running GenomicsCopilot…"):
        state = run_variant_interpreter(
            vcf_path=vcf_path,
            hpo_terms=hpo_terms,
            max_variants=max_variants,
        )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Ranked interpretation report")
        st.markdown(state.report_markdown)
        st.download_button(
            "Download report (.md)",
            data=state.report_markdown,
            file_name="variant_report.md",
            mime="text/markdown",
        )

    with right:
        st.subheader("Reasoning trace")
        for step in state.reasoning_trace:
            with st.expander(f"🧠 {step['node']}"):
                st.write(step["summary"])
                if step.get("evidence"):
                    st.json(step["evidence"])

        st.subheader("Annotated variants (JSON)")
        st.json(
            [v.model_dump() for v in state.variants],
            expanded=False,
        )
else:
    st.info(
        "Configure the run in the sidebar and click **Run GenomicsCopilot**. "
        "The bundled demo VCF contains seven curated variants suitable for demonstration."
    )
