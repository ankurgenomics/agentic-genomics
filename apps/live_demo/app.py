"""GenomicsCopilot — public live demo (Streamlit Community Cloud).

Runs the *deterministic* half of the agentic-genomics pipeline live, against
real MyVariant.info (gnomAD/ClinVar/CADD/SpliceAI) and JAX HPO data — no API
key, no cost, no rate-limit risk to the maintainer.

What this demo deliberately does NOT do, and why:

* No LLM narrative / critic fact-check. Those two nodes (``synthesize_report``,
  ``critic_review``) need ANTHROPIC_API_KEY. Exposing a shared key on a public
  page invites cost abuse, so this demo stops at the deterministic evidence +
  ACMG-lite classification layer and is honest about that boundary in the UI.
* No live Reviewer2 MCP second opinion. Reviewer2 is a sibling repo, not a
  published package — it isn't available in this deployment's environment.
  A *real, previously captured* run (with Reviewer2 checked out locally) is
  shown as a labeled recorded example instead of faking one live.

Run locally with the full agent (LLM + live second opinion) via
``streamlit run apps/streamlit_demo.py`` — see the repo README.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from agentic_genomics.agents.variant_interpreter.showcase import (
    ACTION_LABEL,
    STATUS_COLOR,
    render_chart,
    run_deterministic_pipeline,
    to_cards,
)

HERE = Path(__file__).resolve().parent
SAMPLE_VCF = HERE / "data" / "showcase_hboc.vcf"
EXAMPLE_SNAPSHOT = HERE / "data" / "example_evidence_snapshot.json"

st.set_page_config(
    page_title="GenomicsCopilot — live demo",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GenomicsCopilot")
st.caption(
    "Live variant interpretation — VCF + phenotype in, ranked evidence-grounded "
    "calls out. Research demonstration only, not for clinical use."
)

live_tab, example_tab, about_tab = st.tabs(
    ["▶ Live demo", "📋 Recorded example (Reviewer2 second opinion)", "ℹ️ About the full agent"]
)

# --------------------------------------------------------------------------- #
# Live demo — deterministic pipeline, real data, no API key
# --------------------------------------------------------------------------- #
with live_tab:
    st.markdown(
        "Runs live against **MyVariant.info** (gnomAD / ClinVar / CADD) and the "
        "**JAX Human Phenotype Ontology** API — real public bio-databases, queried "
        "the moment you click Run. No API key needed: this view stops at the "
        "deterministic evidence + ACMG-lite classification layer."
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        use_sample = st.checkbox("Use the bundled 5-variant demo panel", value=True)
        uploaded = None
        if not use_sample:
            uploaded = st.file_uploader("Upload a VCF", type=["vcf", "vcf.gz"])
    with col2:
        hpo_input = st.text_input(
            "HPO phenotype terms (comma-separated)",
            value="HP:0003002,HP:0100615,HP:0001250",
            help="Defaults match the bundled panel: Breast carcinoma, Ovarian "
            "neoplasm, Seizure.",
        )

    run = st.button("Run GenomicsCopilot", type="primary")

    if run:
        if use_sample:
            vcf_path: str | None = str(SAMPLE_VCF)
        elif uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
            tmp.write(uploaded.getvalue())
            tmp.close()
            vcf_path = tmp.name
        else:
            vcf_path = None

        if not vcf_path:
            st.warning("Upload a VCF or use the bundled demo panel.")
            st.stop()

        hpo_terms = [t.strip() for t in hpo_input.split(",") if t.strip()]
        with st.spinner("Querying MyVariant.info + JAX HPO live…"):
            variants = run_deterministic_pipeline(vcf_path, hpo_terms)
            cards = to_cards(variants)

        chart_path = Path(tempfile.mkstemp(suffix=".png")[1])
        render_chart(cards, chart_path)
        st.image(str(chart_path), use_container_width=True)

        for card in cards:
            color = STATUS_COLOR.get(card.call, "#898781")
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                head_l.markdown(f"### {card.gene}  \n`{card.coord}`")
                head_r.markdown(
                    f"<div style='text-align:right;color:{color};font-weight:700;'>"
                    f"{ACTION_LABEL.get(card.call, '')} · {card.call}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(card.rationale)
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("ClinVar", card.clinvar_significance or "not in ClinVar")
                f2.metric(
                    "gnomAD freq",
                    "absent" if card.gnomad_af is None else f"{card.gnomad_af:.2%}",
                )
                f3.metric("CADD", f"{card.cadd_phred:.1f}" if card.cadd_phred else "n/a")
                f4.metric("Phenotype match", card.phenotype_match)
                if card.criteria:
                    st.write(" ".join(f"`{c}`" for c in card.criteria))
                if card.second_opinion_available:
                    st.caption(
                        "Second opinion (Reviewer2): unavailable in this public "
                        "deployment — see the recorded example tab for a real run."
                    )
    else:
        st.info("Configure the run above and click **Run GenomicsCopilot**.")

# --------------------------------------------------------------------------- #
# Recorded example — real Reviewer2 MCP second-opinion round trip
# --------------------------------------------------------------------------- #
with example_tab:
    st.markdown(
        "The full agent also calls **Reviewer2** — a separately built, independent "
        "ACMG engine — live over the **Model Context Protocol**, as a genuine second "
        "opinion on every variant. Reviewer2 is a sibling repo, not published as a "
        "package, so it isn't reachable from this public Space. What follows is a "
        "**real, previously captured run** (not fabricated, not this demo re-running) "
        "from an environment where Reviewer2 was checked out locally — including one "
        "genuine disagreement the two independently-built engines reached on their own."
    )
    example_cards = json.loads(EXAMPLE_SNAPSHOT.read_text(encoding="utf-8"))
    for card in example_cards:
        color = STATUS_COLOR.get(card["call"], "#898781")
        with st.container(border=True):
            head_l, head_r = st.columns([3, 1])
            head_l.markdown(f"### {card['gene']}  \n`{card['coord']}`")
            head_r.markdown(
                f"<div style='text-align:right;color:{color};font-weight:700;'>"
                f"{ACTION_LABEL.get(card['call'], '')} · {card['call']}</div>",
                unsafe_allow_html=True,
            )
            if card["second_opinion_available"]:
                if card["second_opinion_flagged"]:
                    st.error(
                        f"⚠ **Reviewer2 (independent engine, via MCP):** "
                        f"“{card['second_opinion_call']}” — flagged for human review. "
                        f"{card.get('second_opinion_note') or ''}"
                    )
                else:
                    st.success(
                        f"✓ **Reviewer2 (independent engine, via MCP):** "
                        f"concordant — “{card['second_opinion_call']}”"
                    )

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with about_tab:
    st.markdown(
        """
The full GenomicsCopilot agent adds two more steps on top of what this public
demo shows:

1. **`synthesize_report`** (Claude) — ranks the candidate variants and writes
   a plain-English narrative across the whole panel.
2. **`critic_review`** (Claude) — re-reads that narrative against the raw
   evidence JSON and flags any claim that doesn't map back to a fact.

Both require `ANTHROPIC_API_KEY` and aren't run here to avoid exposing a
shared key on a public page. Run the full agent locally:

```bash
git clone https://github.com/ankurgenomics/agentic-genomics
cd agentic-genomics
pip install -e ".[demo]"
export ANTHROPIC_API_KEY=sk-...
streamlit run apps/streamlit_demo.py
```

Source: [github.com/ankurgenomics/agentic-genomics](https://github.com/ankurgenomics/agentic-genomics)
        """
    )
