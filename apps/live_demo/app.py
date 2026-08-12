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
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from agentic_genomics.agents.variant_interpreter.showcase import (
    ACTION_LABEL,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    STATUS_COLOR,
    render_chart,
    run_deterministic_pipeline,
    to_cards,
)

HERE = Path(__file__).resolve().parent
SAMPLE_VCF = HERE / "data" / "showcase_hboc.vcf"
EXAMPLE_SNAPSHOT = HERE / "data" / "example_evidence_snapshot.json"
EXAMPLE_FULL_AGENT_RUN = HERE / "data" / "example_full_agent_run.json"

st.set_page_config(
    page_title="GenomicsCopilot — live demo",
    page_icon="🧬",
    layout="wide",
)


def _badge(label: str, call: str) -> str:
    """A small pill badge, same visual language as the static HTML showcase report."""
    color = STATUS_COLOR.get(call, INK_MUTED)
    return (
        f"<div style='display:inline-block;padding:5px 12px;border-radius:999px;"
        f"background:{color}22;color:{color};border:1px solid {color}55;"
        f"font-weight:700;font-size:0.75rem;white-space:nowrap;'>{label}</div>"
    )


st.markdown(
    f"""
    <div style="padding: 0.5rem 0 0.25rem;">
      <div style="font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; color: {INK_PRIMARY};">
        🧬 GenomicsCopilot
      </div>
      <div style="font-size: 1.05rem; color: {INK_SECONDARY}; margin-top: 0.35rem; max-width: 68ch;">
        Live variant interpretation — VCF + phenotype in, ranked evidence-grounded calls out,
        with a full audit trail of what was queried and why.
      </div>
      <div style="margin-top: 0.65rem; font-size: 0.78rem; color: {INK_MUTED};">
        Open source · MIT · LangGraph · live public bio-databases · research demonstration only, not for clinical use
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

live_tab, example_tab, full_agent_tab, about_tab = st.tabs(
    [
        "▶ Live demo",
        "📋 Recorded example (Reviewer2 second opinion)",
        "🧠 Recorded example (full agentic run)",
        "ℹ️ About the full agent",
    ]
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
        uploaded_vcf_path: str | None = None
        if use_sample:
            vcf_path: str | None = str(SAMPLE_VCF)
        elif uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
            tmp.write(uploaded.getvalue())
            tmp.close()
            vcf_path = uploaded_vcf_path = tmp.name
        else:
            vcf_path = None

        if not vcf_path:
            st.warning("Upload a VCF or use the bundled demo panel.")
            st.stop()

        hpo_terms = [t.strip() for t in hpo_input.split(",") if t.strip()]
        try:
            with st.spinner("Querying MyVariant.info + JAX HPO live…"):
                variants = run_deterministic_pipeline(vcf_path, hpo_terms)
                cards = to_cards(variants)
        finally:
            # Only the sample-uploaded path is ours to delete -- the bundled
            # sample VCF is a repo asset, not a temp file.
            if uploaded_vcf_path:
                Path(uploaded_vcf_path).unlink(missing_ok=True)

        chart_fd, chart_path_str = tempfile.mkstemp(suffix=".png")
        os.close(chart_fd)  # render_chart writes via its own path, not this fd
        chart_path = Path(chart_path_str)
        try:
            render_chart(cards, chart_path)
            st.image(chart_path.read_bytes(), use_container_width=True)
        finally:
            chart_path.unlink(missing_ok=True)

        for card in cards:
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                head_l.markdown(f"#### {card.gene}  \n`{card.coord}`")
                badge_label = f"{ACTION_LABEL.get(card.call, '')} · {card.call}"
                head_r.markdown(
                    f"<div style='text-align:right;'>{_badge(badge_label, card.call)}</div>",
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
                with st.expander("🔍 View raw evidence (the audit trail behind this call)"):
                    st.json(asdict(card))
    else:
        st.info("Configure the run above and click **Run GenomicsCopilot**.")

# --------------------------------------------------------------------------- #
# Recorded example — real Reviewer2 MCP second-opinion round trip
# --------------------------------------------------------------------------- #
with example_tab:
    st.markdown(
        "The full agent also calls **Reviewer2** — a separately built, independent "
        "ACMG engine — live over the **Model Context Protocol**, as a second opinion "
        "on every variant. Reviewer2 is a sibling repo, not published as a package, "
        "so it isn't reachable from this public deployment. What follows is a "
        "**real, previously captured run** (not fabricated, not this demo re-running) "
        "from an environment where Reviewer2 was checked out locally. Its evidence for "
        "this run came from a fixture file, not a separate live lookup — but its "
        "combining-rule logic and gene list are independently written code, and the "
        "disagreement below is a genuine classification Reviewer2's own code reached, "
        "not a live evidence round trip."
    )
    example_cards = json.loads(EXAMPLE_SNAPSHOT.read_text(encoding="utf-8"))
    for card in example_cards:
        with st.container(border=True):
            head_l, head_r = st.columns([3, 1])
            head_l.markdown(f"#### {card['gene']}  \n`{card['coord']}`")
            badge_label = f"{ACTION_LABEL.get(card['call'], '')} · {card['call']}"
            head_r.markdown(
                f"<div style='text-align:right;'>{_badge(badge_label, card['call'])}</div>",
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
            with st.expander("🔍 View raw evidence (the audit trail behind this call)"):
                st.json(card)

# --------------------------------------------------------------------------- #
# Recorded example — real LLM narrative + critic fact-check
# --------------------------------------------------------------------------- #
with full_agent_tab:
    if EXAMPLE_FULL_AGENT_RUN.exists():
        st.markdown(
            "The full agent's two LLM steps, ranking + narrative synthesis, then a "
            "second Claude pass that fact-checks that narrative against the raw "
            "evidence, aren't run live on this public page (see **About the full "
            "agent** for why). What follows is a **real, previously captured run** "
            "against the same bundled panel, generated by "
            "`scripts/generate_full_agent_example.py` — not fabricated, not "
            "paraphrased."
        )
        full_run = json.loads(EXAMPLE_FULL_AGENT_RUN.read_text(encoding="utf-8"))

        st.subheader("Narrative (Claude)")
        st.markdown(full_run.get("report_markdown") or "_(no narrative captured)_")

        critic = full_run.get("critic_review")
        st.subheader("Critic fact-check (a second Claude pass)")
        if critic:
            verdict = critic.get("verdict", "unknown")
            verdict_display = {
                "supported": ("✓ Supported", "success"),
                "partially_supported": ("~ Partially supported", "warning"),
                "unsupported": ("⚠ Unsupported", "error"),
            }.get(verdict, (verdict, "info"))
            getattr(st, verdict_display[1])(f"**Verdict: {verdict_display[0]}**")
            if critic.get("summary"):
                st.caption(critic["summary"])
            for flag in critic.get("flags", []):
                with st.container(border=True):
                    st.markdown(f"**Claim:** {flag.get('claim', '')}")
                    st.caption(f"Concern: {flag.get('concern', '')}")
                    if flag.get("suggestion"):
                        st.caption(f"Suggestion: {flag['suggestion']}")
        else:
            st.info("No critic flags were raised in this captured run.")
    else:
        st.info(
            "No full-agentic-run example has been captured yet. Run "
            "`python scripts/generate_full_agent_example.py` locally with "
            "`ANTHROPIC_API_KEY` set to generate one, then redeploy."
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

st.divider()
st.markdown(
    f"<div style='font-size:0.78rem;color:{INK_MUTED};'>"
    f"🔗 <a href='https://github.com/ankurgenomics/agentic-genomics' style='color:{INK_MUTED};'>"
    f"github.com/ankurgenomics/agentic-genomics</a> · MIT licensed</div>",
    unsafe_allow_html=True,
)
