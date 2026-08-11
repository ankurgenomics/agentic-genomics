#!/usr/bin/env python
"""Generate a presentation-ready showcase report for GenomicsCopilot.

Runs the pipeline's five *deterministic* nodes (no LLM call, no API key
needed) against a small VCF of real, live-verified ClinVar variants, then
renders:

  reports/showcase/evidence_snapshot.json   raw structured output (reproducible)
  reports/showcase/variant_ranking.png      hero chart — where each variant lands
  reports/showcase/showcase_report.html     one-page report, readable by anyone

Usage::

    python scripts/generate_showcase_report.py
    python scripts/generate_showcase_report.py --vcf path/to/other.vcf \\
        --hpo HP:0003002,HP:0100615 --out reports/showcase

Design notes (see the dataviz skill for the full rationale):

* Color is status, not decoration: good / warning / serious / critical map
  onto the 5-tier ACMG scale, with Likely-Benign and Benign sharing "good"
  (matching how a clinician would actually act on them).
* Every status color ships with a text label ("ACT" / "MONITOR" / "NO
  ACTION") — never color alone.
* One variant per row guarantees no label collisions regardless of how
  many tiers are represented; each dot gets its own vertical lane.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentic_genomics.agents.variant_interpreter.showcase import (
    ACTION_LABEL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    STATUS_COLOR,
    SURFACE,
    VariantCard,
    _fmt_af,
    render_chart,
    run_deterministic_pipeline,
    to_cards,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VCF = REPO_ROOT / "data" / "samples" / "showcase_hboc.vcf"
# Breast carcinoma, Ovarian neoplasm, Seizure — the panel spans cancer and
# epilepsy genes on purpose (see data/samples/README.md), so the phenotype
# terms deliberately span both rather than pretending it's a single-indication case.
DEFAULT_HPO = ["HP:0003002", "HP:0100615", "HP:0001250"]
DEFAULT_OUT = REPO_ROOT / "reports" / "showcase"


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def render_html(cards: list[VariantCard], hpo_terms: list[str], chart_filename: str, out_path: Path) -> None:
    card_html = []
    for card in cards:
        color = STATUS_COLOR.get(card.call, INK_MUTED)
        criteria_html = "".join(f'<span class="chip">{c}</span>' for c in card.criteria) or (
            '<span class="chip chip-muted">none triggered</span>'
        )
        # " · " not ", " -- OMIM-style disease names routinely contain their
        # own internal commas (e.g. "Pancreatic cancer, susceptibility to, 3"),
        # so joining multiple diseases with commas makes them unreadable as
        # one run-on string.
        diseases = " · ".join(card.linked_diseases) if card.linked_diseases else "—"
        clinvar = card.clinvar_significance or "not in ClinVar"
        submitters = f" ({card.clinvar_submitters} submitters)" if card.clinvar_submitters else ""
        cadd = f"{card.cadd_phred:.1f}" if card.cadd_phred is not None else "n/a"

        extra_note_html = (
            f'<div class="second-opinion-extra">Reviewer2 also notes: {card.second_opinion_extra_note}</div>'
            if card.second_opinion_extra_note
            else ""
        )
        if card.second_opinion_available and card.second_opinion_flagged:
            second_opinion_html = (
                f'<div class="second-opinion flagged">'
                f"&#9888; <b>Second opinion (Reviewer2, via MCP):</b> independent call "
                f'“{card.second_opinion_call}” — flagged for human review'
                + (f": {card.second_opinion_note}" if card.second_opinion_note else "")
                + "</div>" + extra_note_html
            )
        elif card.second_opinion_available and (card.second_opinion_call or "").lower() == card.call.lower():
            second_opinion_html = (
                f'<div class="second-opinion concordant">'
                f"&#10003; <b>Second opinion (Reviewer2, independent engine, via MCP):</b> "
                f'concordant — “{card.second_opinion_call}”'
                "</div>" + extra_note_html
            )
        elif card.second_opinion_available:
            second_opinion_html = (
                f'<div class="second-opinion concordant">'
                f"&#8776; <b>Second opinion (Reviewer2, independent engine, via MCP):</b> "
                f'“{card.second_opinion_call}” — an adjacent tier, same clinical action, not flagged'
                "</div>" + extra_note_html
            )
        else:
            second_opinion_html = (
                '<div class="second-opinion unavailable">Second opinion unavailable '
                "(Reviewer2 MCP server not reachable for this run).</div>"
            )

        card_html.append(f"""
        <article class="card">
          <div class="card-head">
            <div>
              <h3>{card.gene}</h3>
              <div class="coord">{card.coord}</div>
            </div>
            <div class="badge" style="background:{color}22;color:{color};border-color:{color}55;">
              {ACTION_LABEL[card.call]} &middot; {card.call}
            </div>
          </div>
          <p class="rationale">{card.rationale}</p>
          <dl class="facts">
            <div><dt>ClinVar</dt><dd>{clinvar}{submitters}</dd></div>
            <div><dt>gnomAD frequency</dt><dd>{_fmt_af(card.gnomad_af)}</dd></div>
            <div><dt>CADD score</dt><dd>{cadd}</dd></div>
            <div><dt>Phenotype match</dt><dd>{card.phenotype_match}</dd></div>
            <div class="span2"><dt>Linked conditions</dt><dd>{diseases}</dd></div>
          </dl>
          <div class="chips">{criteria_html}</div>
          {second_opinion_html}
        </article>""")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GenomicsCopilot — showcase report</title>
<style>
  :root {{
    --surface: {SURFACE}; --page: #f9f9f7; --ink: {INK_PRIMARY};
    --ink-2: {INK_SECONDARY}; --ink-muted: {INK_MUTED}; --line: {GRIDLINE};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 24px 64px; }}
  header {{ margin-bottom: 28px; }}
  header .eyebrow {{
    font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
    color: var(--ink-muted); font-weight: 600; margin-bottom: 6px;
  }}
  h1 {{ font-size: 28px; margin: 0 0 8px; }}
  header p {{ color: var(--ink-2); margin: 0; max-width: 68ch; }}
  .meta {{
    display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 14px;
    font-size: 13px; color: var(--ink-2);
  }}
  .meta b {{ color: var(--ink); }}
  .chart {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 8px; margin: 24px 0 36px;
    overflow-x: auto;
  }}
  .chart img {{ width: 100%; height: auto; display: block; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px; display: flex; flex-direction: column;
    gap: 12px; min-width: 0;
  }}
  .card-head {{
    display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
  }}
  .card-head h3 {{ margin: 0; font-size: 17px; }}
  .coord {{ font-family: ui-monospace, monospace; font-size: 11px; color: var(--ink-muted); margin-top: 2px; }}
  .badge {{
    flex-shrink: 0; font-size: 11px; font-weight: 700; padding: 6px 10px;
    border-radius: 999px; border: 1px solid; white-space: nowrap;
  }}
  .rationale {{ font-size: 13.5px; color: var(--ink-2); margin: 0; }}
  dl.facts {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px;
    margin: 0; font-size: 12.5px;
  }}
  dl.facts .span2 {{ grid-column: 1 / -1; }}
  dl.facts dt {{ color: var(--ink-muted); font-weight: 600; margin: 0 0 2px; }}
  dl.facts dd {{ margin: 0; color: var(--ink); word-break: break-word; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .chip {{
    font-size: 11px; font-family: ui-monospace, monospace; background: #f0efec;
    border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px; color: var(--ink-2);
  }}
  .chip-muted {{ color: var(--ink-muted); font-style: italic; font-family: system-ui, sans-serif; }}
  .second-opinion {{
    font-size: 12px; line-height: 1.45; padding: 8px 10px; border-radius: 8px;
    border-left: 3px solid var(--line); background: #f4f3f0; color: var(--ink-2);
  }}
  .second-opinion b {{ color: var(--ink); }}
  .second-opinion.concordant {{ border-left-color: #0ca30c; }}
  .second-opinion.flagged {{ border-left-color: #d03b3b; background: #d03b3b11; }}
  .second-opinion.unavailable {{ color: var(--ink-muted); font-style: italic; }}
  .second-opinion-extra {{
    font-size: 11.5px; color: var(--ink-muted); margin-top: 4px; padding-top: 4px;
    border-top: 1px dashed var(--line);
  }}
  footer {{
    margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
    font-size: 12px; color: var(--ink-muted);
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">GenomicsCopilot &middot; agentic-genomics</div>
    <h1>Variant interpretation report</h1>
    <p>Five variants, ranked and classified from live public evidence (gnomAD, ClinVar,
       CADD, HPO), with the full reasoning kept visible — not a black-box score. Each
       call is also independently re-reviewed by Reviewer2, a separately built ACMG
       engine, over MCP — a genuine second opinion, not a rerun of the same logic.</p>
    <div class="meta">
      <span><b>{len(cards)}</b> variants analyzed</span>
      <span><b>Phenotype:</b> {", ".join(hpo_terms)}</span>
      <span><b>Sources:</b> MyVariant.info (gnomAD/ClinVar/CADD), JAX HPO ontology</span>
      <span><b>Second opinion:</b> Reviewer2 (independent ACMG engine, via MCP)</span>
    </div>
  </header>

  <div class="chart"><img src="{chart_filename}" alt="Variant ranking chart"></div>

  <div class="grid">
    {"".join(card_html)}
  </div>

  <footer>
    Research demonstration only — not for clinical use. Generated by
    <code>scripts/generate_showcase_report.py</code> from live MyVariant.info /
    JAX ontology / gnomAD data; re-run to refresh. This showcase run skips the
    production frequency pre-filter so common Likely-Benign/Benign examples
    stay visible — the CLI and Streamlit app both apply it by default.
  </footer>
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path, default=DEFAULT_VCF)
    parser.add_argument("--hpo", type=str, default=",".join(DEFAULT_HPO))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    hpo_terms = [t.strip() for t in args.hpo.split(",") if t.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Running deterministic pipeline on {args.vcf} ...")
    variants = run_deterministic_pipeline(args.vcf, hpo_terms)
    cards = to_cards(variants)

    snapshot_path = args.out / "evidence_snapshot.json"
    snapshot_path.write_text(
        json.dumps([asdict(c) for c in cards], indent=2), encoding="utf-8"
    )
    print(f"Wrote {snapshot_path}")

    chart_path = args.out / "variant_ranking.png"
    render_chart(cards, chart_path)
    print(f"Wrote {chart_path}")

    html_path = args.out / "showcase_report.html"
    render_html(cards, hpo_terms, chart_path.name, html_path)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
