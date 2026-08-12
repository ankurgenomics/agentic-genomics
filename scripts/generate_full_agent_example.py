#!/usr/bin/env python
"""Capture one real, full end-to-end GenomicsCopilot run for the public demo.

Runs the *complete* graph -- deterministic evidence, ACMG-lite classification,
the live Reviewer2 MCP second opinion, and both LLM nodes (synthesize_report,
critic_review) -- and saves the real output to JSON. That JSON backs the
public live demo's "Recorded example -- full agentic run" tab, so visitors
can see genuine LLM narrative + critic output + a live second opinion without
needing ANTHROPIC_API_KEY exposed on a public page (see apps/live_demo/app.py
and its docstring for why the public deployment can't run this live itself).

Needs ANTHROPIC_API_KEY set and, for a real (not "unavailable") second
opinion, a Reviewer2 checkout as a sibling directory with its own working
environment -- see tools/reviewer2_client.py.

Usage::

    python scripts/generate_full_agent_example.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agentic_genomics.agents.variant_interpreter.graph import run_variant_interpreter
from agentic_genomics.agents.variant_interpreter.showcase import to_cards

REPO_ROOT = Path(__file__).resolve().parent.parent
VCF = REPO_ROOT / "apps" / "live_demo" / "data" / "showcase_hboc.vcf"
HPO_TERMS = ["HP:0003002", "HP:0100615", "HP:0001250"]
OUT = REPO_ROOT / "apps" / "live_demo" / "data" / "example_full_agent_run.json"


def main() -> None:
    print(f"Running the FULL agent graph (LLM narrative + critic + live second opinion) on {VCF} ...")
    state = run_variant_interpreter(
        vcf_path=str(VCF),
        hpo_terms=HPO_TERMS,
        max_variants=10,
    )

    cards = [asdict(c) for c in to_cards(state.variants)]
    payload = {
        "hpo_terms": HPO_TERMS,
        "report_markdown": state.report_markdown,
        "critic_review": state.critic_review.model_dump() if state.critic_review else None,
        "reasoning_trace": state.reasoning_trace,
        "cards": cards,
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Critic verdict: {state.critic_review.verdict if state.critic_review else 'n/a'}")
    print(f"Second opinions available: {sum(1 for c in cards if c['second_opinion_available'])}/{len(cards)}")


if __name__ == "__main__":
    main()
