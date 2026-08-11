"""Shared rendering + deterministic-run logic for GenomicsCopilot showcases.

Used by both ``scripts/generate_showcase_report.py`` (static HTML/PNG report)
and ``apps/hf_space/app.py`` (the public live demo, deterministic-only —
no ANTHROPIC_API_KEY required). Keeping this in one place means the two
surfaces can never silently drift apart on colors, tiers, or field mapping.

Design notes (see the dataviz skill for the full rationale):

* Color is status, not decoration: good / warning / serious / critical map
  onto the 5-tier ACMG scale, with Likely-Benign and Benign sharing "good"
  (matching how a clinician would actually act on them).
* Every status color ships with a text label ("ACT" / "MONITOR" / "NO
  ACTION") -- never color alone.
* One variant per row guarantees no label collisions regardless of how
  many tiers are represented; each dot gets its own vertical lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed -- this runs headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from agentic_genomics.agents.variant_interpreter.nodes import (
    acmg_classify,
    annotate_evidence,
    ingest_variants,
    phenotype_score,
)
from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    VariantInterpreterState,
)

try:
    # second_opinion_review (the Reviewer2-over-MCP node) lands separately
    # from the deterministic-showcase work this module backs -- import it
    # opportunistically so this module works against either version of
    # nodes.py, and degrades to "no second opinion" rather than ImportError
    # when it isn't present yet.
    from agentic_genomics.agents.variant_interpreter.nodes import (
        second_opinion_review as _second_opinion_review,
    )
except ImportError:
    _second_opinion_review = None

STATUS_COLOR = {
    "Pathogenic": "#d03b3b",  # critical
    "Likely Pathogenic": "#ec835a",  # serious
    "Uncertain Significance": "#fab219",  # warning
    "Likely Benign": "#0ca30c",  # good
    "Benign": "#0ca30c",  # good
}
ACTION_LABEL = {
    "Pathogenic": "ACT",
    "Likely Pathogenic": "ACT",
    "Uncertain Significance": "MONITOR",
    "Likely Benign": "NO ACTION",
    "Benign": "NO ACTION",
}
TIER_ORDER = [
    "Benign",
    "Likely Benign",
    "Uncertain Significance",
    "Likely Pathogenic",
    "Pathogenic",
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def run_deterministic_pipeline(vcf_path: Path | str, hpo_terms: list[str]) -> list[AnnotatedVariant]:
    """Run ingest -> annotate -> phenotype -> classify -> second_opinion. No LLM.

    Deliberately *skips* ``frequency_filter``: a real diagnostic run drops
    common variants (gnomAD popmax AF > 1%) from the candidate list before
    ACMG classification -- correct behavior in production, but it would
    silently remove common Likely-Benign/Benign examples that are useful
    to show in a demo. This is a showcase-only pipeline variant, not how
    the CLI/Streamlit-full-agent app actually runs.

    ``second_opinion_review`` calls Reviewer2 -- a separately implemented
    ACMG engine in a sibling repo -- over MCP for an independent re-review
    of each call. It degrades to "unavailable" per-variant if Reviewer2
    isn't checked out alongside this repo; see tools/reviewer2_client.py.
    """
    nodes = [ingest_variants, annotate_evidence, phenotype_score, acmg_classify]
    if _second_opinion_review is not None:
        nodes.append(_second_opinion_review)

    state = VariantInterpreterState(vcf_path=str(vcf_path), hpo_terms=hpo_terms)
    for node in nodes:
        update = node(state)
        state = state.model_copy(update=update)
    return state.variants


@dataclass
class VariantCard:
    """Flattened view of one AnnotatedVariant, shaped for rendering."""

    gene: str
    coord: str
    call: str
    criteria: list[str]
    rationale: str
    clinvar_significance: str | None
    clinvar_submitters: int | None
    gnomad_af: float | None
    cadd_phred: float | None
    phenotype_match: str
    linked_diseases: list[str]
    second_opinion_available: bool
    second_opinion_call: str | None
    second_opinion_flagged: bool
    second_opinion_note: str | None
    second_opinion_extra_note: str | None


def _fmt_af(af: float | None) -> str:
    if af is None:
        return "absent from gnomAD"
    if af == 0:
        return "0 (absent)"
    return f"{af:.2%}" if af >= 0.001 else f"{af:.2e}"


def _first_conflict_message(
    second_opinion: object, conflict_type: str, *, exclude: bool = False
) -> str | None:
    """Pull the first Reviewer2 conflict message matching (or not) a type.

    Reviewer2 can raise several *kinds* of flag on one variant -- a real
    classification disagreement is a different claim from "ClinVar itself
    has conflicting submitters", and conflating them mislabels variants
    where the two engines actually agree (see reviewer2_client.py). This
    keeps the two kinds of note separate in the rendered report.
    """
    conflicts = getattr(second_opinion, "conflicts", None) or []
    for c in conflicts:
        matches = c.type == conflict_type
        if matches != exclude:
            return c.message
    return None


def to_cards(variants: list[AnnotatedVariant]) -> list[VariantCard]:
    cards = []
    for av in variants:
        v = av.variant
        cards.append(
            VariantCard(
                gene=v.gene or "?",
                # v.chrom already carries a "chr" prefix from the VCF's own
                # CHROM column (vcf_parser doesn't strip it) -- don't add a
                # second one.
                coord=f"{v.chrom}:{v.pos} {v.ref}>{v.alt}",
                call=av.acmg.call if av.acmg else "Uncertain Significance",
                criteria=av.acmg.criteria_triggered if av.acmg else [],
                rationale=av.acmg.rationale if av.acmg else "",
                clinvar_significance=av.clinical.clinvar_significance,
                clinvar_submitters=av.clinical.clinvar_submitters,
                gnomad_af=av.population.gnomad_af_popmax or av.population.gnomad_af,
                cadd_phred=av.functional.cadd_phred,
                phenotype_match=av.phenotype.match_strength,
                # JAX's disease-annotation API sometimes lists the same
                # disease name twice under different cross-referenced IDs
                # (e.g. an ORPHA and an OMIM entry both named "Peutz-Jeghers
                # syndrome") -- dedupe before truncating so the report
                # doesn't repeat itself.
                linked_diseases=list(dict.fromkeys(av.phenotype.linked_diseases))[:3],
                # ``second_opinion`` only exists on AnnotatedVariant once the
                # Reviewer2-over-MCP node lands (see the import fallback
                # above) -- getattr keeps this module working against either
                # version of state.py instead of raising AttributeError.
                second_opinion_available=bool(
                    getattr(av, "second_opinion", None) and av.second_opinion.available
                ),
                second_opinion_call=(
                    av.second_opinion.independent_classification
                    if getattr(av, "second_opinion", None)
                    else None
                ),
                second_opinion_flagged=bool(
                    getattr(av, "second_opinion", None) and av.second_opinion.materially_disagrees
                ),
                second_opinion_note=_first_conflict_message(
                    getattr(av, "second_opinion", None), "classification_disagreement"
                ),
                second_opinion_extra_note=_first_conflict_message(
                    getattr(av, "second_opinion", None),
                    "classification_disagreement",
                    exclude=True,
                ),
            )
        )
    # Order by ACMG severity so the chart forms a clean gradient, most severe first.
    cards.sort(key=lambda c: -TIER_ORDER.index(c.call) if c.call in TIER_ORDER else 0)
    return cards


_X_STEP = 1.7  # horizontal distance between adjacent ACMG tiers on the chart


def render_chart(cards: list[VariantCard], out_path: Path | str) -> None:
    n = len(cards)
    fig, ax = plt.subplots(figsize=(13, 1.35 * n + 2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [i * _X_STEP for i in range(5)]  # one slot per ACMG tier
    row_x = xs[0] - 0.6  # left anchor for gene labels + connector lines

    # Background action-band zones (traffic-light metaphor, readable with no
    # genomics knowledge at all).
    ax.axvspan(xs[0] - 0.6, xs[1] + 0.6, color="#0ca30c", alpha=0.07, zorder=0)
    ax.axvspan(xs[1] + 0.6, xs[2] + 0.6, color="#fab219", alpha=0.09, zorder=0)
    ax.axvspan(xs[2] + 0.6, xs[4] + 0.6, color="#d03b3b", alpha=0.07, zorder=0)

    y_positions = list(range(n))[::-1]  # top row = first card (most severe)
    for y, card in zip(y_positions, cards, strict=True):
        tier_idx = TIER_ORDER.index(card.call) if card.call in TIER_ORDER else 2
        x = xs[tier_idx]
        color = STATUS_COLOR.get(card.call, INK_MUTED)

        # Thin connector from the axis to the dot keeps the eye anchored to
        # the row without adding a real gridline across the whole chart.
        ax.plot([row_x, x], [y, y], color=GRIDLINE, lw=1, zorder=1)
        ax.plot(
            x, y, marker="o", markersize=24, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=3,
        )

        # Direct label: action word + ACMG call, stacked just above the dot.
        ax.annotate(
            ACTION_LABEL[card.call],
            (x, y), xytext=(0, 20), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=color,
        )
        ax.annotate(
            card.call, (x, y), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY,
        )

        # Row label on the left: gene + coordinate.
        ax.annotate(
            f"{card.gene}", (row_x, y), xytext=(-14, 0), textcoords="offset points",
            ha="right", va="center", fontsize=12, fontweight="bold", color=INK_PRIMARY,
        )
        ax.annotate(
            card.coord, (row_x, y), xytext=(-14, -15), textcoords="offset points",
            ha="right", va="center", fontsize=8, color=INK_MUTED, family="monospace",
        )

    ax.set_xlim(row_x - 3.0, xs[4] + 1.0)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xticks(xs)
    # Tier name only on the axis -- the plain-English gloss ("no action" /
    # "monitor" / "act") already sits directly above each dot via
    # ACTION_LABEL, so it isn't repeated here where five long strings packed
    # this close together (~1 tier-width apart) collided into each other.
    ax.set_xticklabels(TIER_ORDER, fontsize=9.5, color=INK_SECONDARY)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=10, color="#0ca30c", label="No action"),
        Line2D([0], [0], marker="o", linestyle="", markersize=10, color="#fab219", label="Monitor"),
        Line2D([0], [0], marker="o", linestyle="", markersize=10, color="#d03b3b", label="Act"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.02),
        ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
        handletextpad=0.4, columnspacing=1.2,
    )

    ax.set_title(
        "GenomicsCopilot — where each variant lands",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, pad=34, loc="left",
    )
    fig.text(
        0.01, 0.005,
        "Real ClinVar / gnomAD / CADD evidence, retrieved live from MyVariant.info "
        "— research demonstration, not for clinical use.",
        fontsize=7.5, color=INK_MUTED,
    )

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
