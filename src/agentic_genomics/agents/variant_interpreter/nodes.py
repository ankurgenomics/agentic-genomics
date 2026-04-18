"""LangGraph nodes for the variant interpreter.

Each node is a small function ``(state) -> partial_state_update``. Most
nodes are deterministic; ``synthesize_report`` and ``critic_review`` call
the LLM.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_genomics.agents.variant_interpreter.prompts import (
    CRITIC_SYSTEM,
    CRITIC_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER_TEMPLATE,
)
from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    CriticFlag,
    CriticReview,
    VariantInterpreterState,
)
from agentic_genomics.agents.variant_interpreter.tools import (
    acmg_lite,
    hpo,
    myvariant,
)
from agentic_genomics.agents.variant_interpreter.tools.vcf_parser import parse_vcf
from agentic_genomics.core.llm import get_llm
from agentic_genomics.core.logging import trace

_FREQUENCY_CUTOFF = 0.01  # drop variants with popmax AF > 1% from candidate list


def ingest_variants(state: VariantInterpreterState) -> dict:
    """Parse the VCF into :class:`AnnotatedVariant` stubs."""
    variants = parse_vcf(state.vcf_path, max_variants=state.max_variants)
    annotated = [AnnotatedVariant(variant=v) for v in variants]
    step = trace(
        "ingest_variants",
        f"Parsed {len(annotated)} variants from {state.vcf_path}",
        count=len(annotated),
    )
    return {"variants": annotated, "reasoning_trace": [step.to_dict()]}


def annotate_evidence(state: VariantInterpreterState) -> dict:
    """Populate gnomAD / ClinVar / functional-score evidence for each variant."""
    updated: list[AnnotatedVariant] = []
    hits_clinvar = 0
    for av in state.variants:
        record = myvariant.fetch_variant_record(av.variant)
        av = av.model_copy(
            update={
                "population": myvariant.extract_population(record),
                "functional": myvariant.extract_functional(record),
                "clinical": myvariant.extract_clinvar(record),
            }
        )
        if av.clinical.clinvar_significance:
            hits_clinvar += 1
        updated.append(av)

    step = trace(
        "annotate_evidence",
        f"Annotated {len(updated)} variants; {hits_clinvar} had ClinVar records",
        count=len(updated),
        clinvar_hits=hits_clinvar,
    )
    return {
        "variants": updated,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def frequency_filter(state: VariantInterpreterState) -> dict:
    """Drop common variants from the candidate list."""
    kept: list[AnnotatedVariant] = []
    dropped = 0
    for av in state.variants:
        af = av.population.gnomad_af_popmax or av.population.gnomad_af
        if af is not None and af > _FREQUENCY_CUTOFF:
            dropped += 1
            continue
        kept.append(av)

    step = trace(
        "frequency_filter",
        f"Dropped {dropped} common variants (popmax AF > {_FREQUENCY_CUTOFF}); "
        f"{len(kept)} candidates remain",
        dropped=dropped,
        kept=len(kept),
        cutoff=_FREQUENCY_CUTOFF,
    )
    return {
        "variants": kept,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def phenotype_score(state: VariantInterpreterState) -> dict:
    """Annotate each candidate with Phrank-style HPO semantic similarity.

    Pre-computes an IC corpus from the full set of candidate genes so that
    every variant is scored against the same IC distribution.
    """
    candidate_genes = sorted({
        av.variant.gene for av in state.variants if av.variant.gene
    })
    corpus, corpus_size = hpo.build_annotation_corpus(candidate_genes)

    updated: list[AnnotatedVariant] = []
    strong = moderate = 0
    for av in state.variants:
        match = hpo.score_phenotype_match(
            av.variant.gene,
            state.hpo_terms,
            annotation_corpus=corpus or None,
            corpus_size=corpus_size or None,
        )
        if match.match_strength == "strong":
            strong += 1
        elif match.match_strength == "moderate":
            moderate += 1
        updated.append(av.model_copy(update={"phenotype": match}))

    step = trace(
        "phenotype_score",
        f"Phenotype scoring complete — strong: {strong}, moderate: {moderate}",
        strong=strong,
        moderate=moderate,
        corpus_terms=len(corpus),
    )
    return {
        "variants": updated,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def acmg_classify(state: VariantInterpreterState) -> dict:
    """Assign ACMG-lite classifications using the Richards 2015 combining rules."""
    updated = []
    by_call: dict[str, int] = {}
    for av in state.variants:
        assessment = acmg_lite.classify(av)
        by_call[assessment.call] = by_call.get(assessment.call, 0) + 1
        updated.append(av.model_copy(update={"acmg": assessment}))

    step = trace(
        "acmg_classify",
        f"ACMG-lite classification complete: {by_call}",
        counts=by_call,
    )
    return {
        "variants": updated,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def _variants_to_json(variants: list[AnnotatedVariant]) -> str:
    """Serialise variants into a compact JSON blob for the LLM prompt."""
    compact = []
    for av in variants:
        compact.append(
            {
                "key": av.variant.key,
                "gene": av.variant.gene,
                "consequence": av.variant.consequence,
                "hgvs_p": av.variant.hgvs_p,
                "hgvs_c": av.variant.hgvs_c,
                "gnomad_af_popmax": av.population.gnomad_af_popmax,
                "gnomad_af": av.population.gnomad_af,
                "gnomad_hom": av.population.gnomad_hom,
                "gnomad_pli": av.functional.gnomad_pli,
                "gnomad_loeuf": av.functional.gnomad_loeuf,
                "cadd_phred": av.functional.cadd_phred,
                "revel": av.functional.revel,
                "spliceai_max": av.functional.spliceai_max,
                "clinvar": av.clinical.clinvar_significance,
                "clinvar_review_status": av.clinical.clinvar_review_status,
                "clinvar_conflicts": av.clinical.clinvar_conflicts,
                "phenotype_match_strength": av.phenotype.match_strength,
                "phenotype_score": av.phenotype.score,
                "matched_hpo_terms": av.phenotype.matched_hpo_terms,
                "linked_diseases": av.phenotype.linked_diseases,
                "acmg_call": av.acmg.call if av.acmg else None,
                "acmg_criteria": av.acmg.criteria_triggered if av.acmg else [],
            }
        )
    return json.dumps(compact, indent=2)


def synthesize_report(state: VariantInterpreterState) -> dict:
    """Use the LLM to produce a ranked, explainable markdown report."""
    if not state.variants:
        step = trace(
            "synthesize_report",
            "No candidate variants after filtering — nothing to synthesise.",
        )
        return {
            "report_markdown": (
                "# Variant interpretation report\n\n"
                "> Research/educational output only; not for clinical use.\n\n"
                "No candidate variants remained after frequency filtering."
            ),
            "reasoning_trace": state.reasoning_trace + [step.to_dict()],
        }

    llm = get_llm()
    prompt = SYNTHESIZER_USER_TEMPLATE.format(
        hpo_terms=", ".join(state.hpo_terms) or "(none provided)",
        variants_json=_variants_to_json(state.variants),
    )
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    report = response.content if isinstance(response.content, str) else str(response.content)

    step = trace(
        "synthesize_report",
        f"LLM synthesiser produced a {len(report)}-char markdown report.",
        report_chars=len(report),
    )
    return {
        "report_markdown": report,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def _parse_critic_response(raw: str) -> CriticReview:
    """Parse the critic's JSON response; degrade gracefully on bad output."""
    # Strip common prefix/suffix text (LLMs sometimes wrap JSON in prose).
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return CriticReview(
            verdict="partially_supported",
            summary="Critic returned non-JSON output; defaulting to partially_supported.",
            flags=[],
        )
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return CriticReview(
            verdict="partially_supported",
            summary="Critic JSON was malformed; defaulting to partially_supported.",
            flags=[],
        )

    raw_flags = payload.get("flags") or []
    flags: list[CriticFlag] = []
    for f in raw_flags:
        if not isinstance(f, dict):
            continue
        flags.append(
            CriticFlag(
                severity=f.get("severity", "warn") if f.get("severity") in {"info", "warn", "error"} else "warn",
                claim=str(f.get("claim", ""))[:800],
                concern=str(f.get("concern", ""))[:800],
                suggestion=str(f.get("suggestion", ""))[:400],
            )
        )
    verdict = payload.get("verdict", "partially_supported")
    if verdict not in {"supported", "partially_supported", "unsupported"}:
        verdict = "partially_supported"
    return CriticReview(
        verdict=verdict,
        summary=str(payload.get("summary", ""))[:1000],
        flags=flags,
    )


def critic_review(state: VariantInterpreterState) -> dict:
    """Second LLM pass: fact-check the report against the evidence JSON.

    This is the one genuinely agentic loop in the pipeline. The critic
    sees the same evidence JSON the synthesiser saw, plus the synthesiser's
    draft, and returns structured flags for any unsupported claims. We
    append the flags to the report rather than editing it — so the user
    always sees the original draft AND its critique.
    """
    if not state.variants or not state.report_markdown:
        step = trace(
            "critic_review",
            "Skipped — no report to review.",
        )
        return {
            "critic_review": CriticReview(
                verdict="supported",
                summary="No report drafted; nothing to review.",
            ),
            "reasoning_trace": state.reasoning_trace + [step.to_dict()],
        }

    llm = get_llm()
    prompt = CRITIC_USER_TEMPLATE.format(
        variants_json=_variants_to_json(state.variants),
        hpo_terms=", ".join(state.hpo_terms) or "(none provided)",
        report_markdown=state.report_markdown,
    )
    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    review = _parse_critic_response(raw)

    # Append a short critic-review section to the report so users always
    # see the critique alongside the draft. Keeping both is intentional:
    # it shows the reader how the pipeline catches itself.
    addendum_lines = [
        "",
        "---",
        "",
        "## Critic review",
        f"**Verdict:** {review.verdict.replace('_', ' ')}",
        "",
        review.summary or "(no summary)",
    ]
    if review.flags:
        addendum_lines.append("")
        addendum_lines.append("### Flags")
        for f in review.flags:
            addendum_lines.append(
                f"- **[{f.severity}]** _{f.claim}_ — {f.concern} "
                f"→ _{f.suggestion}_"
            )
    report_with_critique = state.report_markdown + "\n".join(addendum_lines)

    step = trace(
        "critic_review",
        f"Critic verdict: {review.verdict}; {len(review.flags)} flag(s) raised.",
        verdict=review.verdict,
        flag_count=len(review.flags),
    )
    return {
        "critic_review": review,
        "report_markdown": report_with_critique,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }
