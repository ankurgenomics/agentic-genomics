"""LangGraph nodes for the variant interpreter.

Each node is a small function ``(state) -> partial_state_update``. Nodes
are deterministic except for :func:`synthesize_report`, which calls the LLM.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_genomics.agents.variant_interpreter.prompts import (
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER_TEMPLATE,
)
from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    VariantInterpreterState,
)
from agentic_genomics.agents.variant_interpreter.tools import (
    acmg,
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


def phenotype_match(state: VariantInterpreterState) -> dict:
    """Annotate each candidate with HPO-based phenotype match evidence."""
    updated: list[AnnotatedVariant] = []
    strong = moderate = 0
    for av in state.variants:
        match = hpo.score_phenotype_match(av.variant.gene, state.hpo_terms)
        if match.match_strength == "strong":
            strong += 1
        elif match.match_strength == "moderate":
            moderate += 1
        updated.append(av.model_copy(update={"phenotype": match}))

    step = trace(
        "phenotype_match",
        f"Phenotype scoring complete — strong: {strong}, moderate: {moderate}",
        strong=strong,
        moderate=moderate,
    )
    return {
        "variants": updated,
        "reasoning_trace": state.reasoning_trace + [step.to_dict()],
    }


def acmg_classify(state: VariantInterpreterState) -> dict:
    """Assign deterministic ACMG-lite classifications."""
    updated = []
    by_call: dict[str, int] = {}
    for av in state.variants:
        assessment = acmg.classify(av)
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
                "cadd_phred": av.functional.cadd_phred,
                "revel": av.functional.revel,
                "spliceai_max": av.functional.spliceai_max,
                "clinvar": av.clinical.clinvar_significance,
                "clinvar_review_status": av.clinical.clinvar_review_status,
                "clinvar_conflicts": av.clinical.clinvar_conflicts,
                "phenotype_match_strength": av.phenotype.match_strength,
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
