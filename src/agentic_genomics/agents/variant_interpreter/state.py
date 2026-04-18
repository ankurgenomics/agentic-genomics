"""Typed state for the variant interpretation agent.

Every node in the LangGraph reads from and writes to this state. Using
Pydantic models (rather than a plain dict) keeps the contract between
nodes explicit and makes the agent easy to test.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ACMGCall = Literal[
    "Pathogenic",
    "Likely Pathogenic",
    "Uncertain Significance",
    "Likely Benign",
    "Benign",
]


class Variant(BaseModel):
    """A minimal variant record parsed from VCF."""

    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str | None = None
    consequence: str | None = None  # e.g. "missense_variant"
    hgvs_p: str | None = None
    hgvs_c: str | None = None
    rsid: str | None = None

    @property
    def key(self) -> str:
        """Stable key suitable for caching and MyVariant lookups."""
        return f"chr{self.chrom.lstrip('chr')}:g.{self.pos}{self.ref}>{self.alt}"


class ClinicalEvidence(BaseModel):
    """Clinical-significance signal pulled from public databases."""

    clinvar_significance: str | None = None  # e.g. "Likely pathogenic"
    clinvar_review_status: str | None = None
    clinvar_submitters: int | None = None
    clinvar_conflicts: bool = False
    source: str = "clinvar"


class FunctionalScores(BaseModel):
    """In-silico prediction scores used for ACMG PP3/BP4 and PVS1 gating."""

    cadd_phred: float | None = None
    revel: float | None = None
    spliceai_max: float | None = None
    sift: str | None = None
    polyphen: str | None = None
    # Gene-level loss-of-function intolerance (from gnomAD constraint metrics).
    # pLI near 1 and LOEUF < ~0.35 both indicate the gene is haploinsufficient,
    # which is the gating condition for ACMG PVS1.
    gnomad_pli: float | None = None
    gnomad_loeuf: float | None = None


class PopulationFrequency(BaseModel):
    """Allele frequency evidence (ACMG PM2 / BA1 / BS1)."""

    gnomad_af: float | None = None
    gnomad_af_popmax: float | None = None
    gnomad_hom: int | None = None


class PhenotypeMatch(BaseModel):
    """Whether the variant's gene matches the proband's HPO terms.

    ``score`` is a Phrank-style IC-weighted semantic similarity (higher is
    better, roughly 0–10 on typical data). ``match_strength`` is a coarse
    bucket derived from ``score`` for human-readable summaries.
    """

    matched_hpo_terms: list[str] = Field(default_factory=list)
    linked_diseases: list[str] = Field(default_factory=list)
    match_strength: Literal["strong", "moderate", "weak", "none"] = "none"
    score: float = 0.0


class ACMGAssessment(BaseModel):
    """Output of the ACMG classifier."""

    call: ACMGCall
    criteria_triggered: list[str] = Field(default_factory=list)  # e.g. ["PM2_Supporting", "PP3"]
    rationale: str = ""


class CriticFlag(BaseModel):
    """A single concern raised by the critic LLM about the synthesised report."""

    severity: Literal["info", "warn", "error"] = "warn"
    claim: str = ""  # verbatim phrase or statement from the report
    concern: str = ""  # why the critic thinks it's unsupported
    suggestion: str = ""  # how to make it defensible


class CriticReview(BaseModel):
    """Output of the critic LLM pass."""

    verdict: Literal["supported", "partially_supported", "unsupported"] = "partially_supported"
    summary: str = ""
    flags: list[CriticFlag] = Field(default_factory=list)


class AnnotatedVariant(BaseModel):
    """All evidence for a single variant, gathered from tools."""

    variant: Variant
    population: PopulationFrequency = Field(default_factory=PopulationFrequency)
    functional: FunctionalScores = Field(default_factory=FunctionalScores)
    clinical: ClinicalEvidence = Field(default_factory=ClinicalEvidence)
    phenotype: PhenotypeMatch = Field(default_factory=PhenotypeMatch)
    acmg: ACMGAssessment | None = None
    llm_rationale: str = ""
    rank: int | None = None


class VariantInterpreterState(BaseModel):
    """Full agent state, flowing through the LangGraph."""

    vcf_path: str
    hpo_terms: list[str] = Field(default_factory=list)
    max_variants: int = 50  # safety cap for the demo
    variants: list[AnnotatedVariant] = Field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    report_markdown: str = ""
    critic_review: CriticReview | None = None
    error: str | None = None
