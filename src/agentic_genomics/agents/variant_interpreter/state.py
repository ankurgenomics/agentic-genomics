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
    """In-silico prediction scores used for ACMG PP3/BP4."""

    cadd_phred: float | None = None
    revel: float | None = None
    spliceai_max: float | None = None
    sift: str | None = None
    polyphen: str | None = None


class PopulationFrequency(BaseModel):
    """Allele frequency evidence (ACMG PM2 / BA1 / BS1)."""

    gnomad_af: float | None = None
    gnomad_af_popmax: float | None = None
    gnomad_hom: int | None = None


class PhenotypeMatch(BaseModel):
    """Whether the variant's gene matches the proband's HPO terms."""

    matched_hpo_terms: list[str] = Field(default_factory=list)
    linked_diseases: list[str] = Field(default_factory=list)
    match_strength: Literal["strong", "moderate", "weak", "none"] = "none"


class ACMGAssessment(BaseModel):
    """Output of the ACMG classifier."""

    call: ACMGCall
    criteria_triggered: list[str] = Field(default_factory=list)  # e.g. ["PM2_Supporting", "PP3"]
    rationale: str = ""


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
    error: str | None = None
