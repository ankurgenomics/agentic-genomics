# Limitations, known gaps, and prior art

> **TL;DR.** `agentic-genomics` is a research-demonstration scaffold for the agentic-AI pattern applied to variant interpretation. It is deliberately small, readable, and self-critiquing. It is **not** a clinical tool, **not** a complete ACMG/AMP 2015 implementation, and **not** competitive with mature academic or commercial variant interpreters on correctness or coverage. This document enumerates exactly what is missing, in the same detail a senior reviewer would ask for, so that anyone evaluating the code or a report it produces knows what to look out for.

---

## 1. Not for clinical use

- This software has **not** been evaluated or approved by any regulatory body (FDA, EMA, HSA, MHRA, TGA, CDSCO, ...). It is not a medical device, IVD, or SaMD.
- LLM outputs can be miscalibrated, miscite evidence, or misapply criteria. All outputs must be independently verified by a qualified professional before any use beyond exploration or teaching.
- No PHI or identifiable sequencing data should ever be fed through this tool or its upstream third-party APIs (MyVariant.info, the Anthropic API, the JAX HPO API). Use synthetic, de-identified, or public data only.
- See [`DISCLAIMER.md`](./DISCLAIMER.md) for the full legal framing.

---

## 2. What the ACMG classifier does *not* do

`src/agentic_genomics/agents/variant_interpreter/tools/acmg_lite.py` implements a **transparent, deliberately simplified** subset of [Richards et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4544753/). The file name — `acmg_lite` — is chosen on purpose and is surfaced in every rationale string it emits.

### Criteria currently implemented (9 of 28)

| Code | Name | Our implementation |
| --- | --- | --- |
| `PVS1` | Null variant (stop, frameshift, canonical splice) in a LoF-intolerant gene | Stop-gained / frameshift / splice-donor / splice-acceptor consequences **AND** gnomAD pLI ≥ 0.9 (or LOEUF ≤ 0.35 when available). No transcript NMD-escape rules, no last-exon check, no single-exon-gene guard — all of which the ClinGen SVI refinement requires. |
| `PM2_Supporting` | Absent / ultra-rare in population databases | gnomAD popmax AF < 1e-4 or absent |
| `BS1` | Allele frequency higher than expected for the disease | Flat popmax ≥ 1% (the Whiffin et al. 2017 disease-aware threshold is not used) |
| `BA1` | Allele frequency too common to be pathogenic | Flat popmax ≥ 5% |
| `PP3` | Multiple in-silico predictors support deleterious | Any 2+ of {CADD ≥ 20, REVEL ≥ 0.7, SpliceAI ≥ 0.5, SIFT=D, PolyPhen=D/P} |
| `BP4` | Multiple in-silico predictors support benign | Any 2+ of {CADD < 10, REVEL ≤ 0.2, SpliceAI < 0.1, SIFT=T, PolyPhen=B} |
| `PP5` | Reputable source reports as pathogenic | ClinVar significance contains "pathogenic" without conflicts |
| `BP6` | Reputable source reports as benign | ClinVar significance contains "benign" without conflicts |

### Criteria NOT implemented (19 of 28)

- **`PS1`** — same amino-acid change as an established pathogenic variant.
- **`PS2`** — de novo (confirmed) in a patient with the disease.
- **`PS3`** — well-established functional studies show a damaging effect.
- **`PS4`** — prevalence in affected individuals significantly increased.
- **`PM1`** — located in a mutational hotspot / critical functional domain.
- **`PM3`** — for recessive disorders, detected in trans with a pathogenic variant.
- **`PM4`** — protein length changes (in-frame indels, stop-loss).
- **`PM5`** — different missense at a residue where another missense is pathogenic.
- **`PM6`** — assumed de novo.
- **`PP1`** — cosegregation with disease in multiple affected family members.
- **`PP2`** — missense in a gene where missense is a common mechanism.
- **`PP4`** — patient's phenotype highly specific for a single-gene aetiology.
- **`BS2`** — observed in a healthy adult for a fully-penetrant early-onset disease.
- **`BS3`** — well-established functional studies show no damaging effect.
- **`BS4`** — lack of segregation in affected family members.
- **`BP1`** — missense in a gene where only LoF is disease-causing.
- **`BP2`** — observed in trans or in cis with a pathogenic variant (for the wrong inheritance pattern).
- **`BP3`** — in-frame indel in a repetitive region without a known function.
- **`BP5`** — variant found in a case with an alternate molecular basis for disease.
- **`BP7`** — synonymous variant with no predicted splice impact.

### Combining rules

The classifier now uses the **Richards et al. 2015 Table 5** combining rules (PVS1 + 1 Strong → Pathogenic, PVS1 + 1 Moderate → Likely Pathogenic, 2 Strong → Pathogenic, 3 Moderate → Likely Pathogenic, etc.). It does **not** use Bayesian refinements (Tavtigian et al. 2018) or the ClinGen SVI criterion-specific strength modifiers.

### What this means for the classifier's output

- A variant can be called **Uncertain Significance** when a full clinical curation would (correctly) call it Likely Pathogenic — because the evidence types we'd need (`PS3`, `PM1`, `PM5`, `PP1`) aren't checked.
- A variant can be called **Likely Pathogenic** when it actually is a VUS — because we don't apply benign criteria like `BS2` or `BP7`.
- The treatment of `PP5`/`BP6` is naive: ClinGen deprecated these in 2018 in favour of using ClinVar review status to *promote* matching entries to `PS`/`BS` weight. We don't do that.

---

## 3. What the phenotype scoring does *not* do

`src/agentic_genomics/agents/variant_interpreter/tools/hpo.py` now uses a **Phrank-style** information-content weighted semantic similarity over the HPO DAG (Jagadeesh et al. 2019). It is a meaningful improvement over the initial exact-ID-intersection version, but it is still a *demonstration-grade* implementation:

### What it does
- Walks the HPO ontology (parents) from each annotated gene's HPO term set and each patient term.
- Finds the best-matching ancestor for each patient term using **information content (IC)** from the full set of gene-phenotype annotations.
- Produces a symmetric best-match average score.

### What it does NOT do
- It does **not** use the PPI-augmented scoring from [Exomiser hiPHIVE](https://exomiser.monarchinitiative.org/) (which adds cross-organism phenotype similarity through OrthoDB + PPI networks — the single biggest reason Exomiser outperforms Phrank in benchmarks).
- It does **not** perform **disease-level** aggregation like [LIRICAL](https://github.com/TheJacksonLaboratory/LIRICAL) (likelihood-ratio per HPO term per disease) or [AMELIE](https://amelie.stanford.edu/).
- It does **not** account for **annotation propagation**: HPO annotations have a "frequency" attribute (obligate / very frequent / frequent / occasional) that proper tools use.
- It does **not** handle **negated phenotypes** (`NOT HP:...`) or age-of-onset modifiers.
- The HPO term set for a gene is fetched from [ontology.jax.org](https://ontology.jax.org/api/docs), which itself exposes a simplified view. An offline workflow using the `genes_to_phenotype.txt` file + the OBO ontology would be more complete.

### What this means
Expect the phenotype score to **usefully rank** genes that are obviously relevant to the patient's phenotype. Do **not** expect it to match Exomiser / LIRICAL on fine-grained discrimination or on rare-term-heavy phenotype profiles.

---

## 4. What the VCF handling does *not* do

- **No left-alignment or normalisation.** Real pipelines run `bcftools norm -m -both -f <reference.fa>` first so that each line describes a single, canonically-represented variant. Without this, the same variant expressed two different ways will produce two different MyVariant lookups (or miss entirely).
- **No multi-allelic splitting.** We iterate `rec.alts` but we don't reconstruct per-allele genotypes, per-allele AD, or per-allele VAF. This is fine for SNV annotation, wrong for anything requiring zygosity.
- **No FILTER-column handling.** Variants with `LowQual`, `RefCall`, or caller-specific hard-filters are passed through. A clinical pipeline would only consider `PASS`.
- **No de-duplication** of near-duplicate records (e.g., caller overlap in merged multi-sample VCFs).
- **No inheritance model.** Single-sample VCF only. No trio (de novo detection), no compound-het detection, no X-linked-recessive zygosity adjustment. A heterozygous variant in a recessive gene is scored identically to a homozygous one.
- **Assembly is assumed `hg38`.** GRCh37/hg19 VCFs will silently produce wrong MyVariant lookups. There is a single constant in `tools/myvariant.py`.
- **Gene symbols come from a VEP `CSQ` INFO field only.** If the input VCF wasn't pre-annotated with Ensembl VEP, gene symbols will be empty and the whole downstream pipeline quietly scores `gene=None`. There is no fallback to `bcftools annotate` + RefSeq.

---

## 5. What the "agent" is and is not

The graph is **linear**, not a dynamic planner. The LLM never chooses which tool to call next; it sees the aggregated evidence and produces a report. This is a deliberate design choice (auditability, reproducibility, testability) but it means the word "agentic" is doing less work here than it does in, say, [ReAct](https://arxiv.org/abs/2210.03629) or [Toolformer](https://arxiv.org/abs/2302.04761)-style systems.

We do include **one genuine agentic pattern**: a second LLM pass (`critic_review`) that cross-checks the synthesiser's prose against the evidence JSON and flags unsupported claims. This is a conservative agent loop; not a fully-autonomous one.

We do **not** include: dynamic tool selection, ReAct-style interleaved reasoning/action traces, multi-agent negotiation, self-reflection / retry loops on the tool layer, or cost-aware planning.

---

## 6. Things the pipeline straight-up does not cover

- **Structural variants, CNVs, STRs, mobile-element insertions.** SNV/indel only.
- **Mitochondrial variants.** Heteroplasmy-aware scoring is out of scope.
- **Somatic variants / tumour context.** No AMP/ASCO/CAP tier I-IV framework, no OncoKB/CIViC integration.
- **Non-coding / regulatory variants.** SpliceAI is pulled but there's no UCSC cCRE, ENCODE regulatory-element, or UTR-motif (MTR / GeneHancer) layer.
- **Gene panels / indication-aware filtering.** A real workflow would restrict to an [OMIM](https://www.omim.org/) or [PanelApp](https://panelapp.genomicsengland.co.uk/) gene list for the referral indication. We score every gene equally.
- **HGVS normalisation.** We use whatever HGVS the VEP annotation supplied; we do not call [Mutalyzer](https://mutalyzer.nl/) or [VariantValidator](https://variantvalidator.org/) to canonicalise it.
- **Evaluation harness.** There is no held-out benchmark (e.g., replay of ClinVar-labelled variants) so we cannot quote recall@k or calibration metrics.

---

## 7. Prior art worth knowing about

Anyone comparing this project to the state of the art should know the following exists:

### Rule-based / curation-first variant interpreters
- **[InterVar](https://wintervar.wglab.org/)** (Li & Wang, Am J Hum Genet 2017) — all 28 ACMG criteria, ANNOVAR-driven.
- **[CharGer](https://github.com/ding-lab/CharGer)** — cancer-focused ACMG + AMP adaptation.
- **[PathoMAN](https://pathoman.mskcc.org/)** — MSK's web tool, cancer-focused.
- **[CardioClassifier](https://www.cardioclassifier.org/)** — disease-specific ClinGen expert-panel refinements.
- **[Franklin by Genoox](https://franklin.genoox.com/)**, **[Fabric GEM](https://fabricgenomics.com/)**, **[Varsome](https://varsome.com/)** — commercial platforms with curated evidence.

### Phenotype-driven prioritisation
- **[Exomiser](https://www.sanger.ac.uk/tool/exomiser/)** (Smedley et al., Nat Protoc 2015) — the Monarch Initiative's hiPHIVE scoring is the gold standard and worth emulating.
- **[LIRICAL](https://github.com/TheJacksonLaboratory/LIRICAL)** (Robinson et al., Am J Hum Genet 2020) — likelihood-ratio framework, per-disease scores.
- **[AMELIE](https://amelie.stanford.edu/)** (Birgmeier et al., Sci Transl Med 2020) — literature-grounded phenotype-to-gene scoring.
- **[Phrank](https://bitbucket.org/bejerano/phrank)** (Jagadeesh et al., Genet Med 2019) — the IC-weighted semantic-similarity method we approximate.
- **[Phen2Gene](https://phen2gene.wglab.org/)** — fast phenotype-to-gene ranking using HPO.

### LLM / agent-based efforts in genomics
- **[GeneGPT](https://github.com/ncbi-nlp/GeneGPT)** (Jin et al., Bioinformatics 2024) — GPT-4 + NCBI E-utilities. An early demonstration that tool-calling LLMs can answer factual gene queries.
- **[VarChat](https://github.com/crs4/VarChat)** — LLM-based variant-literature retrieval.
- **[GeneAgent](https://www.nature.com/articles/s42256-025-00985-0)** (Nat Mach Intell 2025) — multi-agent literature-grounded gene-function reasoning.
- ClinGen, Invitae, and other groups have internal pilots using LLMs for *assisted curation*; most are not open-source.

### Where this project sits relative to the above

This repo is **deliberately smaller and more pedagogical** than any of the tools above. It exists to (a) show a clean agentic-AI pattern (typed state + deterministic tools + a targeted LLM step + a critic loop), and (b) serve as a reference architecture for adding genomics-specific agents. It is not trying to compete with InterVar on ACMG coverage or with Exomiser on phenotype scoring.

---

## 8. Roadmap — what would make this genuinely useful

These are listed in rough priority order. Each item is deliberately scoped to "one afternoon of focused work" to make them concrete.

### High-priority correctness work
1. **Add `bcftools norm` pre-step** (or a pure-Python left-aligner) so variant representation is canonical.
2. **Add `PS1` / `PM5`** — both require a pre-indexed ClinVar-per-residue lookup but are a small amount of code once that exists.
3. **Add `PM4`** — in-frame indels and stop-loss variants.
4. **Adopt ClinGen's ClinVar promotion scheme** (review-status star rating → promote PP5/BP6 to PS/BS weight) and deprecate the flat PP5/BP6 triggers.
5. **Evaluation harness** — replay a held-out slice of ClinVar P/LP + B/LB labels, report recall@k and call-accuracy confusion matrix.

### Medium-priority scope extensions
6. **Inheritance modelling** — trio VCF ingestion, de novo detection, compound-het detection for recessive genes.
7. **Gene panel awareness** — accept a PanelApp / OMIM gene list alongside HPO terms and down-weight off-panel hits.
8. **Disease-aware AF thresholds** (Whiffin et al. 2017) replacing the flat 1% / 5% bins.
9. **GRCh37 support** — make assembly a CLI option; route MyVariant queries accordingly.

### Lower-priority agent-design work
10. **ReAct-style tool-calling variant** — let the LLM decide when to re-query MyVariant with a different key, when to walk a phenotype term's children, etc.
11. **Confidence calibration** — have the synthesiser emit a scalar "report confidence" and evaluate its calibration on held-out data.
12. **Cost-aware planning** — LangSmith-integrated cost tracking per run.

---

## 9. Who this is for

- **Engineers and researchers** prototyping agentic-AI patterns on genomic data who want a readable reference that isn't a toy.
- **Learners** who want to see LangGraph + typed state + LLM + critic-loop wired together on a non-trivial domain.
- **Interviewers** who want a concrete artefact to have a technical conversation over.

This is not for clinical genomicists looking for a production tool. They should use the tools listed in §7.
