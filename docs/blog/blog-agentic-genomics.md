# Why I Built an Open-Source AI Agent for Genomic Variant Interpretation

*And what I learned about making AI systems that show their work.*

---

Every year, sequencing labs generate millions of genetic variants that need expert review. A single whole-genome sequence produces around 4–5 million variants per patient. Of those, maybe a few dozen are clinically relevant. Finding them is like searching for a needle in a haystack — except the haystack is different for every patient, and getting it wrong can mean a missed diagnosis.

The bottleneck isn't sequencing anymore. It's interpretation.

There are roughly 5,000 board-certified clinical geneticists in the United States, and the number isn't growing fast enough to match the volume of genomic data being produced. The gap is even wider in Southeast Asia, where I'm based. This isn't a technology problem waiting for a better algorithm. It's a capacity problem waiting for better tools.

That's why I built **agentic-genomics**.

## What It Actually Does

agentic-genomics is a LangGraph agent that takes two inputs — a VCF file (the standard format for genetic variant data) and a set of HPO phenotype terms (standardised descriptions of the patient's symptoms) — and returns a ranked, explainable report of candidate variants.

The key word is **explainable**. Every step in the reasoning chain is logged, traceable, and auditable by a human expert.

Here's the pipeline:

1. **Ingest** — Parse the VCF, validate format, extract variant records
2. **Annotate** — For each variant, fetch population frequency (gnomAD), clinical significance (ClinVar), and splice impact (SpliceAI) via the MyVariant.info API
3. **Filter** — Remove common variants (allele frequency > 1% in gnomAD). Most disease-causing variants are rare.
4. **Phenotype scoring** — Score each variant's associated gene against the patient's phenotype using Phrank-style HPO semantic similarity. A variant in a gene known to cause seizures should rank higher for a patient presenting with seizures.
5. **ACMG-lite classification** — Apply a transparent subset of the ACMG/AMP variant classification framework: PVS1 (null variants in genes intolerant to loss of function), PM2 (absent from population databases), PP3 (computational evidence), and others. Combine using the Richards et al. 2015 rules — the same logic clinical geneticists use manually.
6. **LLM synthesis** — Claude reads the evidence JSON for each variant and writes a narrative report, ranking candidates by strength of evidence.
7. **Critic review** — A second Claude call fact-checks the synthesis against the raw evidence. If the synthesiser claimed "this variant is pathogenic in ClinVar" but ClinVar actually says "uncertain significance," the critic flags it.

The output is a markdown report with a full reasoning trace in JSON. A human can verify every claim.

## Why Not Just Fine-Tune a Model?

The obvious question: why build a 7-node agent pipeline when you could fine-tune a model on variant-outcome pairs?

Three reasons:

**Auditability.** In clinical genomics, you can't use a black box. When a geneticist reviews an AI-assisted report, they need to see *why* the system ranked variant X above variant Y. "The neural network said so" isn't acceptable in a domain where decisions affect patient care. A reasoning trace that shows "this variant is in a LoF-intolerant gene (pLI = 0.99), absent from gnomAD, and the patient's phenotype matches the associated disease" is something a clinician can evaluate.

**Composability.** Each node is independently testable and replaceable. If gnomAD releases v5, I swap out the annotation node. If ACMG guidelines update, I modify the classification node. The LLM is just one component — not the whole system.

**Honesty.** I shipped a `LIMITATIONS.md` with the repository that documents exactly what this system does not do. It doesn't handle structural variants. It doesn't do trio analysis. It uses a simplified ACMG rule set, not the full 28-criteria framework. Being explicit about limitations is more valuable than overpromising.

## What I Learned

**Deterministic nodes do the heavy lifting.** The LLM handles maybe 20% of the work (synthesis and critic). The other 80% is traditional bioinformatics — API calls, frequency filtering, rule engines. The agent architecture lets you put the LLM where it's actually useful (natural language reasoning) and keep everything else reproducible.

**The critic node catches real errors.** In testing, the synthesiser occasionally over-stated evidence — calling a variant "likely pathogenic" when the evidence only supported "variant of uncertain significance." The critic caught these in about 70% of cases. Not perfect, but a meaningful safety net.

**llms.txt and structured metadata matter more than you'd think.** I added a machine-readable site summary (llms.txt) and JSON-LD schemas to the project's portfolio page. Within days, AI assistants were able to accurately describe the project when asked. If you're building in the open, make your work machine-readable — not just human-readable.

## Try It

The code is MIT-licensed and available at [github.com/ankurgenomics/agentic-genomics](https://github.com/ankurgenomics/agentic-genomics).

If you work in clinical genomics and want to discuss the architecture, or if you're building agentic systems in other scientific domains, I'd genuinely like to hear from you.

Portfolio with full project details: [ankurgenomics.github.io/agentic-genomics](https://ankurgenomics.github.io/agentic-genomics/)

---

*Ankur Sharma, PhD — Senior ML & Agentic AI Engineer. Building reasoning-traceable AI for genomics. Based in Singapore.*
