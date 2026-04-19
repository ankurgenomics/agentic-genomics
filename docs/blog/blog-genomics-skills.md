# 8 Genomics Tools That AI Agents Can Actually Call

*Building a skill layer between LLMs and real biomedical data.*

---

There's a pattern I keep seeing in AI demos: someone asks an LLM a genomics question, and the model confidently generates an answer that's either hallucinated or six months out of date. "BRCA1 is located on chromosome 13" (it's chromosome 17). "TP53 mutations are rare in cancer" (they're the most common). The model isn't lying — it just doesn't have access to real data.

The fix isn't a better prompt. It's giving the agent **tools that return real data from real databases**.

That's what genomics-skills is: 8 standalone Python modules that an AI agent can call to get actual, verifiable, up-to-date genomic data. No hallucination possible — the data comes from TCGA, cBioPortal, NCBI, PDB, and other primary sources.

## The Problem With "Ask the LLM"

If you ask Claude or GPT-4 "What is the expression pattern of TP53 across cancer types?", you'll get a plausible paragraph that mixes memorised facts with guesses. You have no way to verify whether the numbers are real.

If you give an agent a **tool** that queries the TCGA database and returns a dataframe of TP53 expression across 31 cancer types and 9,479 patient samples — now you have real data. The agent can describe what it sees. You can check the numbers against the source.

This is the difference between an AI that sounds right and an AI that *is* right.

## The 8 Skills

Each skill follows the same contract:

- A `SKILL.md` file that describes what the skill does, its inputs, outputs, and limitations (so an agent can decide whether to use it)
- A CLI entrypoint (`genomics-skill run <skill-name> --args`)
- Deterministic output: TSV for data, PNG/SVG for plots
- Parquet caching so repeat queries are instant

Here's what's in the library:

### 1. TCGA Pan-Cancer Expression

Queries real RNA-seq expression data for any gene across 31 cancer types. The data comes from cBioPortal's API, covering 9,479 patient samples from The Cancer Genome Atlas. Returns a TSV of expression values per cancer type and generates a box plot comparing expression distributions.

```
genomics-skill run tcga-expression --gene TP53 --mode pan-cancer
```

This isn't a summary from an LLM. It's actual RSEM-normalised expression values from real tumour samples.

### 2. Survival Analysis

Takes a gene and cancer type, splits patients into high/low expression groups, and runs Kaplan-Meier survival analysis with Cox proportional hazards regression. Returns a survival curve plot and a p-value.

```
genomics-skill run survival-analysis --gene BRCA1 --cancer BRCA
```

When a researcher asks "does high BRCA1 expression correlate with survival in breast cancer?", this skill gives them an answer grounded in real patient outcome data — not a literature summary.

### 3. GO/KEGG Enrichment

Takes a gene list and runs Gene Ontology and KEGG pathway enrichment analysis. Returns the top enriched pathways with p-values and fold enrichment scores.

### 4. PubMed Search

Queries NCBI's E-utilities API for recent publications on any gene, variant, or topic. Returns structured results (title, authors, journal, year, PMID) rather than hallucinated citations.

This is critical. LLMs are notorious for inventing paper titles and DOIs. This skill returns only real publications from the actual PubMed database.

### 5. Protein Variant Mapper

Maps a missense variant to its position on the protein, showing which functional domain it falls in. Uses MyVariant.info for annotation.

### 6. 3D Protein Structure Viewer

Retrieves protein structures from PDB or AlphaFold and generates a 3D visualisation with the variant position highlighted.

### 7. Volcano Plot

Generates publication-quality volcano plots from differential expression data, highlighting significant genes.

### 8. Variant Context

Fetches comprehensive annotation for a specific variant: population frequency, clinical significance, functional predictions, and conservation scores.

## The Routing Layer

An agent shouldn't need to know which of the 8 skills to call. That's what the LLM routing layer handles.

When a user says "show me survival data for BRCA1 in breast cancer," Claude Haiku parses the intent and maps it to `survival-analysis` with the right parameters. The routing is cheap (Haiku is fast and inexpensive) and the actual data retrieval is deterministic.

```
genomics-skill suggest "what's the expression of TP53 across cancers?"
→ Routed to: tcga-expression (gene=TP53, mode=pan-cancer)
```

## Why This Architecture Matters

The broader point isn't about genomics specifically. It's about a pattern that applies to any domain where LLMs interact with specialised data:

**Separate the reasoning from the data retrieval.** Let the LLM reason, plan, and communicate. Let deterministic tools handle data access. Cache aggressively. Make every tool's contract explicit so the agent (and the human reviewing the output) knows exactly what each tool does and doesn't do.

This is the design philosophy behind agentic-genomics (the variant interpretation agent) and genomics-skills (the tool layer). The agent reasons; the skills retrieve; the human verifies.

## Try It

Both projects are MIT-licensed:

- **genomics-skills**: [github.com/ankurgenomics/genomics-skills](https://github.com/ankurgenomics/genomics-skills)
- **agentic-genomics**: [github.com/ankurgenomics/agentic-genomics](https://github.com/ankurgenomics/agentic-genomics)

If you're building tool-augmented agents in biomedicine, drug discovery, or any scientific domain — I'd like to compare notes. The challenge of grounding LLMs in real data is the same everywhere.

---

*Ankur Sharma, PhD — Senior ML & Agentic AI Engineer. Portfolio: [ankurgenomics.github.io/agentic-genomics](https://ankurgenomics.github.io/agentic-genomics/)*
