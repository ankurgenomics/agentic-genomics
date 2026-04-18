# Why agentic AI for genomics?

> *"Bioinformatics pipelines compute. Bioinformaticians reason."*
> The opportunity of agentic AI is to automate more of the second half.

## The shape of a genomics analysis today

A typical rare-disease or tumour case goes roughly:

1. **Sequencing + primary processing** (alignment, variant calling) — fully automated, deterministic.
2. **Annotation** (VEP, ClinVar join, frequency filter) — automated, deterministic.
3. **Interpretation** — *deeply human*. The analyst:
   - Cross-references ClinVar submissions and disagreements.
   - Reads the recent literature for the gene.
   - Checks HPO phenotype-gene databases.
   - Weighs SpliceAI against population frequency against inheritance pattern.
   - Applies ACMG/AMP criteria, sometimes subjectively.
   - Writes a narrative for the case.

Step 3 is exactly the kind of work LLMs do well: multi-source evidence synthesis with an explicit chain of reasoning. But vanilla LLMs fail at it because they (a) don't have reliable access to up-to-date bio databases and (b) make arithmetic and rule-application errors.

**Agentic AI** — LLMs with tool use, planning, and typed state — is the right primitive: the agent calls deterministic bio-database tools for facts, and reasons over the aggregated evidence.

## Three principles that make this work

### 1. Use LLMs for reasoning, not arithmetic

Don't ask an LLM to "filter variants with gnomAD popmax AF < 1%". That's a filter — write a filter. Do ask an LLM: *"Given these 8 candidate variants with their full evidence, which are most consistent with the patient's phenotype and why?"*

In practice this means:
- **Deterministic nodes** for ingestion, filtering, annotation, scoring, ACMG rule application.
- **One LLM synthesis node** at the end for ranking and narrative generation.

This is simpler, cheaper, and far more auditable than a tool-calling free-for-all.

### 2. Every decision is traceable

`agentic-genomics` attaches a `reasoning_trace` to every run: an append-only list of steps, each with a node name, a one-line summary, and machine-readable evidence. This matters because:
- **Audit**: reviewers can see *why* a variant was dropped or kept.
- **Debugging**: when a report looks wrong, you can pinpoint which node went off.
- **Reproducibility**: traces + cached tool outputs = exactly-reproducible runs.

### 3. Calibrated humility

The synthesiser prompt explicitly tells the LLM: "Prefer 'uncertain' over speculation. Do not use language that implies clinical actionability." This is not window-dressing. It's the difference between a useful research copilot and a liability.

## What's genuinely new here?

Rule-based interpreters have existed for years (InterVar, CharGer, Franklin). What agentic AI adds:

1. **Narrative explanations** — a human-readable evidence chain per variant, not just a label.
2. **Phenotype-aware ranking** — the LLM can weigh HPO match strength against molecular evidence in a way rule engines don't.
3. **Handles ambiguity** — ClinVar conflicts, borderline scores, and VUS calls get thoughtful language, not a blank cell.
4. **Extensibility** — adding literature mining or cohort-level reasoning is a new node, not a new codebase.

## What's NOT agentic about this MVP (on purpose)

- The graph is linear, not a dynamic planner. Deterministic orchestration beats emergent orchestration when the workflow is well-defined.
- Tools are not exposed to the LLM as `tool_calls`. They're called by code. This eliminates a huge class of failure modes (LLM calls the wrong tool, with the wrong args, at the wrong time).

Future agents in this repo (e.g. a literature-mining agent) will use tool-calling where it genuinely earns its keep.

## Further reading

- ACMG/AMP 2015 standards: [Richards et al., *Genet Med*, 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4544753/)
- HPO: [Human Phenotype Ontology](https://hpo.jax.org/)
- LangGraph: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
