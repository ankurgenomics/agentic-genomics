# Roadmap

`agentic-genomics` is a growing collection of agents. This document tracks what's built, what's next, and why.

## Current

### 🟢 `variant_interpreter` — GenomicsCopilot (MVP)

Autonomous variant interpretation. Takes a VCF + HPO terms and returns a ranked, explainable report. See [`architecture.md`](./architecture.md).

Status: **alpha**. Works end-to-end on small cohorts using public demo data.

#### Near-term improvements

- [ ] Add inheritance-pattern node (de novo / recessive / dominant) for trio VCFs.
- [ ] Include SpliceAI / AlphaMissense as first-class evidence when available.
- [ ] LangSmith tracing toggle in `.env`.
- [ ] Offline demo mode with pre-cached API responses (for no-network environments).
- [ ] Evaluation harness: replay known pathogenic variants and measure recall@k.

## Planned

### 🔵 `nextflow_agent` — NL → production pipelines

**Problem:** writing Nextflow pipelines is boilerplate-heavy and error-prone. nf-core is great but still requires deep DSL2 knowledge and config fluency.

**Agent shape:**
- Input: a natural-language task + target compute environment (local / SLURM / AWS Batch).
- Tools:
  - `list_biocontainers`, `fetch_tool_schema` — pick correct container images and CLI flags.
  - `render_nextflow_module`, `render_workflow`, `render_config` — template-driven DSL2 generation.
  - `dry_run`, `parse_log` — self-healing loop: run, read `.nextflow.log`, diagnose, patch.
- Output: a runnable Nextflow project + execution report.

**Why it's a great fit:** this is exactly the kind of multi-step, feedback-driven task agentic AI is built for. Leans on [`ankurgenomics/gwas_nf`](https://github.com/ankurgenomics/gwas_nf) as a real-world example.

### 🔵 `scrna_agent` — Autonomous single-cell analysis

**Problem:** single-cell analysis is judgment-heavy. Choosing clustering resolution, annotating cell types from marker genes, and interpreting differential expression all require expertise.

**Agent shape:**
- Input: an `.h5ad` file + a scientific question ("find rare populations", "compare treated vs control").
- Tools: Scanpy (QC, normalization, clustering), CellTypist (cell-type annotation), a marker-gene lookup tool, plotting helpers.
- Output: a Scanpy notebook with QC plots, UMAPs, annotated clusters, and a written answer to the question.

**Why it's compelling:** visual demos (UMAPs + cell-type labels) make for a strong portfolio piece.

### 🔵 `litminer` — Literature → hypotheses

**Problem:** staying current with a gene or disease's literature is intractable manually.

**Agent shape:**
- Input: gene / variant / disease.
- Tools: PubMed E-utilities, bioRxiv API, optional Europe PMC, optional local vector store over abstracts.
- Output: a structured synthesis — known mechanisms, recent findings, conflicting reports, plus 2–3 *testable ML hypotheses* with suggested datasets.

**Why it's unique:** bridges literature mining, retrieval-augmented generation, and hypothesis generation — which most RAG demos don't attempt.

## Philosophy for what gets added

Each new agent must earn its place by:

1. Solving a **real** genomics pain point the author has encountered.
2. Being **agentic for a reason** — i.e., the LLM does meaningful reasoning, not just templating.
3. Having a **compelling demo** runnable in under 5 minutes on a laptop.
4. Using **only public data** in examples.

If it doesn't meet all four, it stays a design sketch.
