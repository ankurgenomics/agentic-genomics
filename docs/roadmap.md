# Roadmap

`agentic-genomics` is a growing collection of agents. This document tracks what's built, what's next, and why.

> See [`../LIMITATIONS.md`](../LIMITATIONS.md) for a full accounting of what the current implementation does NOT do. The "near-term improvements" below are derived from that limitations inventory.

## Current

### 🟢 `variant_interpreter` — GenomicsCopilot (MVP)

Research-demonstration variant interpretation. Takes a VCF + HPO terms and returns a ranked, explainable report plus a critic-reviewed fact-check of the LLM's claims. See [`architecture.md`](./architecture.md).

Status: **alpha**. Works end-to-end on small demo VCFs using public APIs (MyVariant.info + JAX HPO). Not suitable for clinical use.

#### What shipped in the v0.2 pass

- PVS1 with a real LoF-intolerance gate (gnomAD pLI ≥ 0.9 or LOEUF ≤ 0.35).
- Richards et al. 2015 (Table 5) combining rules replacing the naive "count the hits" logic.
- Phrank-style IC-weighted HPO semantic similarity replacing exact-ID overlap.
- A `critic_review` LangGraph node: second LLM pass that fact-checks the synthesiser's prose against the evidence JSON and flags unsupported claims.
- `LIMITATIONS.md` enumerating what the system does NOT do.

#### Near-term correctness work (high priority)

- [ ] `bcftools norm` (or pure-Python left-aligner) pre-step so variant representation is canonical.
- [ ] Add `PS1` / `PM5` — ClinVar-per-residue lookup + same-AA / different-AA-same-residue checks.
- [ ] Add `PM4` — in-frame indel and stop-loss detection.
- [ ] ClinGen ClinVar-promotion scheme (use review status to promote PP5/BP6 to PS/BS weight) replacing the flat PP5/BP6 triggers deprecated in 2018.
- [ ] Evaluation harness: replay a held-out ClinVar slice, report recall@k and call-accuracy confusion matrix.

#### Medium-priority scope extensions

- [ ] Trio VCF ingestion + de novo detection.
- [ ] Compound-heterozygote detection for recessive genes.
- [ ] PanelApp / OMIM gene-panel awareness — down-weight off-panel hits.
- [ ] Disease-aware AF thresholds (Whiffin et al. 2017) replacing flat 1% / 5% bins.
- [ ] GRCh37 / hg19 support — CLI option routing to the correct MyVariant assembly.
- [ ] Structural-variant handling (current pipeline is SNV/indel only).

#### Lower-priority agent-design work

- [ ] ReAct-style tool-calling variant — let the LLM decide when to re-query MyVariant with a different key or walk a phenotype term's children.
- [ ] Confidence-calibration metric — have the synthesiser emit a scalar "report confidence" and evaluate its calibration on held-out data.
- [ ] LangSmith tracing toggle in `.env` + cost-aware planning.
- [ ] Offline demo mode with a pre-baked API-response cache for no-network environments.

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
