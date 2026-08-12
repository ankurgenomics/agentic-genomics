# Talk deck source — "Agentic AI for Genomics: Reasoning You Can Audit"

Build this directly into PowerPoint/Google Slides/Keynote. Each `##` is one slide.
Structured for a ~18–20 min talk + Q&A to a mixed technical/hiring-manager audience
in healthcare + AI/ML — trim the ⭐**optional**⭐ slides if your slot is shorter.

---

## 0. The one-sentence pitch (keep this in your head, not on a slide)

> **Most "agentic AI" demos are a chatbot with plugins. This is what agentic AI looks
> like when the domain is regulated, the stakes are clinical, and every claim has to
> be traceable back to a fact — using genomic variant interpretation as the proving
> ground.**

Three audiences, three hooks — weave all three through the talk, don't pick one:

| Audience | What they actually came to evaluate | The line that lands |
|---|---|---|
| **CEO / business** | Is this a real capability or a demo toy? Is there a market? | "Genetic testing volume is outgrowing the number of humans who can interpret it — this is a supply problem with a software answer." |
| **CTO / technical** | Is the architecture sound? Would I trust this in a pipeline? | "The LLM never touches arithmetic. Deterministic nodes compute; LLM nodes reason — and a second LLM fact-checks the first against the evidence JSON before anything ships." |
| **Hiring manager (AI/ML + healthcare)** | Can this person actually build production-grade systems, or just prompt? | "This is a typed, tested, cache-aware pipeline hitting four live public bio-databases — the engineering discipline is the same whether the output is a genomics report or a fraud score." |

---

## 1. Title slide

**Title:** Agentic AI for Genomics: Reasoning You Can Audit
**Subtitle:** Turning variant interpretation from a black box into a traceable, evidence-grounded pipeline
**Footer:** Your name · title · date · GitHub: agentic-genomics

**Speaker notes:** Open with energy, not apology. Don't lead with "this is a side project" — lead with the problem it solves.

---

## 2. The problem, stated as a business problem

**Slide headline:** *"Sequencing got cheap. Interpretation didn't."*

**Visual:** A simple two-line divergence chart (you can sketch this in PowerPoint natively — no need to pull real cost curves):
- Line 1, sloping steeply down: cost per genome (illustrative — public knowledge that WGS cost has fallen ~10,000x since 2003)
- Line 2, roughly flat: number of trained variant scientists/genetic counselors

**Bullets:**
- Every clinical genome/exome produces dozens to hundreds of candidate variants.
- Most of those variants are **VUS — Variants of Uncertain Significance** — not yes, not no.
- Resolving a VUS today means a human reading ClinVar submissions, population frequency databases, in-silico predictors, and recent literature, one variant at a time.
- That human step is the bottleneck. It doesn't scale with sequencing volume — it scales with headcount.

**Speaker notes:** This is the CEO hook. Say it plainly: this is a supply-and-demand mismatch, and it's a software-shaped problem. Don't dwell — 60-90 seconds, then move.

---

## 3. Why this is a *reasoning* problem, not a lookup problem

**Slide headline:** *"You can't SQL your way to a diagnosis."*

**Bullets — three real judgment calls an analyst makes on every case:**
- *"The patient has seizures and microcephaly — which of these 40 rare variants are even in phenotype-relevant genes?"*
- *"This missense has conflicting ClinVar entries and a moderate CADD score — escalate or downgrade?"*
- *"SpliceAI predicts a donor loss, but gnomAD shows it in three homozygotes — that's a real signal against pathogenicity."*

**Speaker notes:** Each of these requires weighing multiple sources against each other — that's exactly the shape of task LLMs are good at *when grounded in real data*, and exactly the shape of task they're dangerous at when ungrounded. That tension is the whole talk. Pause here — this is the thesis statement.

---

## 4. Introducing GenomicsCopilot

**Slide headline:** GenomicsCopilot — a variant-interpretation research copilot

**Bullets:**
- **Input:** a VCF file (the standard genomic-variant file format) + the patient's phenotype, as standardized HPO terms (Human Phenotype Ontology).
- **Output:** a ranked, *explainable* list of candidate variants — each with a full evidence chain, an ACMG-style classification, and a plain-English rationale.
- Built on **LangGraph** — a typed, multi-step agent framework — not a single mega-prompt.
- Every run against **live public data**: MyVariant.info (aggregating ClinVar, gnomAD, CADD, SpliceAI), and the JAX Human Phenotype Ontology API. No synthetic scores, no fixtures at run time.

**Speaker notes:** Position this as a *research copilot*, explicitly not a diagnostic device — that framing is both accurate and the correct regulatory posture; say it confidently, not defensively.

---

## 5. The core design principle (this is the technical spine of the talk)

**Slide headline:** *"Agents reason. Pipelines compute."*

**Visual:** Two-column split.
| Deterministic (code) | Agentic (LLM) |
|---|---|
| Parse VCF | — |
| Filter by population frequency | — |
| Fetch ClinVar / gnomAD / CADD / SpliceAI | — |
| Score phenotype match against HPO | — |
| Apply ACMG combining rules | — |
| — | Rank variants by aggregate evidence + write the narrative |
| — | Fact-check that narrative against the evidence JSON |

**Bullets:**
- Five deterministic nodes handle every step that has a *correct* answer — arithmetic, API calls, rule application.
- Two LLM nodes touch only the two steps that require *judgment*: synthesis and narrative.
- A **critic-review** node re-reads the narrative against the raw evidence and flags any claim not directly supported by it — nothing ships that the LLM "just felt."

**Speaker notes:** This is the single most important slide for the CTO in the room. The mistake most "agentic" demos make is letting the LLM do arithmetic or call tools freely with no verification layer. Here the LLM's blast radius is deliberately small, and even inside that small radius, a second model checks the first.

---

## 6. Architecture, one diagram

**Visual:** Recreate this flow as a clean left-to-right diagram (5–7 boxes, arrows). Source: `docs/architecture.md` / repo README — reuse `docs/architecture-diagram.png` directly as a starting image if you want to save time, or rebuild it in PowerPoint's SmartArt for a cleaner brand match.

```
VCF + HPO terms
   → ingest_variants
   → frequency_filter        (gnomAD / MyVariant)
   → annotate_functional     (VEP / CADD / SpliceAI)
   → clinical_lookup         (ClinVar)
   → phenotype_score         (HPO semantic similarity)
   → acmg_lite_classify      (Richards et al. 2015 combining rules)
   → second_opinion_review   (Reviewer2 — independent engine, via MCP)
   → reasoning_synthesizer   (LLM — ranks + writes narrative)
   → critic_review           (LLM — fact-checks narrative against evidence)
   → ranked report + full reasoning trace
```

**Speaker notes:** Walk left to right in one breath — don't over-explain each box, the shape is the point. Land on: "every one of these steps writes to a structured, append-only reasoning trace — so a reviewer can reconstruct *why* a variant ranked where it did, months later."

---

## 7. Now — let's look at real output, not a mockup

**Slide headline:** *"This isn't a slide. This is a live run."*

**Visual:** Full-bleed screenshot of `reports/showcase/showcase_report.html` (the chart at top, 1–2 variant cards visible). Take a fresh screenshot at presentation-friendly width (~1400px) so text is crisp when projected.

**Bullets:**
- 5 real variants, pulled live from MyVariant.info/ClinVar minutes before this run — not hand-typed fixtures.
- Spans real genes tied to real conditions: epilepsy (SCN1A), a cancer-polyposis syndrome (STK11), hereditary breast/ovarian cancer genes (TP53, PALB2, BRCA2).
- The system doesn't just say "pathogenic" to everything — it lands across the full spectrum: **Pathogenic → Uncertain → Likely Benign → Benign**, matching what's actually known about each variant.

**Speaker notes:** Let the visual breathe for a few seconds before talking over it. This is your "wow" beat — a real, dense, professional-looking artifact, not a toy chatbot answer.

---

## 8. Zoom in: one variant, full evidence chain

**Slide headline:** SCN1A — chr2:166044010 G>A

**Visual:** Pull the SCN1A card straight from the HTML report (crop it), or rebuild as a clean slide:

- **Call:** Pathogenic · ACT
- **Evidence:**
  - Stop-gain (null) variant in a gene where loss-of-function is a known disease mechanism → **PVS1**
  - Absent from gnomAD (population database of ~800K+ individuals) → **PM2_Supporting**
  - ClinVar: Pathogenic, 7 concordant submitters → **PP5**
- **Linked conditions:** Dravet syndrome, Lennox-Gastaut syndrome, Generalized epilepsy with febrile seizures plus
- **Rationale (verbatim from the system):** *"null consequence in a LoF-intolerant gene; absent/ultra-rare in gnomAD; ClinVar reports (likely) pathogenic."*

**Speaker notes:** This is the payoff of slide 5 — point at each bullet and say "this is one machine-checkable fact, not a vibe." The rationale sentence is generated, but every clause in it maps to a field in the JSON right above it — that mapping is exactly what the critic-review node verifies.

---

## 9. And it says "I don't know" when that's the honest answer

**Slide headline:** TP53 — chr17:7674944 C>T → **Uncertain Significance**

**Bullets:**
- Rare (gnomAD AF ~6.6 in a million), phenotype-relevant, computational predictors lean deleterious — **but** ClinVar itself has 8 submitters with no consensus.
- The system doesn't force a verdict. It reports **MONITOR**, not ACT — an honest "not enough evidence yet," same as a careful human reviewer would say.

**Speaker notes:** This is the trust-building slide. A system that calls everything "pathogenic" is useless (and dangerous); a system that's willing to say "uncertain" — loudly, in the same visual language as its confident calls — is the difference between a research copilot and a liability. Explicitly name the calibration prompt: *"prefer uncertain over speculation" is written directly into the synthesis instructions.*

---

## 10. Why "real live data" is a harder bar than it sounds — and why that's the point

**Slide headline:** *"Real data doesn't ask permission to be messy."*

**Bullets — frame these as *engineering rigor*, never as "bugs we found":**
- Public bio-APIs return heterogeneous shapes (a frequency field is sometimes a nested object, sometimes a flat number, depending on the record) — the pipeline normalizes both, not just the happy path.
- Genome coordinates come in two incompatible builds (GRCh37/hg19 vs GRCh38/hg38); the pipeline scopes every query explicitly rather than trusting a default.
- Some evidence fields (e.g., gene-level population constraint) are simply **absent** from one otherwise-excellent data source across its entire index — the system was built to detect that gap and pull the same fact from a second live source instead, rather than silently reporting nothing.
- Every external call is cached and typed, and every node degrades to "no data" rather than guessing when a source is unavailable.

**Speaker notes:** This is your CTO/hiring-manager slide, phrased as capability rather than confession: *"integrating with the messy reality of public bioinformatics APIs, and building a system that stays correct anyway, is real distributed-systems engineering — the same discipline whether the downstream output is a genomics report or a fraud score."* Do not narrate specific fixes or "bugs" — keep this at the level of engineering principle.

---

## 11. Engineering signal, for the room that's hiring

**Slide headline:** What's under the hood

**Bullets:**
- Fully typed Python, `mypy`-clean, `ruff`-clean.
- Test suite covering every deterministic node's edge cases — currently 59+ passing tests.
- Structured logging + an append-only reasoning trace on every run — this is an auditability requirement, not a nice-to-have, in any regulated domain.
- Zero proprietary data dependencies — every source is a public API, so the whole thing is independently reproducible by anyone in this room tonight.
- Open source, MIT-licensed, on GitHub.

**Speaker notes:** Say this fast and confidently — it's a list, not a story. The point is breadth of engineering craft (types, tests, logging, reproducibility), signaling "I build production systems," not "I write prompts."

---

## 12. Why the *agentic* framing matters, specifically

**Slide headline:** What's genuinely new here vs. a rules engine

**Bullets:**
- Rule-based interpreters (InterVar, CharGer, Franklin) have existed for years and are excellent at applying fixed criteria.
- What agentic AI adds on top:
  1. **Narrative explanation** — a human-readable evidence chain per variant, not just a label.
  2. **Phenotype-aware ranking** — weighing HPO match strength against molecular evidence the way a rule table can't.
  3. **Graceful handling of ambiguity** — ClinVar conflicts and borderline scores get thoughtful language instead of a blank cell.
  4. **Extensibility** — a new reasoning capability is a new graph node, not a new codebase.
- What's deliberately **not** agentic here: the graph is linear, not a dynamic planner, and tools are called by code, not offered to the LLM as free-form `tool_calls`. That's a design choice — deterministic orchestration beats emergent orchestration when the workflow is well understood, and it closes off an entire class of failure mode (wrong tool, wrong args, wrong time).

**Speaker notes:** This slide innoculates you against the sharpest technical question in the room: *"why not just let the LLM call the tools itself?"* Answer it before it's asked — control matters more than flexibility when the domain is clinical.

---

## 13. A second, independent agent checks the first — live, over MCP

**Slide headline:** Reasoning worth trusting is reasoning worth checking twice

**Visual:** Screenshot the SCN1A card from the live report — it has a red "flagged for human review" box; most of the other four cards show a green "concordant" box. That contrast is the whole slide.

**Bullets:**
- Every call GenomicsCopilot makes is automatically re-reviewed by **Reviewer2** — a second, separately built ACMG engine (its own gene list, its own combining-rule code, its own evidence handling) — over the **Model Context Protocol**. GenomicsCopilot's graph launches Reviewer2's MCP server as a real subprocess and calls its `review_variant_tool` for every candidate, live, on every run.
- This is a genuine second opinion, not the same code run twice: on tonight's 5-variant panel, **4 of 5 are independently concordant** — and **1 is flagged**.
- The flagged case (SCN1A) is the interesting one: GenomicsCopilot calls it Pathogenic; Reviewer2 independently calls it Uncertain significance. Why? Reviewer2's engine only applies its strongest criterion (PVS1, "null variant in a gene where loss-of-function causes disease") to a curated list of cancer-predisposition genes — and SCN1A, an epilepsy gene, isn't on that list. That's not a bug in either system — it's an honest, inspectable difference in scope between two independently engineered classifiers, surfaced automatically instead of silently picked for you.
- That's the point of a second reviewer: not to always agree, but to make disagreement visible, explained, and routed to a human — exactly how concordance review works in a real clinical lab.

**Speaker notes:** This is a genuine live capability, not a mockup — say so explicitly ("this ran minutes ago"). Walk through the SCN1A contrast concretely: two rigorous, evidence-grounded, independently-built systems looked at the exact same molecular facts and reached different conclusions, and the system tells you *why* in one sentence instead of hiding it. That's a stronger trust argument than 5/5 agreement would have been — perfect agreement would actually raise the question of whether the second opinion is really independent.

---

## 14. The roadmap — this is one agent, not the whole plan

**Slide headline:** GenomicsCopilot is agent #1

| Agent | Status | What it does |
|---|---|---|
| **GenomicsCopilot** — variant interpretation | 🟢 Live | VCF + phenotype → ranked, explainable variants |
| **NextflowAgent** | 🔵 Planned | Natural language → production bioinformatics pipelines, self-healing |
| **scRNA-Agent** | 🔵 Planned | Single-cell data + question → analysis notebook + cell-type-aware answer |
| **LitMiner** | 🔵 Planned | Gene/variant → literature synthesis → testable hypotheses |

**Speaker notes:** The point of this slide is scale of ambition without overclaiming — one working agent, a clear pattern, a real pipeline of what's next. Keep it to 30 seconds.

---

## 15. Why this matters beyond genomics

**Slide headline:** The pattern is the product

**Bullets:**
- Healthcare AI's biggest adoption blocker isn't capability — it's **trust**. Clinicians and regulators don't adopt systems they can't audit.
- The architecture on display here — deterministic core, narrow LLM reasoning surface, self-fact-checking, calibrated uncertainty — is a template for *any* high-stakes domain: finance, legal, safety-critical ops.
- "Agentic AI" is often marketed as autonomy. In regulated domains, the winning version of agentic AI is **accountable** autonomy — every step traceable, every claim checkable.

**Speaker notes:** This is your closing thesis, said directly to the CEOs/CTOs in the room: this isn't a genomics talk, it's a template for how to ship LLM reasoning into any domain where being wrong is expensive.

---

## 16. Close / the ask

**Slide headline:** Let's talk

**Bullets:**
- Everything shown tonight is open source, MIT-licensed: `github.com/<your-org>/agentic-genomics`
- Research demonstration only — not a medical device, not for clinical use (say this out loud, own it, it builds credibility rather than undermining it)
- I'm looking for [conversations about roles / collaborators / feedback — tailor to your actual goal for tomorrow]
- Contact: [your email / LinkedIn]

**Speaker notes:** End on the invitation, not a recap. Thank the organizers by name if appropriate, then open for Q&A.

---

## Appendix — anticipated Q&A (prep these, don't put on slides)

**Q: "Is this validated against real clinical labs / certified ACMG software?"**
A: No — and say so plainly. This is explicitly a research demonstration implementing a *transparent subset* of the ACMG/AMP 2015 criteria, not a certified clinical tool like InterVar, Franklin, or CardioClassifier. The goal here is to demonstrate the *architecture pattern* — evidence-grounded, auditable agentic reasoning — not to compete with certified clinical software.

**Q: "How do you know the LLM isn't hallucinating the evidence?"**
A: Two answers. First, the LLM never generates evidence — every fact in a report comes from a live API call executed by deterministic code before the LLM ever sees it. Second, the critic-review node re-reads the narrative specifically looking for claims that don't map back to that evidence JSON, and flags them.

**Q: "What's the cost / latency of a run?"**
A: [fill in actual numbers if you have them — e.g., "under N seconds per variant, two LLM calls per run regardless of how many variants are analyzed, since ranking and narrative both happen once over the full candidate set."]

**Q: "Why not just fine-tune a model on ACMG classification directly?"**
A: A fine-tuned classifier would still be a black box making a single unexplained prediction. The value here isn't the classification itself — rule-based tools already do that well — it's the auditable *evidence chain* behind it, which a fine-tuned end-to-end model can't produce without the same deterministic-plus-reasoning split shown here.

**Q: "What would it take to make this clinical-grade?"**
A: Full ACMG/AMP 28-criteria coverage (this is intentionally a lite subset), expert clinical review of every rule, a validation study against known-truth variant sets, and regulatory clearance. That's a multi-year, multi-expert effort — worth naming honestly rather than hand-waving.

**Q: "Couldn't the second opinion just be two buggy systems agreeing with each other, or disagreeing for no good reason?"**
A: Two things guard against that. First, they're independently implemented end to end — different gene lists, different combining-rule code, different codebases — so a shared bug is unlikely, and when they do disagree, the dossier states the specific, inspectable reason (e.g., "SCN1A isn't in Reviewer2's v1 gene list") rather than a bare score. Second, agreement was never the goal — the goal is that disagreement is visible and explained instead of silently averaged away or hidden. A system that can't disagree with itself isn't a second opinion, it's an echo.

---

## Visual asset checklist before you build slides

- [ ] Fresh screenshot of `reports/showcase/showcase_report.html` at ~1400px width (slide 7)
- [ ] Cropped SCN1A card from the same report — its red "flagged for human review" second-opinion box (slides 8 and 13)
- [ ] Cropped TP53 or BRCA2 card showing a green "concordant" second-opinion box, for contrast (slide 13)
- [ ] `reports/showcase/variant_ranking.png` chart, full resolution (slide 7 or its own slide)
- [ ] `docs/architecture-diagram.png` or a rebuilt version for slide 6
- [ ] Re-run `scripts/generate_showcase_report.py` the morning of the talk so the data — and the second-opinion MCP round trip — is maximally fresh; mention the run time out loud ("this ran an hour ago, live, including the second-opinion call") for extra credibility
