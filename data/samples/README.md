# Sample demo data

## `proband_demo.vcf`

A small synthetic VCF containing seven hand-picked variants from well-studied
disease-associated genes. It is **not** a real patient — positions/alleles
are chosen to represent a mix of expected outcomes when run through the
variant interpreter:

| Gene   | Genomic coord (GRCh38) | Expected flavour                              |
| ------ | ---------------------- | --------------------------------------------- |
| BRCA1  | chr17:43093843 A>G     | Missense, likely interesting                  |
| BRCA2  | chr13:32319101 C>T     | Common in gnomAD → should be filtered or benign |
| CFTR   | chr7:117559593 ATCT>A  | F508del-like in-frame deletion (classic P)   |
| LDLR   | chr19:11113431 G>A     | Candidate missense (VUS-ish)                  |
| SCN1A  | chr2:166179712 T>C     | Missense, paired with seizure HPO terms       |
| TP53   | chr17:7676154 G>A      | Near a known hotspot                          |
| DMD    | chrX:31496350 C>T      | Intronic, low-impact — should rank low        |

**Recommended HPO terms** for demo: `HP:0001250` (Seizure), `HP:0001263` (Global developmental delay).

> ⚠️ Synthetic for demonstration purposes. Coordinates are close to real
> disease genes but no specific pathogenic variant is claimed here — the
> pipeline's job is to figure out what each variant looks like from
> public evidence. Do not use this file for any real-world analysis.

## `showcase_hboc.vcf`

Five **real** variants pulled live from MyVariant.info
(`clinvar.rcv.clinical_significance` + `dbnsfp.genename` queries — see
`scripts/generate_showcase_report.py`) rather than hand-picked coordinates. A
mixed diagnostic panel spanning epilepsy, polyposis, and hereditary cancer
genes — picked to genuinely span the ACMG tiers, not to match one phenotype:

| Gene  | Genomic coord (GRCh38) | Consequence   | Real ClinVar classification (at retrieval) |
| ----- | ----------------------- | ------------- | ------------------------------------------- |
| SCN1A | chr2:166044010 G>A      | stop_gained   | Pathogenic (7 concordant submitters)         |
| STK11 | chr19:1220650 G>T       | stop_gained   | Likely pathogenic                            |
| TP53  | chr17:7674944 C>T       | missense      | Uncertain significance                       |
| PALB2 | chr16:23622972 C>T      | missense      | Likely benign (gnomAD AF 0.017)              |
| BRCA2 | chr13:32332592 A>C      | missense      | Benign (gnomAD AF 0.238)                     |

Notes from actually building this against live data (see
`scripts/generate_showcase_report.py` and `tools/gnomad_constraint.py` for the
full story):

* Coordinates were pulled with `assembly=hg38` explicitly scoped on
  MyVariant.info's query endpoint and re-verified against each gene's real
  GRCh38 span — an *unscoped* query silently mixes in hg19-native records
  (MyVariant.info's `/v1/variant/{id}` endpoint requires `assembly=hg38` for
  hg38-native records and 404s without it, and vice versa for hg19-native
  ones — there's no universal "always/never pass this param" rule).
* MyVariant.info's `gnomad_constraint` fields (pLI / LOEUF) are unpopulated
  for every record in its current index, so PVS1 — the only combining-rule
  path to Pathogenic/Likely-Pathogenic in the v1 criteria set — could never
  fire against live MyVariant data. `tools/gnomad_constraint.py` fills that
  gap from gnomAD's own public API, gated per-gene.
* The BRCA1/CHEK2 variants in an earlier draft of this file turned out to be
  a real nonsense and a real canonical-splice variant respectively, not
  missense as first assumed from ClinVar alone — and neither BRCA1 nor CHEK2
  carries strong gnomAD constraint (pLI/LOEUF), a well-documented limitation
  of population-constraint metrics for adult-onset cancer genes. SCN1A/STK11
  (epilepsy, Peutz-Jeghers) are genuinely constrained, so PVS1 fires cleanly
  there instead.

Each call above is also independently re-reviewed by **Reviewer2**
(`../../Reviewer2`, sibling repo) — a separately implemented ACMG engine
with its own gene list and combining-rule code — over MCP
(`tools/reviewer2_client.py`, wired in as the `second_opinion_review` graph
node). Reviewer2's evidence for these 5 variants lives in its own
`eval/fixtures/evidence.json`, seeded with the same real ClinVar/gnomAD/VEP
facts verified above (Reviewer2's live evidence provider is a documented v1
stub that returns no data, so fixtures — not a second live-API round trip —
are what make the second opinion meaningful; see that repo's
`src/reviewer2/evidence.py`). On this panel, 4 of 5 calls are independently
concordant; SCN1A is flagged, because Reviewer2's v1 gene allow-list is
scoped to cancer-predisposition genes and doesn't include SCN1A (an
epilepsy gene) — a real, inspectable scope difference between the two
engines, not an error in either.

**Recommended HPO terms:** `HP:0003002` (Breast carcinoma), `HP:0100615`
(Ovarian neoplasm).

Unlike `proband_demo.vcf`, ClinVar significance may drift as new submissions
land — that's real evidence, not a bug. Re-run `scripts/generate_showcase_report.py`
to refresh against current MyVariant.info data before relying on it.
