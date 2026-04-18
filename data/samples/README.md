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
