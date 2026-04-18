"""Parse a VCF file into lightweight :class:`Variant` records.

We intentionally *don't* use pyVCF: pysam is more robust and handles
bgzipped VCFs, multi-allelic splits, and VEP CSQ INFO fields cleanly.

For the MVP, we keep parsing minimal:
- read SNVs and small indels
- extract gene / consequence from a VEP CSQ field if present
- cap at ``max_variants`` for demo safety
"""

from __future__ import annotations

from pathlib import Path

import pysam

from agentic_genomics.agents.variant_interpreter.state import Variant


def _parse_csq_field(csq: str, csq_header: list[str]) -> dict[str, str]:
    """Parse a single VEP CSQ INFO entry into a dict keyed by field name."""
    values = csq.split("|")
    return dict(zip(csq_header, values, strict=False))


def parse_vcf(vcf_path: str | Path, max_variants: int = 50) -> list[Variant]:
    """Read a (possibly bgzipped) VCF and return up to ``max_variants`` records."""
    path = str(vcf_path)
    vcf = pysam.VariantFile(path)

    csq_header: list[str] = []
    csq_info = vcf.header.info.get("CSQ")
    if csq_info is not None and csq_info.description:
        # VEP embeds field names in the CSQ description string, after "Format: ".
        desc = csq_info.description
        if "Format:" in desc:
            raw = desc.split("Format:", 1)[1].strip().strip('"')
            csq_header = [x.strip() for x in raw.split("|")]

    variants: list[Variant] = []
    for rec in vcf.fetch():
        if rec.alts is None:
            continue
        for alt in rec.alts:
            gene = consequence = hgvs_p = hgvs_c = None
            if csq_header and "CSQ" in rec.info:
                csq_entries = rec.info["CSQ"]
                if csq_entries:
                    first = csq_entries[0]
                    parsed = _parse_csq_field(first, csq_header)
                    gene = parsed.get("SYMBOL") or None
                    consequence = parsed.get("Consequence") or None
                    hgvs_p = parsed.get("HGVSp") or None
                    hgvs_c = parsed.get("HGVSc") or None

            variants.append(
                Variant(
                    chrom=rec.chrom,
                    pos=rec.pos,
                    ref=rec.ref or "",
                    alt=alt,
                    gene=gene,
                    consequence=consequence,
                    hgvs_p=hgvs_p,
                    hgvs_c=hgvs_c,
                    rsid=rec.id if rec.id and rec.id != "." else None,
                )
            )
            if len(variants) >= max_variants:
                return variants
    return variants
