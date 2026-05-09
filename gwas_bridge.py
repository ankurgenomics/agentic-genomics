#!/usr/bin/env python3
"""
gwas_bridge.py
--------------
Bridge between gwas_nf (REGENIE Nextflow GWAS pipeline) and
agentic-genomics / GenomicsCopilot (LangGraph variant interpretation pipeline).

Reads REGENIE tophit files (.regenie.filtered.gz), extracts genome-wide
significant variants (default: p < 5e-8), and formats them as a
GenomicsCopilot-compatible JSON payload for downstream LLM interpretation.

Usage
-----
# Single ethnic group, one phenotype
python gwas_bridge.py \\
    --input gwas_nf/OUTPUT/TEMUS_ETH_0/results/tophits/ \\
    --phenotype Y1 \\
    --out gwas_bridge_output/

# All 4 TEMUS ethnic groups, one phenotype (deduplicates cross-group hits)
python gwas_bridge.py \\
    --input gwas_nf/OUTPUT/ \\
    --phenotype Y1 \\
    --all-groups \\
    --out gwas_bridge_output/

# Then feed into GenomicsCopilot:
#   cd agentic-genomics
#   python run_copilot.py --from-gwas ../gwas_bridge_output/Y1_variants.json

Pipeline diagram
----------------

    gwas_nf (Nextflow + REGENIE)          agentic-genomics (LangGraph)
    ─────────────────────────────         ─────────────────────────────────────
    Genotype QC & pruning                 gwas_bridge.py  ◄── you are here
    REGENIE Step 1 (whole-genome fit)          │
    REGENIE Step 2 (association tests)    VCF ingest → MyVariant.info annotation
    Interactive Manhattan plots       ──► gnomAD / ClinVar / SpliceAI
    tophits/*.regenie.filtered.gz         HPO semantic similarity scoring
                                          ACMG-lite classification
                                          LLM synthesiser → LLM critic
                                          Interpretation report (PDF + JSON)

See: https://github.com/ankurgenomics/agentic-genomics
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

# REGENIE Step 2 output columns (space-delimited)
REGENIE_COLS = [
    "CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1",
    "A1FREQ", "N", "TEST", "BETA", "SE", "CHISQ", "LOG10P", "EXTRA",
]

# Genome-wide significance threshold: -log10(5e-8) ≈ 7.301
GW_THRESHOLD = 7.301


def parse_regenie_gz(filepath: Path, phenotype: str, group: str) -> list[dict]:
    """
    Parse a .regenie.filtered.gz tophits file.
    Returns only variants meeting the genome-wide significance threshold.
    """
    variants = []
    with gzip.open(filepath, "rt") as fh:
        header_parsed = False
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if not header_parsed:
                header_parsed = True
                continue  # skip header row
            parts = line.split()
            if len(parts) < len(REGENIE_COLS):
                continue
            record = dict(zip(REGENIE_COLS, parts))
            log10p = float(record.get("LOG10P", 0))
            if log10p < GW_THRESHOLD:
                continue
            chrom = record["CHROM"].lstrip("chr")
            pos = record["GENPOS"]
            ref = record["ALLELE0"]
            alt = record["ALLELE1"]
            variants.append({
                "variant_id": f"{chrom}:{pos}:{ref}:{alt}",
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "rsid": record.get("ID", "."),
                "beta": float(record["BETA"]),
                "se": float(record["SE"]),
                "log10p": log10p,
                "p_value": round(10 ** (-log10p), 12),
                "phenotype": phenotype,
                "ethnic_group": group,
                "source": "REGENIE_GWAS",
                "pipeline": "gwas_nf",
            })
    return variants


def find_tophit_files(
    input_dir: Path, phenotype: str, all_groups: bool
) -> list[tuple[Path, str, str]]:
    """Locate .regenie.filtered.gz files matching the given phenotype."""
    files = []
    if all_groups:
        for group_dir in sorted(input_dir.glob("TEMUS_ETH_*")):
            tophits_dir = group_dir / "results" / "tophits"
            if tophits_dir.exists():
                for f in tophits_dir.glob(f"*{phenotype}*.regenie.filtered.gz"):
                    files.append((f, phenotype, group_dir.name))
    else:
        tophits_dir = input_dir / "results" / "tophits"
        if not tophits_dir.exists():
            tophits_dir = input_dir
        for f in tophits_dir.glob(f"*{phenotype}*.regenie.filtered.gz"):
            files.append((f, phenotype, input_dir.name))
    return files


def format_for_copilot(variants: list[dict]) -> dict:
    """Wrap variant list in a GenomicsCopilot-compatible input envelope."""
    return {
        "source": "gwas_bridge",
        "pipeline_version": "1.0.0",
        "gwas_threshold": f"-log10p >= {GW_THRESHOLD} (p < 5e-8)",
        "total_variants": len(variants),
        "variants": variants,
        "interpretation_request": {
            "mode": "gwas_tophits",
            "run_acmg": True,
            "run_hpo_scoring": True,
            "run_llm_synthesis": True,
            "emit_reasoning_trace": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge gwas_nf REGENIE tophits → GenomicsCopilot interpretation."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Tophits directory, or OUTPUT/ root when using --all-groups",
    )
    parser.add_argument(
        "--phenotype", required=True,
        help="Phenotype label to extract, e.g. Y1",
    )
    parser.add_argument(
        "--all-groups", action="store_true",
        help="Scan all TEMUS_ETH_* subdirectories under --input",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for JSON payloads",
    )
    parser.add_argument(
        "--threshold", type=float, default=GW_THRESHOLD,
        help=f"Minimum -log10p threshold (default: {GW_THRESHOLD})",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    files = find_tophit_files(args.input, args.phenotype, args.all_groups)
    if not files:
        print(
            f"No tophit files found for phenotype '{args.phenotype}' in {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_variants: list[dict] = []
    for filepath, phenotype, group in files:
        variants = parse_regenie_gz(filepath, phenotype, group)
        print(f"  {group} / {phenotype}: {len(variants)} genome-wide significant variants")
        all_variants.extend(variants)

    # Deduplicate by variant_id (same locus detected in multiple ethnic groups)
    seen: set[str] = set()
    unique_variants = [
        v for v in all_variants
        if not (v["variant_id"] in seen or seen.add(v["variant_id"]))  # type: ignore[func-returns-value]
    ]
    print(
        f"\nTotal: {len(all_variants)} hits → "
        f"{len(unique_variants)} unique variants after deduplication"
    )

    payload = format_for_copilot(unique_variants)
    out_file = args.out / f"{args.phenotype}_variants.json"
    with open(out_file, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nOutput written: {out_file}")
    print("Next step:")
    print(f"  cd agentic-genomics")
    print(f"  python run_copilot.py --from-gwas {out_file.resolve()}")


if __name__ == "__main__":
    main()
