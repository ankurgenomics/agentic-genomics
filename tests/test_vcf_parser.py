"""Tests for the VCF parser, covering CSQ extraction and variant capping."""

from __future__ import annotations

import textwrap
from pathlib import Path

from agentic_genomics.agents.variant_interpreter.tools.vcf_parser import parse_vcf


def _write_vcf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "demo.vcf"
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def test_parse_minimal_vcf(tmp_path):
    vcf = _write_vcf(
        tmp_path,
        """\
        ##fileformat=VCFv4.2
        ##contig=<ID=chr1>
        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
        chr1	100	.	A	T	50	PASS	.
        chr1	200	.	G	C	50	PASS	.
        """,
    )
    variants = parse_vcf(vcf)
    assert len(variants) == 2
    assert variants[0].chrom.endswith("1")
    assert variants[0].ref == "A" and variants[0].alt == "T"


def test_parse_respects_max_variants(tmp_path):
    body = "##fileformat=VCFv4.2\n##contig=<ID=chr1>\n"
    body += "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO\n"
    for i in range(10):
        body += f"chr1	{100 + i}	.	A	T	50	PASS	.\n"
    vcf = _write_vcf(tmp_path, body)
    assert len(parse_vcf(vcf, max_variants=3)) == 3


def test_parse_multi_allelic(tmp_path):
    vcf = _write_vcf(
        tmp_path,
        """\
        ##fileformat=VCFv4.2
        ##contig=<ID=chr1>
        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
        chr1	100	.	A	T,C	50	PASS	.
        """,
    )
    variants = parse_vcf(vcf)
    assert {v.alt for v in variants} == {"T", "C"}


def test_parse_demo_vcf_with_csq():
    """End-to-end check against the real demo VCF shipped in data/samples/.

    This guards against regressions in CSQ/VEP header parsing and verifies
    the 7 curated variants are recognised with the expected gene symbols.
    """
    from pathlib import Path

    demo_vcf = Path(__file__).resolve().parents[1] / "data" / "samples" / "proband_demo.vcf"
    variants = parse_vcf(demo_vcf)
    assert len(variants) == 7
    genes = {v.gene for v in variants}
    assert genes == {"BRCA1", "BRCA2", "CFTR", "LDLR", "SCN1A", "TP53", "DMD"}
    assert all(v.consequence for v in variants)
