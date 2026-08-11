"""Tests for MyVariant.info record extraction.

Exercises the pure ``extract_*`` functions against hand-built record
shapes — no network access required.
"""

from __future__ import annotations

from agentic_genomics.agents.variant_interpreter.tools import myvariant


def test_extract_population_handles_nested_af():
    record = {
        "gnomad_genome": {
            "af": {"af": 0.0123, "af_popmax": 0.045},
            "ac": {"ac_hom": 3},
        }
    }
    pop = myvariant.extract_population(record)
    assert pop.gnomad_af == 0.0123
    assert pop.gnomad_af_popmax == 0.045
    assert pop.gnomad_hom == 3


def test_extract_population_handles_flat_af():
    """MyVariant.info has returned "af" as a bare float for some records.

    Regression test: the extractor used to assume "af" was always a
    nested dict; a flat float silently produced gnomad_af=None, which
    then incorrectly triggers PM2_Supporting ("absent from gnomAD") for a
    variant that is actually common.
    """
    record = {"gnomad_genome": {"af": 0.02}}
    pop = myvariant.extract_population(record)
    assert pop.gnomad_af == 0.02
    assert pop.gnomad_af_popmax is None


def test_extract_population_handles_missing_record():
    assert myvariant.extract_population(None).gnomad_af is None
    assert myvariant.extract_population({}).gnomad_af is None
