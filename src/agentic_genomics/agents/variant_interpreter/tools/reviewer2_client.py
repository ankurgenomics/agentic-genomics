"""MCP client for Reviewer2's independent ACMG second-review server.

Reviewer2 (https://github.com/ankurgenomics/reviewer2, sibling repo) is a
*separately implemented* evidence-grounded ACMG classifier — its own gene
list, its own combining-rule code, its own evidence handling. Calling it
here over the Model Context Protocol gives GenomicsCopilot a genuine second
opinion: not a re-run of the same logic with different data, but an
independently-built engine checking the same facts.

We launch Reviewer2's MCP server (``mcp_server/server.py``) as a stdio
subprocess and call its ``review_variant_tool`` once per variant, all
within a single session for the run rather than one subprocess per variant.

This integration is optional and must never break the main pipeline: if
Reviewer2 isn't installed alongside this repo, or the MCP server fails to
start, every variant simply gets ``SecondOpinion(available=False)`` and the
run continues.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agentic_genomics.agents.variant_interpreter.state import (
    AnnotatedVariant,
    SecondOpinion,
    SecondOpinionConflict,
)
from agentic_genomics.core.logging import logger

# Reviewer2 is a sibling checkout, not a package dependency of this repo —
# it's independently built/versioned on purpose (see the docstring above).
# Overridable via env var for anyone whose checkout layout differs.
_DEFAULT_REVIEWER2_DIR = Path(__file__).resolve().parents[6] / "Reviewer2"


def _reviewer2_dir() -> Path:
    return Path(os.getenv("REVIEWER2_REPO_DIR", str(_DEFAULT_REVIEWER2_DIR)))


def _reviewer2_python(repo_dir: Path) -> str:
    override = os.getenv("REVIEWER2_PYTHON")
    if override:
        return override
    venv_python = repo_dir / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


async def get_second_opinions(
    variants: list[AnnotatedVariant],
) -> dict[str, SecondOpinion]:
    """Fetch an independent Reviewer2 dossier for each variant, keyed by variant.key.

    Opens exactly one MCP session for the whole batch. On any failure to
    connect (Reviewer2 not checked out, server won't start, etc.) logs a
    warning and returns an empty dict — callers treat a missing key the
    same as ``SecondOpinion(available=False)``.
    """
    repo_dir = _reviewer2_dir()
    if not (repo_dir / "mcp_server" / "server.py").exists():
        logger.warning(
            "Reviewer2 not found at %s (set REVIEWER2_REPO_DIR) — skipping second opinion.",
            repo_dir,
        )
        return {}

    try:
        # Imported lazily: the ``mcp`` client package is an optional extra
        # (``pip install -e ".[mcp]"``), and the rest of the pipeline must
        # import cleanly without it.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning("The 'mcp' package isn't installed — skipping second opinion.")
        return {}

    server_params = StdioServerParameters(
        command=_reviewer2_python(repo_dir),
        args=["-m", "mcp_server.server"],
        cwd=str(repo_dir),
        env={
            **os.environ,
            # Fixtures are deliberately preferred over Reviewer2's live
            # provider here: the live gnomAD/ClinVar/VEP fetchers are
            # documented v1 stubs that return no evidence at all (see
            # reviewer2/evidence.py), which would make every call look
            # like "insufficient evidence" rather than a real second
            # opinion. The fixture file carries the same real ClinVar/
            # gnomAD facts GenomicsCopilot itself already verified live
            # for these variants — see eval/fixtures/evidence.json.
            "REVIEWER2_EVIDENCE_PROVIDER": "fixtures",
        },
    )

    results: dict[str, SecondOpinion] = {}
    try:
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            for av in variants:
                results[av.variant.key] = await _review_one(session, av)
    except Exception as exc:  # noqa: BLE001 - any transport/process failure degrades gracefully
        logger.warning("Reviewer2 MCP session failed (%s) — skipping second opinion.", exc)
        return {}

    return results


async def _review_one(session, av: AnnotatedVariant) -> SecondOpinion:  # type: ignore[no-untyped-def]
    variant = av.variant
    chrom = variant.chrom.removeprefix("chr")
    proposed = av.acmg.call if av.acmg else None
    try:
        result = await session.call_tool(
            "review_variant_tool",
            arguments={
                "chrom": chrom,
                "pos": variant.pos,
                "ref": variant.ref,
                "alt": variant.alt,
                "genome": "GRCh38",
                "gene": variant.gene,
                "proposed_classification": proposed,
            },
        )
        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        dossier = json.loads(text)
        if not isinstance(dossier, dict):
            raise ValueError(f"expected a JSON object from review_variant_tool, got {type(dossier).__name__}")

        conflicts = [
            SecondOpinionConflict(
                type=c.get("type", ""),
                severity=c.get("severity", "info"),
                message=c.get("message", ""),
            )
            for c in dossier.get("conflicts", [])
        ]
        # "Materially disagrees" must mean the two engines actually landed in
        # different clinical action bands — not just that Reviewer2 raised *any*
        # major-severity flag. Reviewer2 also raises MAJOR flags for e.g. ClinVar
        # itself having conflicting submitters, which is real, useful signal but
        # is not the same claim as "the two independent engines disagree" — an
        # earlier version of this code conflated the two, which mislabelled
        # PALB2/BRCA2 (where both engines actually agree) as "flagged".
        materially_disagrees = any(
            c.type == "classification_disagreement" and c.severity in {"major", "critical"}
            for c in conflicts
        )
        return SecondOpinion(
            available=True,
            independent_classification=dossier.get("independent_classification"),
            disagreement_score=dossier.get("disagreement_score"),
            materially_disagrees=materially_disagrees,
            conflicts=conflicts,
            summary=dossier.get("summary", ""),
        )
    except Exception as exc:  # noqa: BLE001 - one bad variant shouldn't sink the batch
        # Everything that can fail for THIS variant -- the tool call, the JSON
        # parse, or a malformed (non-dict) dossier -- is caught right here, so
        # it degrades to "unavailable" for this one variant only. Letting a
        # malformed-dossier AttributeError escape uncaught (as it used to)
        # meant it propagated past this function into get_second_opinions'
        # batch-level try/except, which silently zeroed out the second
        # opinion for every variant in the run, not just the bad one.
        return SecondOpinion(available=False, error=str(exc))
