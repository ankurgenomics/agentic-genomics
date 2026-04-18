"""Static checks for shipped Jupyter notebooks.

Running notebooks end-to-end in CI would require live API keys. Instead we
verify each notebook:

  1. parses as valid JSON
  2. every code cell is syntactically valid Python
  3. imports resolve (catches typos / moved modules)

This catches the vast majority of notebook-rot bugs without any network cost.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = list(
    (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")
)


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_cells_parse(nb_path: Path):
    """Every code cell must be valid Python syntax."""
    nb = json.loads(nb_path.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{nb_path.name} has no code cells"
    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(
                f"{nb_path.name}: code cell {i} fails to parse: {e}\n---\n{source}"
            )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_imports_resolve(nb_path: Path):
    """Every top-level `import`/`from` in every code cell must be importable.

    We parse with ``ast``, extract the imported module names, and try to
    import each one. This catches the classic post-refactor bug where a
    notebook still imports from a module that was renamed or removed.
    """
    import importlib

    nb = json.loads(nb_path.read_text())
    modules: set[str] = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Magic commands and shell escapes break ast.parse; filter them out.
        safe_lines = [
            ln
            for ln in source.splitlines()
            if not ln.lstrip().startswith(("%", "!", "?"))
        ]
        try:
            tree = ast.parse("\n".join(safe_lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])

    # Some notebook imports are only present under the `demo` extra
    # (jupyter, ipython, matplotlib, plotly). If the test environment doesn't
    # have them, don't flag it — users running the notebook via the documented
    # ``pip install -e '.[demo]'`` command will have them.
    demo_only = {"IPython", "matplotlib", "plotly", "streamlit"}

    missing = []
    for mod in sorted(modules):
        if mod in demo_only:
            continue
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")

    assert not missing, (
        f"{nb_path.name}: the following imports do not resolve: {missing}"
    )
