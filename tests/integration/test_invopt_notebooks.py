"""Integration: run both example notebooks headless.

These tests are the closest thing to a real end-to-end smoke check.
If either notebook starts emitting execution errors, calibration has
silently broken something the unit tests don't cover (figure sizing
glitches aside, code-cell exceptions are caught).

Wall time: ~2 min each. Marked ``slow`` so the inner loop stays fast;
the CI workflow runs them on every push.
"""
from __future__ import annotations

import pathlib

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize(
    "nb_name", ["full_lifecycle_NL.ipynb", "real_data_entso_e.ipynb"]
)
def test_notebook_executes_cleanly(nb_name: str) -> None:
    """Execute the notebook in a fresh kernel; fail on any cell error."""
    nb_path = EXAMPLES / nb_name
    assert nb_path.exists(), f"notebook {nb_path} not found"
    nb = nbformat.read(nb_path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3",
        allow_errors=False,
    )
    client.execute(cwd=str(EXAMPLES))
    # Belt-and-braces: also verify no cell has an error output even though
    # allow_errors=False would have raised.
    errs = [
        out for c in nb.cells if c.cell_type == "code"
        for out in c.get("outputs", []) if out.get("output_type") == "error"
    ]
    assert not errs, f"{nb_name} produced {len(errs)} error outputs"
