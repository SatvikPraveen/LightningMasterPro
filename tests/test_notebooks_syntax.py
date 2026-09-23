# tests/test_notebooks_syntax.py
"""Every code cell in every notebook must parse, and no notebook may use APIs removed in Lightning 2.0."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))

REMOVED_APIS = [
    r"from pytorch_lightning\.loops",
    r"from lightning\.pytorch\.loops\.base",
    r"track_grad_norm\s*=\s*\d",
    r"def (training|validation|test)_epoch_end\(",
    r"auto_lr_find\s*=",
    r"auto_scale_batch_size\s*=",
    r"\bgpus\s*=",
]


def _code_cells(path: Path):
    nb = json.loads(path.read_text())
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
        yield idx, src


def _code_without_comments(src: str) -> str:
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_code_cells_parse(path):
    for idx, src in _code_cells(path):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {idx}: {exc}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_no_removed_lightning_apis(path):
    for idx, src in _code_cells(path):
        code = _code_without_comments(src)
        for pattern in REMOVED_APIS:
            if re.search(pattern, code):
                pytest.fail(f"{path.name} cell {idx} uses an API removed in Lightning 2.x: {pattern}")


def test_notebook_count():
    assert len(NOTEBOOKS) == 20
