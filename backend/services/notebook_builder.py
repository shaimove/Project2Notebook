"""Builds a clean Jupyter notebook (.ipynb) from a NotebookSpec using nbformat."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from backend.config import settings
from backend.schemas.notebook import NotebookSpec


def build_notebook(project_id: str, spec: NotebookSpec) -> Path:
    """Render a NotebookSpec into an .ipynb file and return its path."""
    nb = new_notebook()
    cells = []

    # Title cell
    cells.append(new_markdown_cell(f"# {spec.title}\n"))

    for idx, section in enumerate(spec.sections, start=1):
        cells.append(new_markdown_cell(f"## {idx}. {section.title}\n"))
        for cell in section.cells:
            if cell.cell_type == "code":
                cells.append(new_code_cell(cell.source))
            else:
                cells.append(new_markdown_cell(cell.source))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "project2notebook": {"project_id": project_id},
    }

    out_path = settings.notebooks_dir / f"{project_id}_final_notebook.ipynb"
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return out_path
