"""Schemas describing the generated notebook structure."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

CellType = Literal["markdown", "code"]


class NotebookCell(BaseModel):
    cell_type: CellType
    source: str


class NotebookSection(BaseModel):
    title: str
    cells: List[NotebookCell] = Field(default_factory=list)


class NotebookSpec(BaseModel):
    title: str = "Project2Notebook — Final Report"
    sections: List[NotebookSection] = Field(default_factory=list)
