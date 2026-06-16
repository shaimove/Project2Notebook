"use client";

import { notebookDownloadUrl } from "../lib/api";

export function DownloadNotebookButton({ projectId }: { projectId: string }) {
  return (
    <a
      className="btn"
      style={{ display: "block", textAlign: "center", textDecoration: "none" }}
      href={notebookDownloadUrl(projectId)}
      target="_blank"
      rel="noreferrer"
    >
      ⬇ Download final_notebook.ipynb
    </a>
  );
}
