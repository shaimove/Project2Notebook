"use client";

import { useEffect, useState } from "react";
import { getNotebook } from "../lib/api";
import { Markdown } from "./Markdown";
import { DownloadNotebookButton } from "./DownloadNotebookButton";

export function NotebookPreview({ projectId }: { projectId: string }) {
  const [nb, setNb] = useState<any>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    getNotebook(projectId)
      .then(setNb)
      .catch((e) => setError(e.message));
  }, [projectId]);

  return (
    <div className="card">
      <h2>Notebook preview</h2>
      {error && <p className="pill err">{error}</p>}
      {nb && <DownloadNotebookButton projectId={projectId} />}
      <div style={{ marginTop: 10, maxHeight: 460, overflow: "auto" }}>
        {nb?.sections?.map((sec: any, i: number) => (
          <div key={i} style={{ marginBottom: 10 }}>
            <h3>
              {i + 1}. {sec.title}
            </h3>
            {sec.cells.map((cell: any, j: number) => (
              <div key={j} className={`nb-cell ${cell.cell_type}`}>
                <div className="label">{cell.cell_type}</div>
                {cell.cell_type === "code" ? (
                  <pre>
                    <code>{cell.source}</code>
                  </pre>
                ) : (
                  <div className="body">
                    <Markdown text={cell.source} />
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
