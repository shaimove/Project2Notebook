"use client";

import { useState } from "react";
import { ProjectInfo, createProject, uploadFile } from "../lib/api";

interface Props {
  project: ProjectInfo | null;
  setProject: (p: ProjectInfo) => void;
  onLog: (msg: string) => void;
}

export function UploadPanel({ project, setProject, onLog }: Props) {
  const [name, setName] = useState("Churn Project");
  const [doc, setDoc] = useState<string>("");
  const [csvs, setCsvs] = useState<string[]>([]);
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  async function ensureProject(): Promise<ProjectInfo> {
    if (project) return project;
    const p = await createProject(name);
    setProject(p);
    onLog(`Created project ${p.project_id}`);
    return p;
  }

  async function handleUpload(
    kind: "project-document" | "csv" | "pdf",
    files: FileList | null
  ) {
    if (!files || files.length === 0) return;
    setBusy(true);
    try {
      const p = await ensureProject();
      for (const file of Array.from(files)) {
        await uploadFile(p.project_id, kind, file);
        if (kind === "project-document") setDoc(file.name);
        else if (kind === "csv") setCsvs((c) => [...c, file.name]);
        else setPdfs((c) => [...c, file.name]);
        onLog(`Uploaded ${kind}: ${file.name}`);
      }
    } catch (e: any) {
      onLog(`Upload error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>1 · Inputs</h2>
      <div className="field">
        <label>Project name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={!!project}
        />
      </div>

      <div className="field">
        <label>Project description (md / txt / pdf)</label>
        <input
          type="file"
          accept=".md,.txt,.pdf"
          onChange={(e) => handleUpload("project-document", e.target.files)}
        />
        {doc && <div className="name">✓ {doc}</div>}
      </div>

      <div className="field">
        <label>CSV dataset(s)</label>
        <input
          type="file"
          accept=".csv"
          multiple
          onChange={(e) => handleUpload("csv", e.target.files)}
        />
        {csvs.map((c) => (
          <div key={c} className="name">
            ✓ {c}
          </div>
        ))}
      </div>

      <div className="field">
        <label>Reference PDFs (optional)</label>
        <input
          type="file"
          accept=".pdf"
          multiple
          onChange={(e) => handleUpload("pdf", e.target.files)}
        />
        {pdfs.map((c) => (
          <div key={c} className="name">
            ✓ {c}
          </div>
        ))}
      </div>

      {project && (
        <div className="banner small">
          Project <span className="mono">{project.project_id}</span> ·{" "}
          {csvs.length} CSV · {pdfs.length} PDF
        </div>
      )}
      {busy && (
        <div className="row small muted" style={{ marginTop: 8 }}>
          <span className="spinner" /> uploading…
        </div>
      )}
    </div>
  );
}
