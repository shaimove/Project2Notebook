"use client";

import { useState } from "react";
import {
  ProjectInfo,
  RunResponse,
  createProject,
  runPipeline,
  uploadFile,
} from "../lib/api";

interface Props {
  project: ProjectInfo | null;
  setProject: (p: ProjectInfo) => void;
  onResult: (r: RunResponse) => void;
  onLog: (msg: string) => void;
  running: boolean;
  setRunning: (v: boolean) => void;
}

export function DashboardSetup({
  project,
  setProject,
  onResult,
  onLog,
  running,
  setRunning,
}: Props) {
  const [name, setName] = useState("ML Project");
  const [taskFile, setTaskFile] = useState<File | null>(null);
  const [primaryCsv, setPrimaryCsv] = useState<File | null>(null);
  const [additionalFiles, setAdditionalFiles] = useState<File[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [enablePriorArt, setEnablePriorArt] = useState(true);
  const [maxIterations, setMaxIterations] = useState(3);
  const [minRel, setMinRel] = useState(0.05);
  const [resume, setResume] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const canStart = !!taskFile && !!primaryCsv && !running;

  async function start() {
    if (!taskFile || !primaryCsv) return;
    if (resume && !project) {
      onLog("Resume requires a previous run — uncheck resume for a fresh start.");
      return;
    }
    setRunning(true);
    onLog("Creating project and uploading files…");
    try {
      let p = project;

      if (resume && p) {
        onLog(`Resuming project ${p.project_id}`);
      } else {
        p = await createProject(name);
        setProject(p);
        onLog(`Project ${p.project_id}`);

        await uploadFile(p.project_id, "project-document", taskFile);
        onLog(`Task brief: ${taskFile.name}`);

        await uploadFile(p.project_id, "csv", primaryCsv);
        onLog(`Primary dataset: ${primaryCsv.name}`);

        for (const file of additionalFiles) {
          const lower = file.name.toLowerCase();
          if (lower.endsWith(".csv")) {
            await uploadFile(p.project_id, "csv", file);
          } else if (lower.endsWith(".pdf")) {
            await uploadFile(p.project_id, "pdf", file);
          } else {
            onLog(`Skipped unsupported file: ${file.name}`);
            continue;
          }
          onLog(`Additional data: ${file.name}`);
        }
      }

      onLog("Running pipeline…");
      const result = await runPipeline(p.project_id, {
        enablePriorArt,
        maxIterations,
        minRelImprovement: minRel,
        resume,
      });
      onResult(result);
      setCollapsed(true);
      const errCount = result.errors?.length ?? 0;
      onLog(
        errCount
          ? `Done (${result.status}) — ${errCount} step error(s)`
          : `Done (${result.status})`
      );
    } catch (e: any) {
      onLog(`Error: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  function onAdditionalChange(files: FileList | null) {
    if (!files) return;
    setAdditionalFiles(Array.from(files));
  }

  return (
    <div className={`card setup-card ${collapsed ? "setup-collapsed" : ""}`}>
      <div className="setup-header">
        <h2>Project setup</h2>
        {project && (
          <button
            type="button"
            className="btn-link small"
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? "Show inputs" : "Hide inputs"}
          </button>
        )}
      </div>

      {!collapsed && (
        <>
          <div className="setup-row">
            <div className="field">
              <label>Project name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={running}
              />
            </div>

            <div className="field">
              <label>Task brief — .md, .txt, .pdf, .doc, .docx</label>
              <input
                type="file"
                accept=".md,.txt,.pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                disabled={running}
                onChange={(e) => setTaskFile(e.target.files?.[0] ?? null)}
              />
              {taskFile && <div className="file-chip ok">✓ {taskFile.name}</div>}
            </div>

            <div className="field">
              <label>Primary dataset — one CSV</label>
              <input
                type="file"
                accept=".csv"
                disabled={running}
                onChange={(e) => setPrimaryCsv(e.target.files?.[0] ?? null)}
              />
              {primaryCsv && <div className="file-chip ok">✓ {primaryCsv.name}</div>}
            </div>

            <div className="field">
              <label>Additional data — CSV / PDF</label>
              <input
                type="file"
                accept=".csv,.pdf"
                multiple
                disabled={running}
                onChange={(e) => onAdditionalChange(e.target.files)}
              />
              {additionalFiles.map((f) => (
                <div key={f.name} className="file-chip">
                  ✓ {f.name}
                </div>
              ))}
            </div>

            <div className="setup-actions">
              <button className="btn start-btn" disabled={!canStart} onClick={start}>
                {running ? "Running…" : "Start"}
              </button>
            </div>
          </div>

          <div className="setup-meta">
            <button
              type="button"
              className="btn-link small muted"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? "Hide" : "Show"} advanced options
            </button>
            {running && (
              <span className="row small muted">
                <span className="spinner" /> Agents working…
              </span>
            )}
          </div>

          {showAdvanced && (
            <div className="setup-advanced">
              <div className="checkbox">
                <input
                  type="checkbox"
                  id="pa"
                  checked={enablePriorArt}
                  onChange={(e) => setEnablePriorArt(e.target.checked)}
                />
                <label htmlFor="pa">Enable Prior Art Agent</label>
              </div>
              <div className="field">
                <label>Max iterations</label>
                <input
                  type="number"
                  min={0}
                  max={3}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>Min relative improvement (stop rule)</label>
                <input
                  type="number"
                  step={0.01}
                  value={minRel}
                  onChange={(e) => setMinRel(Number(e.target.value))}
                />
              </div>
              <div className="checkbox">
                <input
                  type="checkbox"
                  id="resume"
                  checked={resume}
                  onChange={(e) => setResume(e.target.checked)}
                />
                <label htmlFor="resume">Resume from last checkpoint</label>
              </div>
            </div>
          )}
        </>
      )}

      {collapsed && project && (
        <div className="small muted">
          Project <span className="mono">{project.project_id}</span>
          {taskFile && ` · ${taskFile.name}`}
          {primaryCsv && ` · ${primaryCsv.name}`}
          {additionalFiles.length > 0 && ` · +${additionalFiles.length} extra file(s)`}
        </div>
      )}
    </div>
  );
}
