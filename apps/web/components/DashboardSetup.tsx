"use client";

import { useId, useState } from "react";
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

function FileDocIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function SetupFolderIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function FileField({
  id,
  label,
  accept,
  multiple,
  disabled,
  buttonLabel,
  file,
  files,
  onChange,
}: {
  id: string;
  label: string;
  accept: string;
  multiple?: boolean;
  disabled?: boolean;
  buttonLabel: string;
  file?: File | null;
  files?: File[];
  onChange: (files: FileList | null) => void;
}) {
  const inline =
    multiple &&
    (files?.length
      ? files.length === 1
        ? files[0].name
        : `${files.length} files chosen`
      : "No file chosen");

  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className={`file-picker ${multiple ? "file-picker-row" : ""}`}>
        <label className="file-btn" htmlFor={id}>
          <FileDocIcon />
          {buttonLabel}
        </label>
        {multiple && (
          <span className={`file-inline ${files?.length ? "" : "muted"}`}>{inline}</span>
        )}
        <input
          id={id}
          type="file"
          className="sr-only"
          accept={accept}
          multiple={multiple}
          disabled={disabled}
          onChange={(e) => onChange(e.target.files)}
        />
      </div>
      {!multiple && file && <div className="file-ok">✓ {file.name}</div>}
    </div>
  );
}

export function DashboardSetup({
  project,
  setProject,
  onResult,
  onLog,
  running,
  setRunning,
}: Props) {
  const taskId = useId();
  const csvId = useId();
  const extraId = useId();

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
  const [statusText, setStatusText] = useState("");

  const canStart = !!taskFile && !!primaryCsv && !running;

  async function start() {
    if (!taskFile || !primaryCsv) return;
    if (resume && !project) {
      onLog("Resume requires a previous run — uncheck resume for a fresh start.");
      return;
    }
    setRunning(true);
    setStatusText("uploading…");
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
      setStatusText("running pipeline…");
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
      setStatusText(`Error: ${e.message}`);
    } finally {
      setRunning(false);
      setStatusText("");
    }
  }

  return (
    <div className={`card setup-card ${collapsed ? "setup-collapsed" : ""}`}>
      <div className="setup-header">
        <div className="setup-title">
          <span className="setup-icon" aria-hidden="true">
            <SetupFolderIcon />
          </span>
          <h2>Project setup</h2>
        </div>
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
            <div className="setup-col-name">
              <div className="field">
                <label>Project name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={running}
                />
              </div>
              <div className="setup-meta">
                <button
                  type="button"
                  className="btn-link small"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  {showAdvanced ? "Hide" : "Show"} advanced options
                </button>
              </div>
            </div>

            <FileField
              id={taskId}
              label="Task brief — .md, .txt, .pdf, .doc, .docx"
              accept=".md,.txt,.pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              buttonLabel="Choose File"
              file={taskFile}
              disabled={running}
              onChange={(files) => setTaskFile(files?.[0] ?? null)}
            />

            <FileField
              id={csvId}
              label="Primary dataset — one CSV"
              accept=".csv"
              buttonLabel="Choose File"
              file={primaryCsv}
              disabled={running}
              onChange={(files) => setPrimaryCsv(files?.[0] ?? null)}
            />

            <FileField
              id={extraId}
              label="Additional data — CSV / PDF"
              accept=".csv,.pdf"
              multiple
              buttonLabel="Choose Files"
              files={additionalFiles}
              disabled={running}
              onChange={(files) => setAdditionalFiles(files ? Array.from(files) : [])}
            />

            <div className="setup-start">
              <button className="btn btn-start start-btn" disabled={!canStart} onClick={start}>
                {running ? "Running…" : "Start"}
              </button>
              <div className="start-hint">
                {!running && !statusText && (
                  <>
                    <span className="hint-icon" title="Typical run time on demo-sized datasets">
                      ⓘ
                    </span>
                    <span>Running pipeline (~2–5 min)</span>
                  </>
                )}
                {statusText && <span>{statusText}</span>}
              </div>
            </div>
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
