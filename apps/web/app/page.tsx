"use client";

import { useState } from "react";
import { ProjectInfo, RunResponse, API_BASE } from "../lib/api";
import { UploadPanel } from "../components/UploadPanel";
import { RunPanel } from "../components/RunPanel";
import { ProjectSummary } from "../components/ProjectSummary";
import { AgentTimeline } from "../components/AgentTimeline";
import { ToolCallViewer } from "../components/ToolCallViewer";
import { DataAuditCard } from "../components/DataAuditCard";
import { PlotGallery } from "../components/PlotGallery";
import { ModelComparisonTable } from "../components/ModelComparisonTable";
import { IterationHistory } from "../components/IterationHistory";
import { NotebookPreview } from "../components/NotebookPreview";
import { Markdown } from "../components/Markdown";
import { PipelineErrors } from "../components/PipelineErrors";

export default function Home() {
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const log = (m: string) =>
    setLogs((l) => [`${new Date().toLocaleTimeString()} · ${m}`, ...l].slice(0, 40));

  const artifacts = result?.artifacts || {};

  return (
    <div>
      <div className="app">
        <div className="header">
          <div>
            <h1>Project2Notebook</h1>
            <div className="sub">
              Agentic ML engineer · brief + data → reproducible notebook ·{" "}
              <span className="mono">{API_BASE}</span>
            </div>
          </div>
          <div className="row">
            <span className="pill run">MCP tools</span>
            <span className="pill ok">Leakage-aware</span>
            <span className="pill run">Iteration loop</span>
          </div>
        </div>

        {/* LEFT */}
        <div className="col">
          <UploadPanel project={project} setProject={setProject} onLog={log} />
          <RunPanel project={project} onResult={setResult} onLog={log} />
          <div className="card">
            <h2>Activity log</h2>
            {logs.length === 0 && <p className="muted small">No activity yet.</p>}
            <div className="small mono">
              {logs.map((l, i) => (
                <div key={i} className="muted">
                  {l}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CENTER */}
        <div className="col">
          <ProjectSummary result={result} />
          <PipelineErrors result={result} />
          <ModelComparisonTable rows={artifacts.model_comparison || []} />
          {result && project && <NotebookPreview projectId={project.project_id} />}
          {!result && (
            <div className="card">
              <h2>How it works</h2>
              <Markdown
                text={
                  "Project2Notebook reads your **brief**, inspects the **data**, and runs a deterministic agent workflow:\n\n" +
                  "- Project Understanding → Prior Art → Data Audit → EDA Planning → **Executable EDA**\n" +
                  "- Leakage-aware Preprocessing & Split → Baseline Modeling → First Conclusion\n" +
                  "- **Iterative Improvement Loop** (max 3, stop at <5% gain) → Leakage Review\n" +
                  "- **Final Test Evaluation** (test used once) → **Notebook Author**\n\n" +
                  "Every step calls tools through an **MCP client**; all calls are logged on the right."
                }
              />
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div className="col">
          <AgentTimeline timeline={result?.timeline || []} />
          {result && <DataAuditCard audit={artifacts.data_audit_report} />}
          {result && project && (
            <PlotGallery
              projectId={project.project_id}
              plotNames={artifacts.plot_names || []}
            />
          )}
          {result && (
            <IterationHistory
              iterations={artifacts.iteration_reports || []}
              summary={artifacts.iteration_summary}
            />
          )}
          {result && (
            <div className="card">
              <h2>Final test report</h2>
              <Markdown text={artifacts.final_test_report} />
            </div>
          )}
          {result && artifacts.working_context && (
            <div className="card">
              <h2>Working context (agent memory)</h2>
              {artifacts.code_files && artifacts.code_files.length > 0 && (
                <div className="small muted" style={{ marginBottom: 8 }}>
                  Authored code:{" "}
                  {artifacts.code_files.map((f: string) => (
                    <span key={f} className="pill run" style={{ marginRight: 4 }}>
                      {f}
                    </span>
                  ))}
                </div>
              )}
              <div style={{ maxHeight: 280, overflow: "auto" }}>
                <Markdown text={artifacts.working_context} />
              </div>
            </div>
          )}
          {result && (
            <div className="card">
              <h2>Prior-art summary</h2>
              {artifacts.prior_art_report?.enabled ? (
                <Markdown text={artifacts.prior_art_report?.summary} />
              ) : (
                <p className="muted small">Prior Art Agent disabled.</p>
              )}
              {artifacts.prior_art_report?.message && (
                <div className="banner warn">{artifacts.prior_art_report.message}</div>
              )}
            </div>
          )}
          <ToolCallViewer toolCalls={result?.tool_calls || []} />
        </div>
      </div>
    </div>
  );
}
