"use client";

import { useState } from "react";
import { ProjectInfo, RunResponse } from "../lib/api";
import { ScrollBox } from "./ScrollBox";
import { Markdown } from "./Markdown";
import { AgentDecisionsPanel } from "./AgentDecisionsPanel";
import { ModelComparisonTable, modelDisplayName } from "./ModelComparisonTable";
import { PlotGallery } from "./PlotGallery";
import { NotebookPreview } from "./NotebookPreview";
import { AgentTimeline } from "./AgentTimeline";
import { ToolCallViewer } from "./ToolCallViewer";
import { PipelineErrors } from "./PipelineErrors";
import { DataAuditCard } from "./DataAuditCard";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "quality", label: "Data Quality" },
  { id: "eda", label: "EDA" },
  { id: "preprocessing", label: "Split Data" },
  { id: "models", label: "Models" },
  { id: "conclusions", label: "Conclusions" },
  { id: "mcp", label: "MCP Calls" },
  { id: "pipeline", label: "Pipeline Log" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function MetricsPanel({ spec, final, modelRows }: { spec: any; final: any; modelRows: any[] }) {
  const primary = spec.primary_metric || "—";
  const secondary = (spec.secondary_metrics || []).slice(0, 2);
  const best = modelRows[0];
  const gap = best?.train_valid_gap;
  const gapLabel =
    gap === null || gap === undefined
      ? "—"
      : Number(gap).toFixed(4);

  return (
    <div className="metrics-panel">
      <h3>Evaluation metrics</h3>
      <table>
        <tbody>
          <tr>
            <td className="muted">Primary metric</td>
            <td className="mono">{primary}</td>
          </tr>
          <tr>
            <td className="muted">Secondary metrics (max 2)</td>
            <td className="mono">{secondary.length ? secondary.join(", ") : "—"}</td>
          </tr>
          <tr>
            <td className="muted">Train–valid gap (overfit check)</td>
            <td className="mono">
              {gapLabel}
              {best && (
                <span className="muted small"> · best model: {modelDisplayName(best)}</span>
              )}
            </td>
          </tr>
          {final?.test_metrics && primary !== "—" && (
            <tr>
              <td className="muted">Test {primary}</td>
              <td className="mono">
                {final.test_metrics[primary] !== undefined
                  ? Number(final.test_metrics[primary]).toFixed(4)
                  : "—"}
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        The train–valid gap compares training vs validation on the primary metric. A large
        gap suggests overfitting; iteration stops when validation gains fall below the threshold.
      </p>
    </div>
  );
}

export function DashboardResults({
  result,
  project,
}: {
  result: RunResponse | null;
  project: ProjectInfo | null;
}) {
  const [tab, setTab] = useState<TabId>("overview");

  if (!result) {
    return (
      <div className="dashboard-results dashboard-empty">
        <div className="results-header">
          <h2>Results</h2>
          <p className="muted small" style={{ margin: 0 }}>
            Select your task brief and primary CSV, then click <strong>Start</strong>.
          </p>
        </div>
        <ScrollBox maxHeight="100%" className="dashboard-empty-scroll">
          <p className="muted">
            Results will appear here across tabs: data cleaning, EDA, preprocessing, models, and
            conclusions.
          </p>
          <Markdown
            text={
              "**Pipeline steps:**\n" +
              "1. Project Understanding → Prior Art → Data Quality Review\n" +
              "2. Data Audit → EDA → EDA Review\n" +
              "3. Preprocessing & split → baseline models → iteration\n" +
              "4. Leakage review → final test → notebook"
            }
          />
        </ScrollBox>
      </div>
    );
  }

  const artifacts = result.artifacts || {};
  const spec = artifacts.project_spec || {};
  const final = artifacts.final_test_report_obj || {};
  const modelRows = artifacts.model_comparison || [];

  return (
    <div className="dashboard-results">
      <div className="results-header">
        <div>
          <h2>Results</h2>
          <p className="muted small" style={{ margin: 0 }}>
            {spec.business_goal || "ML project"} ·{" "}
            <span className={`pill ${result.status.includes("completed") ? "ok" : "err"}`}>
              {result.status}
            </span>
          </p>
        </div>
      </div>

      <div className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.id === "mcp" && (result.tool_calls?.length ?? 0) > 0
              ? ` (${result.tool_calls!.length})`
              : ""}
          </button>
        ))}
      </div>

      <div className="tab-panel">
        {tab === "overview" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              <p className="small">{result.summary}</p>
              <MetricsPanel spec={spec} final={final} modelRows={modelRows} />
              <table style={{ marginTop: 12 }}>
                <tbody>
                  <tr>
                    <td className="muted">Task</td>
                    <td className="mono">{spec.ml_task_type}</td>
                  </tr>
                  <tr>
                    <td className="muted">Target</td>
                    <td className="mono">{(spec.targets || []).join(", ") || "—"}</td>
                  </tr>
                  <tr>
                    <td className="muted">Split strategy</td>
                    <td className="mono">{spec.recommended_split || artifacts.split_report?.strategy}</td>
                  </tr>
                  <tr>
                    <td className="muted">Selected model</td>
                    <td className="mono">{final.final_model || modelDisplayName(modelRows[0] || {}) || "—"}</td>
                  </tr>
                  <tr>
                    <td className="muted">Goal met</td>
                    <td className="mono">{final.goal_met ? "yes" : "not clearly"}</td>
                  </tr>
                </tbody>
              </table>
              <PipelineErrors result={result} />
            </div>
          </ScrollBox>
        )}

        {tab === "quality" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              {project && artifacts.quality_plotly_html && (
                <div className="card">
                  <h2>Data Quality Overview (Plotly)</h2>
                  <iframe
                    title="Data Quality Plotly"
                    className="plotly-frame"
                    src={`/api/projects/${project.project_id}/plots/${encodeURIComponent(artifacts.quality_plotly_html)}`}
                  />
                </div>
              )}
              {artifacts.data_quality_report ? (
                <AgentDecisionsPanel
                  sections={["quality"]}
                  dataQuality={artifacts.data_quality_report}
                />
              ) : (
                <p className="muted">No data quality report available.</p>
              )}
            </div>
          </ScrollBox>
        )}

        {tab === "eda" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              {project && artifacts.eda_plotly_html && (
                <div className="card">
                  <h2>EDA Feature Plots (Plotly)</h2>
                  <p className="small">
                    <a
                      href={`/api/projects/${project.project_id}/plots/${encodeURIComponent(artifacts.eda_plotly_html)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open interactive Plotly chart in new tab ↗
                    </a>
                  </p>
                  <iframe
                    title="EDA Plotly"
                    className="plotly-frame"
                    src={`/api/projects/${project.project_id}/plots/${encodeURIComponent(artifacts.eda_plotly_html)}`}
                  />
                </div>
              )}
              {(artifacts.eda_plotly_conclusions || []).length > 0 && (
                <div className="card">
                  <h2>EDA Conclusions From Plots</h2>
                  <ul className="small" style={{ paddingLeft: 18 }}>
                    {(artifacts.eda_plotly_conclusions || []).map((c: string, i: number) => (
                      <li key={i}>{c.replace(/\*\*/g, "")}</li>
                    ))}
                  </ul>
                </div>
              )}
              {artifacts.eda_findings ? (
                <AgentDecisionsPanel
                  sections={["eda"]}
                  edaFindings={artifacts.eda_findings}
                />
              ) : (
                <p className="muted">No EDA findings yet.</p>
              )}
              {project && (artifacts.plot_names || []).length > 0 && (
                <PlotGallery
                  projectId={project.project_id}
                  plotNames={artifacts.plot_names || []}
                />
              )}
            </div>
          </ScrollBox>
        )}

        {tab === "preprocessing" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              {project && artifacts.split_plotly_html && (
                <div className="card">
                  <h2>Target Distribution by Split (Plotly)</h2>
                  <p className="small">
                    <a
                      href={`/api/projects/${project.project_id}/plots/${encodeURIComponent(artifacts.split_plotly_html)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open interactive Plotly chart in new tab ↗
                    </a>
                  </p>
                  <iframe
                    title="Split Plotly"
                    className="plotly-frame"
                    src={`/api/projects/${project.project_id}/plots/${encodeURIComponent(artifacts.split_plotly_html)}`}
                  />
                </div>
              )}
              <DataAuditCard audit={artifacts.data_audit_report} />
              {artifacts.preprocessing_plan ? (
                <AgentDecisionsPanel
                  sections={["preprocessing"]}
                  preprocessingPlan={artifacts.preprocessing_plan}
                  splitReport={artifacts.split_report}
                  splitRatios={artifacts.split_ratios}
                />
              ) : (
                <p className="muted">No preprocessing plan available.</p>
              )}
            </div>
          </ScrollBox>
        )}

        {tab === "models" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              <ModelComparisonTable rows={modelRows} spec={spec} />
              {(artifacts.iteration_reports || []).length > 0 && (
                <div className="card" style={{ marginTop: 12 }}>
                  <h2>Iteration history</h2>
                  {(artifacts.iteration_reports || []).map((it: any) => (
                    <div className="tool" key={it.iteration}>
                      <div className="top">
                        <div>
                          <strong>Iteration {it.iteration}</strong> ·{" "}
                          <code>{it.model_name}</code>
                        </div>
                        <span className={`pill ${it.accepted ? "ok" : "err"}`}>
                          {it.accepted ? "accepted" : "rejected"}
                        </span>
                      </div>
                      <div className="summary small">
                        <div>{it.hypothesis}</div>
                        <div className="muted">{it.decision_reason}</div>
                      </div>
                    </div>
                  ))}
                  {artifacts.iteration_summary && (
                    <ScrollBox maxHeight={160}>
                      <pre className="md">{artifacts.iteration_summary}</pre>
                    </ScrollBox>
                  )}
                </div>
              )}
            </div>
          </ScrollBox>
        )}

        {tab === "conclusions" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              {artifacts.first_conclusion && (
                <div className="card">
                  <h2>First modeling conclusion</h2>
                  <Markdown text={artifacts.first_conclusion} />
                </div>
              )}
              {artifacts.final_test_report && (
                <div className="card">
                  <h2>Final test evaluation</h2>
                  <Markdown text={artifacts.final_test_report} />
                </div>
              )}
              {artifacts.prior_art_report?.enabled && (
                <div className="card">
                  <h2>Prior art</h2>
                  <Markdown text={artifacts.prior_art_report.summary} />
                </div>
              )}
              {project && <NotebookPreview projectId={project.project_id} />}
            </div>
          </ScrollBox>
        )}

        {tab === "mcp" && (
          <ScrollBox maxHeight="100%">
            <ToolCallViewer toolCalls={result.tool_calls || []} />
          </ScrollBox>
        )}

        {tab === "pipeline" && (
          <ScrollBox maxHeight="100%">
            <div className="tab-section">
              <AgentTimeline timeline={result.timeline || []} />
              {artifacts.working_context && (
                <div className="card">
                  <h2>Agent memory</h2>
                  <ScrollBox maxHeight={220}>
                    <Markdown text={artifacts.working_context} />
                  </ScrollBox>
                </div>
              )}
              <p className="muted small">
                Full MCP trace ({result.tool_calls?.length ?? 0} calls) — see the{" "}
                <strong>MCP Calls</strong> tab.
              </p>
            </div>
          </ScrollBox>
        )}
      </div>
    </div>
  );
}
