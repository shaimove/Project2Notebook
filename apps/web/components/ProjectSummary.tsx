"use client";

import { RunResponse } from "../lib/api";

export function ProjectSummary({ result }: { result: RunResponse | null }) {
  if (!result) {
    return (
      <div className="card">
        <h2>Project summary</h2>
        <p className="muted small">
          Upload a brief + CSV and run the pipeline to see the project summary.
        </p>
      </div>
    );
  }
  const spec = result.artifacts.project_spec || {};
  const final = result.artifacts.final_test_report_obj || {};
  const metric = spec.primary_metric;
  const testScore = final.test_metrics ? final.test_metrics[metric] : undefined;

  return (
    <div className="card">
      <h2>Project summary</h2>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>{spec.business_goal || "ML project"}</h3>
        <span className={`pill ${result.status === "completed" ? "ok" : "err"}`}>
          {result.status}
        </span>
      </div>
      <p className="small">{result.summary}</p>
      <table>
        <tbody>
          <Row k="Task" v={spec.ml_task_type} />
          <Row k="Target" v={(spec.targets || []).join(", ")} />
          <Row k="Primary metric" v={spec.primary_metric} />
          <Row k="Recommended split" v={spec.recommended_split} />
          <Row k="Selected model" v={final.final_model} />
          <Row
            k={`Test ${metric || "score"}`}
            v={testScore !== undefined ? Number(testScore).toFixed(4) : "—"}
          />
          <Row k="Goal met" v={final.goal_met ? "yes" : "not clearly"} />
        </tbody>
      </table>
    </div>
  );
}

function Row({ k, v }: { k: string; v: any }) {
  return (
    <tr>
      <td className="muted">{k}</td>
      <td className="mono">{v ?? "—"}</td>
    </tr>
  );
}
