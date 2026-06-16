"use client";

import { useState } from "react";
import { ProjectInfo, RunResponse, runPipeline } from "../lib/api";

interface Props {
  project: ProjectInfo | null;
  onResult: (r: RunResponse) => void;
  onLog: (msg: string) => void;
}

export function RunPanel({ project, onResult, onLog }: Props) {
  const [enablePriorArt, setEnablePriorArt] = useState(true);
  const [maxIterations, setMaxIterations] = useState(3);
  const [minRel, setMinRel] = useState(0.05);
  const [running, setRunning] = useState(false);

  const canRun = !!project && project.csv_paths.length >= 0;

  async function run() {
    if (!project) return;
    setRunning(true);
    onLog("Running Project2Notebook…");
    try {
      const r = await runPipeline(project.project_id, {
        enablePriorArt,
        maxIterations,
        minRelImprovement: minRel,
      });
      onResult(r);
      const errCount = r.errors?.length ?? 0;
      onLog(
        errCount
          ? `Run ${r.status} — ${r.tool_calls.length} tool calls, ${errCount} error(s)`
          : `Run ${r.status} — ${r.tool_calls.length} tool calls`
      );
    } catch (e: any) {
      onLog(`Run error: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card">
      <h2>2 · Run</h2>
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
      <button className="btn" disabled={!canRun || running} onClick={run}>
        {running ? "Running…" : "Run Project2Notebook"}
      </button>
      {running && (
        <div className="row small muted" style={{ marginTop: 8 }}>
          <span className="spinner" /> agents working (synchronous run)…
        </div>
      )}
    </div>
  );
}
