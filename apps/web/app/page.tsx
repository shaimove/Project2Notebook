"use client";

import { useState } from "react";
import { ProjectInfo, RunResponse, API_BASE } from "../lib/api";
import { DashboardSetup } from "../components/DashboardSetup";
import { DashboardResults } from "../components/DashboardResults";
import { ScrollBox } from "../components/ScrollBox";

export default function Home() {
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  const log = (m: string) =>
    setLogs((l) => [`${new Date().toLocaleTimeString()} · ${m}`, ...l].slice(0, 50));

  return (
    <div className="dashboard-app">
      <header className="header">
        <div>
          <h1>Project2Notebook</h1>
          <div className="sub">
            Agentic ML engineer · brief + data → reproducible notebook ·{" "}
            <span className="mono">{API_BASE}</span>
          </div>
        </div>
        <div className="row">
          {running && (
            <span className="row small muted">
              <span className="spinner" /> Running…
            </span>
          )}
          <span className="pill run">14-step pipeline</span>
          <span className="pill ok">Leakage-aware</span>
        </div>
      </header>

      <DashboardSetup
        project={project}
        setProject={setProject}
        onResult={setResult}
        onLog={log}
        running={running}
        setRunning={setRunning}
      />

      <div className="dashboard-body">
        <DashboardResults result={result} project={project} />

        <aside className="dashboard-sidebar">
          <div className="card sidebar-log">
            <h2>Activity log</h2>
            <ScrollBox maxHeight="100%">
              {logs.length === 0 && <p className="muted small">No activity yet.</p>}
              <div className="small mono">
                {logs.map((l, i) => (
                  <div key={i} className="muted log-line">
                    {l}
                  </div>
                ))}
              </div>
            </ScrollBox>
          </div>
        </aside>
      </div>
    </div>
  );
}
