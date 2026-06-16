"use client";

import { useState } from "react";
import { ToolCall } from "../lib/api";

export function ToolCallViewer({ toolCalls }: { toolCalls: ToolCall[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const byServer = new Map<string, number>();
  toolCalls.forEach((t) => byServer.set(t.server, (byServer.get(t.server) || 0) + 1));

  return (
    <div className="card">
      <h2>MCP tool calls ({toolCalls.length})</h2>
      <div className="small muted" style={{ marginBottom: 8 }}>
        {Array.from(byServer.entries()).map(([s, n]) => (
          <span key={s} className="pill run" style={{ marginRight: 4 }}>
            {s}: {n}
          </span>
        ))}
      </div>
      {toolCalls.map((t, i) => (
        <div className="tool" key={i} onClick={() => setOpen(open === i ? null : i)}>
          <div className="top">
            <div>
              <span className="muted">{t.server}</span> · <code>{t.tool}</code>
            </div>
            <div className="row">
              <span className={`pill ${t.status === "success" ? "ok" : "err"}`}>
                {t.status}
              </span>
              <span className="muted small">{t.duration_ms}ms</span>
            </div>
          </div>
          <div className="summary">{t.output_summary}</div>
          {open === i && (
            <pre className="md" style={{ marginTop: 8 }}>
              {JSON.stringify(t.input, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}
