"use client";

import { TimelineItem } from "../lib/api";

export function AgentTimeline({ timeline }: { timeline: TimelineItem[] }) {
  return (
    <div className="card">
      <h2>Agent timeline</h2>
      {timeline.length === 0 && <p className="muted small">No run yet.</p>}
      {timeline.map((t) => (
        <div
          key={t.step}
          className={`timeline-item ${t.status === "completed" ? "ok" : "err"}`}
        >
          <div className="dot">{t.status === "completed" ? t.step : "!"}</div>
          <div>
            <div className="t-title">{t.title}</div>
            {t.detail && <div className="t-detail">{t.detail}</div>}
            {t.duration_ms !== undefined && (
              <div className="t-detail">{t.duration_ms} ms</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
