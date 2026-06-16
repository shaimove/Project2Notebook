"use client";

export function IterationHistory({
  iterations,
  summary,
}: {
  iterations: any[];
  summary?: string;
}) {
  return (
    <div className="card">
      <h2>Iteration history</h2>
      {(!iterations || iterations.length === 0) && (
        <p className="muted small">No iterations were run.</p>
      )}
      {(iterations || []).map((it) => (
        <div className="tool" key={it.iteration}>
          <div className="top">
            <div>
              <strong>Iteration {it.iteration}</strong> · <code>{it.model_name}</code>
            </div>
            <span className={`pill ${it.accepted ? "ok" : "err"}`}>
              {it.accepted ? "accepted" : "rejected"}
            </span>
          </div>
          <div className="summary">
            <div>
              <em>Hypothesis:</em> {it.hypothesis}
            </div>
            <div>
              <em>Motivation:</em> {it.motivation}
            </div>
            <div>
              valid={it.valid_score?.toFixed?.(4) ?? it.valid_score} · prev best=
              {it.previous_best?.toFixed?.(4) ?? it.previous_best} · rel.impr=
              {it.relative_improvement}
            </div>
            <div className="muted">{it.decision_reason}</div>
          </div>
        </div>
      ))}
      {summary && (
        <details>
          <summary className="muted small" style={{ cursor: "pointer" }}>
            iteration_summary.md
          </summary>
          <pre className="md">{summary}</pre>
        </details>
      )}
    </div>
  );
}
