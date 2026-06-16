"use client";

import { RunResponse } from "../lib/api";

export function PipelineErrors({ result }: { result: RunResponse | null }) {
  if (!result?.errors?.length) {
    return null;
  }

  return (
    <div className="card">
      <h2>Pipeline errors</h2>
      <p className="small muted">
        {result.errors.length} step(s) failed. Later steps may have used partial
        or default data.
      </p>
      <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
        {result.errors.map((err, i) => (
          <li key={i} style={{ marginBottom: 8 }}>
            <strong>{err.step}</strong>
            <div className="mono muted">{err.error}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
