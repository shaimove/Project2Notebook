"use client";

export function ModelComparisonTable({ rows }: { rows: any[] }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card">
        <h2>Model comparison</h2>
        <p className="muted small">No models trained yet.</p>
      </div>
    );
  }
  const metric = rows[0].primary_metric;
  return (
    <div className="card">
      <h2>Model comparison (validation {metric})</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Family</th>
            <th>{metric}</th>
            <th>Train-Valid gap</th>
            <th>Runtime (s)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} className={i === 0 ? "best" : ""}>
              <td className="mono">
                {r.model}
                {i === 0 ? " ★" : ""}
              </td>
              <td className="muted">{r.family}</td>
              <td className="mono">
                {r.valid_score !== null && r.valid_score !== undefined
                  ? Number(r.valid_score).toFixed(4)
                  : "—"}
              </td>
              <td className="mono">
                {r.train_valid_gap !== null && r.train_valid_gap !== undefined
                  ? Number(r.train_valid_gap).toFixed(4)
                  : "—"}
              </td>
              <td className="mono">{r.runtime_seconds ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
