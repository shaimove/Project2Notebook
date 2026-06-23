"use client";

export function modelDisplayName(row: { model?: string; display_name?: string | null }) {
  const slug = row?.model || "";
  const dn = row?.display_name;
  if (dn && dn !== slug) return dn;
  const labels: Record<string, string> = {
    dummy: "Majority Class Baseline",
    linear: "Logistic Regression",
    tree: "Decision Tree (max_depth=6)",
    random_forest: "Random Forest",
    boosting: "XGBoost",
    svm: "RBF SVM",
  };
  return labels[slug] || dn || slug;
}

export function ModelComparisonTable({ rows, spec }: { rows: any[]; spec?: any }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card">
        <h2>Model Comparison</h2>
        <p className="muted small">No models trained yet.</p>
      </div>
    );
  }
  const metric = rows[0].primary_metric;
  const secondary = (spec?.secondary_metrics || []).slice(0, 2);
  return (
    <div className="card">
      <h2>Model Comparison</h2>
      <p className="muted small">
        Ranked by validation <span className="mono">{metric}</span>. Train–Validation gap flags overfitting.
        {secondary.length > 0 && (
          <> Secondary metrics: {secondary.join(", ")}.</>
        )}
      </p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Family</th>
            <th>{metric}</th>
            <th>Train–Valid Gap</th>
            <th>Runtime (s)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} className={i === 0 ? "best" : ""}>
              <td>
                <div className="mono">
                  {modelDisplayName(r)}
                  {i === 0 ? " ★" : ""}
                </div>
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
