"use client";

function metricLabel(metric: string) {
  if (!metric) return "Score";
  return metric.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function sparklineSvg(values: number[], color: string, w = 72, h = 26) {
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values
    .map((v, i) => {
      const x = 2 + (i / Math.max(values.length - 1, 1)) * (w - 4);
      const y = h - 2 - ((v - min) / range) * (h - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="sparkline" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={pts}
      />
    </svg>
  );
}

function modelStats(rows: any[]) {
  const scores = rows.map((m) => Number(m.valid_score)).filter((v) => !Number.isNaN(v));
  const gaps = rows.map((m) => Number(m.train_valid_gap)).filter((v) => !Number.isNaN(v));
  const runtimes = rows.map((m) => Number(m.runtime_seconds)).filter((v) => !Number.isNaN(v));
  const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
  return {
    best: rows[0],
    topScore: scores[0],
    avgGap: avg(gaps),
    avgRuntime: avg(runtimes),
    scoreSeries: scores,
    gapSeries: gaps,
  };
}

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

function ModelKpiCards({ rows, metric }: { rows: any[]; metric: string }) {
  const s = modelStats(rows);
  const ml = metricLabel(metric);
  return (
    <div className="kpi-grid">
      <div className="kpi-card kpi-highlight">
        <div className="kpi-label">Best model</div>
        <div className="kpi-value kpi-sm">{modelDisplayName(s.best)}</div>
        <div className="kpi-sub">
          {ml} gap{" "}
          {s.best?.train_valid_gap != null
            ? Number(s.best.train_valid_gap).toFixed(4)
            : "—"}
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Top {ml} (valid)</div>
        <div className="kpi-row">
          <div className="kpi-value">
            {s.topScore != null ? s.topScore.toFixed(4) : "—"}
          </div>
          {sparklineSvg(s.scoreSeries, "#5b8def")}
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Avg. train–valid gap</div>
        <div className="kpi-row">
          <div className="kpi-value">
            {s.avgGap != null ? s.avgGap.toFixed(4) : "—"}
          </div>
          {sparklineSvg(s.gapSeries, "#ab63fa")}
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Models evaluated</div>
        <div className="kpi-value">{rows.length}</div>
        <div className="kpi-sub">baseline comparison set</div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Avg. runtime</div>
        <div className="kpi-value">
          {s.avgRuntime != null ? `${s.avgRuntime.toFixed(3)}s` : "—"}
        </div>
        <div className="kpi-sub">per model train + valid</div>
      </div>
    </div>
  );
}

export function ModelComparisonTable({ rows, spec }: { rows: any[]; spec?: any }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="model-panel">
        <h2>Model comparison</h2>
        <p className="muted small">No models trained yet.</p>
      </div>
    );
  }
  const metric = rows[0].primary_metric || spec?.primary_metric || "metric";
  const ml = metricLabel(metric);
  const secondary = (spec?.secondary_metrics || []).slice(0, 2);

  return (
    <div className="tab-section">
      <ModelKpiCards rows={rows} metric={metric} />
      <div className="model-panel">
        <h2>Model comparison</h2>
        <p className="lead">
          Ranked by validation <strong>{ml}</strong> (higher is better). Train–valid gap
          flags overfitting (lower is better).
          {secondary.length > 0 && <> Secondary: {secondary.join(", ")}.</>}
        </p>
        <table className="model-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th>Family</th>
              <th>{ml} (valid) ↑</th>
              <th>Train–valid gap ↓</th>
              <th>Runtime (s)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.model} className={i === 0 ? "best" : ""}>
                <td className="rank">{i + 1}</td>
                <td>
                  <span className="model-name">
                    {modelDisplayName(r)}
                    {i === 0 ? " ★" : ""}
                  </span>
                </td>
                <td>
                  <span className="family-pill">{r.family || "—"}</span>
                </td>
                <td className={`num ${i === 0 ? "score-good" : ""}`}>
                  {r.valid_score != null ? Number(r.valid_score).toFixed(4) : "—"}
                </td>
                <td className="num">
                  {r.train_valid_gap != null ? Number(r.train_valid_gap).toFixed(4) : "—"}
                </td>
                <td className="num muted">
                  {r.runtime_seconds != null ? Number(r.runtime_seconds).toFixed(3) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
