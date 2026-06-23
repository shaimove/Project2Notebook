"use client";

import { useState } from "react";

type SectionProps = {
  title: string;
  step: string;
  summary?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
};

function DecisionSection({ title, step, summary, defaultOpen = false, children }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="decision-section">
      <button type="button" className="decision-header" onClick={() => setOpen(!open)}>
        <div>
          <div className="decision-step">{step}</div>
          <div className="decision-title">{title}</div>
          {summary && <div className="decision-summary muted small">{summary}</div>}
        </div>
        <span className="decision-toggle">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="decision-body">{children}</div>}
    </div>
  );
}

function TagList({ items, empty = "None" }: { items?: string[]; empty?: string }) {
  if (!items || items.length === 0) {
    return <p className="muted small">{empty}</p>;
  }
  return (
    <div className="tag-list">
      {items.map((item) => (
        <span key={item} className="pill run">
          {item}
        </span>
      ))}
    </div>
  );
}

function IssueList({ issues }: { issues?: any[] }) {
  if (!issues?.length) return <p className="muted small">No issues recorded.</p>;
  return (
    <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
      {issues.slice(0, 12).map((issue, i) => (
        <li key={i} style={{ marginBottom: 4 }}>
          <span className={`pill ${issue.severity === "critical" ? "err" : issue.severity === "warning" ? "warn-pill" : "run"}`}>
            {issue.severity || "info"}
          </span>{" "}
          {issue.column ? (
            <span className="mono">{issue.column}</span>
          ) : null}{" "}
          {issue.description}
        </li>
      ))}
    </ul>
  );
}

export function AgentDecisionsPanel({
  sections = ["quality", "eda", "preprocessing"],
  dataQuality,
  edaFindings,
  preprocessingPlan,
  splitReport,
  splitRatios,
}: {
  sections?: ("quality" | "eda" | "preprocessing")[];
  dataQuality?: any;
  edaFindings?: any;
  preprocessingPlan?: any;
  splitReport?: any;
  splitRatios?: Record<string, string>;
}) {
  const show = (s: "quality" | "eda" | "preprocessing") => sections.includes(s);
  const hasAny =
    (show("quality") && dataQuality) ||
    (show("eda") && edaFindings) ||
    (show("preprocessing") && preprocessingPlan);
  if (!hasAny) return null;

  const wrapCard = sections.length === 3;

  const body = (
    <>
      {show("quality") && dataQuality && (
        <DecisionSection
          title="Data Quality Review"
          step="Step 3"
          summary={dataQuality.summary}
          defaultOpen
        >
          <div className="small muted" style={{ marginBottom: 8 }}>
            Target: <span className="mono">{dataQuality.target_column || "—"}</span>
            {" · "}
            Cleaned file:{" "}
            <span className="mono">{dataQuality.cleaned_csv_path?.split("/").pop() || "—"}</span>
            {" · "}
            {dataQuality.n_rows_before}×{dataQuality.n_cols_before} → {dataQuality.n_rows_after}×
            {dataQuality.n_cols_after}
          </div>
          {(dataQuality.image_columns || []).length > 0 && (
            <div className="banner warn" style={{ marginBottom: 8 }}>
              Image column(s) detected: {dataQuality.image_columns.join(", ")} — excluded from
              tabular models; requires image feature extraction or a vision model.
            </div>
          )}
          {(dataQuality.column_profiles || []).length > 0 && (
            <>
              <h3>Column Profiles</h3>
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Role</th>
                    <th>Missing</th>
                    <th>Unique</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {dataQuality.column_profiles.map((p: any) => (
                    <tr key={p.name}>
                      <td className="mono">{p.name}</td>
                      <td>{p.column_type}</td>
                      <td>
                        <span className="pill run">{p.role}</span>
                      </td>
                      <td className="mono">{(p.pct_missing * 100).toFixed(1)}%</td>
                      <td className="mono">{p.n_unique}</td>
                      <td className="small">{p.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {(dataQuality.modeling_features || []).length > 0 && (
            <>
              <h3>Selected Modeling Features</h3>
              <TagList items={dataQuality.modeling_features} />
            </>
          )}
          {(dataQuality.excluded_columns || []).length > 0 && (
            <>
              <h3>Excluded Columns</h3>
              <TagList items={dataQuality.excluded_columns} />
            </>
          )}
          {(dataQuality.feature_engineering_notes || []).length > 0 && (
            <>
              <h3>Feature Engineering Notes</h3>
              <ul className="small" style={{ paddingLeft: 18 }}>
                {dataQuality.feature_engineering_notes.map((n: string, i: number) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </>
          )}
          {(dataQuality.bad_row_examples || []).length > 0 && (
            <>
              <h3>Bad Row Examples</h3>
              {dataQuality.bad_row_examples.map((ex: any, i: number) => (
                <div key={i} className="banner" style={{ marginBottom: 8 }}>
                  <strong>{ex.issue_type}</strong>: {ex.description}
                </div>
              ))}
            </>
          )}
          {(dataQuality.column_renames || []).length > 0 && (
            <>
              <h3>Column Renames</h3>
              <ul className="small mono" style={{ paddingLeft: 18 }}>
                {dataQuality.column_renames.map((r: any) => (
                  <li key={r.original}>
                    {r.original} → {r.cleaned}
                  </li>
                ))}
              </ul>
            </>
          )}
          {(dataQuality.actions_applied || []).length > 0 && (
            <>
              <h3>Cleaning Actions Applied</h3>
              <ul className="small muted" style={{ paddingLeft: 18 }}>
                {dataQuality.actions_applied.map((a: string, i: number) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </>
          )}
          <h3>Data Issues</h3>
          <IssueList issues={dataQuality.issues} />
        </DecisionSection>
      )}

      {show("eda") && edaFindings && (
        <DecisionSection
          title="EDA Review"
          step="Step 7"
          summary={edaFindings.summary}
        >
          <h3>Important Columns</h3>
          <TagList items={edaFindings.important_columns} empty="None flagged" />
          <h3>Recommended Drops</h3>
          <TagList items={edaFindings.features_to_drop} empty="None" />
          {(edaFindings.features_to_engineer || []).length > 0 && (
            <>
              <h3>Feature Engineering Ideas</h3>
              <ul className="small" style={{ paddingLeft: 18 }}>
                {edaFindings.features_to_engineer.map((e: any, i: number) => (
                  <li key={i}>
                    <span className="mono">{e.base_column}</span> → {e.transform}
                    {e.rationale ? `: ${e.rationale}` : ""}
                  </li>
                ))}
              </ul>
            </>
          )}
          {(edaFindings.preprocessing_implications || []).length > 0 && (
            <>
              <h3>Preprocessing Implications</h3>
              <ul className="small muted" style={{ paddingLeft: 18 }}>
                {edaFindings.preprocessing_implications.map((x: string, i: number) => (
                  <li key={i}>{x}</li>
                ))}
              </ul>
            </>
          )}
        </DecisionSection>
      )}

      {show("preprocessing") && preprocessingPlan && (
        <DecisionSection
          title="Split Data & Scaling"
          step="Step 8"
          summary={
            splitReport
              ? `Strategy: ${splitReport.strategy} (${splitReport.train_rows}/${splitReport.valid_rows}/${splitReport.test_rows})`
              : undefined
          }
        >
          {splitReport && (
            <table>
              <tbody>
                <tr>
                  <td className="muted">Train / Validation / Test</td>
                  <td className="mono">
                    {splitReport.train_rows} / {splitReport.valid_rows} / {splitReport.test_rows}
                  </td>
                </tr>
                {splitRatios && (
                  <tr>
                    <td className="muted">Ratios</td>
                    <td className="mono">
                      {splitRatios.Train} {splitRatios.Validation} {splitRatios.Test}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
          <h3>Keep Columns</h3>
          <TagList items={preprocessingPlan.keep_columns} />
          <h3>Drop Columns</h3>
          <TagList items={preprocessingPlan.drop_columns} empty="None" />
          <table>
            <tbody>
              <tr>
                <td className="muted">Scaling Method</td>
                <td className="mono">{preprocessingPlan.scaling_strategy || "—"} (fit on train only)</td>
              </tr>
              <tr>
                <td className="muted">Encoding</td>
                <td className="mono">{preprocessingPlan.encoding_strategy || "—"}</td>
              </tr>
              <tr>
                <td className="muted">Missing Values</td>
                <td className="mono">{preprocessingPlan.missing_value_strategy || "—"}</td>
              </tr>
            </tbody>
          </table>
          {(preprocessingPlan.notes || []).length > 0 && (
            <>
              <h3>Notes</h3>
              <ul className="small muted" style={{ paddingLeft: 18 }}>
                {preprocessingPlan.notes.map((n: string, i: number) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </>
          )}
        </DecisionSection>
      )}
    </>
  );

  if (wrapCard) {
    return (
      <div className="card">
        <h2>Agent decisions</h2>
        <p className="muted small" style={{ marginTop: 0 }}>
          What each review agent decided before modeling — cleaning, EDA insights, and preprocessing.
        </p>
        {body}
      </div>
    );
  }

  return <div className="decision-tab-content">{body}</div>;
}
