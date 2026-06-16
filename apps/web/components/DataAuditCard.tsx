"use client";

export function DataAuditCard({ audit }: { audit: any }) {
  if (!audit) return null;
  return (
    <div className="card">
      <h2>Data audit</h2>
      <table>
        <tbody>
          <tr>
            <td className="muted">Shape</td>
            <td className="mono">
              {audit.n_rows} rows × {audit.n_cols} cols
            </td>
          </tr>
          <tr>
            <td className="muted">Duplicates</td>
            <td className="mono">{audit.n_duplicate_rows}</td>
          </tr>
          <tr>
            <td className="muted">Class imbalance</td>
            <td className="mono">{audit.class_imbalance ?? "—"}</td>
          </tr>
          <tr>
            <td className="muted">Time columns</td>
            <td className="mono">{(audit.time_columns || []).join(", ") || "—"}</td>
          </tr>
          <tr>
            <td className="muted">Entity / ID</td>
            <td className="mono">{(audit.entity_columns || []).join(", ") || "—"}</td>
          </tr>
          <tr>
            <td className="muted">Leakage-prone</td>
            <td className="mono" style={{ color: "var(--warn)" }}>
              {(audit.leakage_prone_columns || []).join(", ") || "none"}
            </td>
          </tr>
        </tbody>
      </table>
      {(audit.notes || []).length > 0 && (
        <ul className="small muted" style={{ paddingLeft: 18 }}>
          {audit.notes.map((n: string, i: number) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
