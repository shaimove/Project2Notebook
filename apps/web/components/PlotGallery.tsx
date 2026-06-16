"use client";

import { plotUrl } from "../lib/api";

export function PlotGallery({
  projectId,
  plotNames,
}: {
  projectId: string;
  plotNames: string[];
}) {
  return (
    <div className="card">
      <h2>Generated plots ({plotNames.length})</h2>
      {plotNames.length === 0 && <p className="muted small">No plots yet.</p>}
      <div className="plot-grid">
        {plotNames.map((name) => (
          <a key={name} href={plotUrl(projectId, name)} target="_blank" rel="noreferrer">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={plotUrl(projectId, name)} alt={name} />
            <div className="muted small">{name}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
