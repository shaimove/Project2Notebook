"use client";

// Minimal, dependency-free markdown renderer for headings, bullets, bold and
// fenced code. Good enough to display agent reports without pulling in a lib.
import React from "react";

export function Markdown({ text }: { text?: string | null }) {
  if (!text) return <p className="muted small">(not available)</p>;

  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  const inline = (s: string): React.ReactNode => {
    // bold **x** and inline `code`
    const parts = s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return parts.map((p, idx) => {
      if (p.startsWith("**") && p.endsWith("**"))
        return <strong key={idx}>{p.slice(2, -2)}</strong>;
      if (p.startsWith("`") && p.endsWith("`"))
        return (
          <code key={idx} className="mono" style={{ color: "var(--accent-2)" }}>
            {p.slice(1, -1)}
          </code>
        );
      return <span key={idx}>{p}</span>;
    });
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++;
      blocks.push(
        <pre key={key++} className="nb-cell code" style={{ margin: "6px 0" }}>
          <code>{code.join("\n")}</code>
        </pre>
      );
      continue;
    }
    if (/^#{1,6}\s/.test(line)) {
      const level = line.match(/^#+/)![0].length;
      const content = line.replace(/^#+\s/, "");
      blocks.push(
        React.createElement(
          `h${Math.min(level + 2, 6)}`,
          { key: key++, style: { margin: "8px 0 4px" } },
          inline(content)
        )
      );
      i++;
      continue;
    }
    if (/^\s*[-*]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s/, ""));
        i++;
      }
      blocks.push(
        <ul key={key++} style={{ margin: "4px 0", paddingLeft: 18 }}>
          {items.map((it, idx) => (
            <li key={idx}>{inline(it)}</li>
          ))}
        </ul>
      );
      continue;
    }
    if (line.trim() === "") {
      i++;
      continue;
    }
    blocks.push(
      <p key={key++} style={{ margin: "4px 0" }}>
        {inline(line)}
      </p>
    );
    i++;
  }

  return <div className="small">{blocks}</div>;
}
