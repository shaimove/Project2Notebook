"""EDA MCP Server.

Each tool *generates Python plotting code and runs it through the controlled
code runner* (backend.services.code_runner). This means EDA is genuinely
executable: code is saved under ``artifacts/{project_id}/code/`` and plots under
``artifacts/{project_id}/plots/``. Only matplotlib is used (per MVP rules).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.services import code_runner
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="eda-tools",
    description="Generate and execute EDA plots/summaries via the controlled runner.",
)

_PREAMBLE = """import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)
df = pd.read_csv({csv_path!r})
"""


def _run(project_id: str, body: str, filename: str) -> Dict[str, Any]:
    res = code_runner.run_python(project_id, body, filename=filename)
    plots = [f for f in res.new_files if f.endswith(".png")]
    return {
        "ok": res.ok,
        "stdout": res.stdout[-2000:],
        "stderr": res.stderr[-2000:] if not res.ok else "",
        "plot_paths": plots,
        "code_path": res.code_path,
    }


_BASE_SCHEMA = {
    "project_id": {"type": "string", "required": True},
    "csv_path": {"type": "string", "required": True},
}


@server.tool("Plot the target distribution (histogram or bar chart).", {
    **_BASE_SCHEMA, "target": {"type": "string", "required": True}})
def create_target_distribution_plot(args: Dict[str, Any]) -> Dict[str, Any]:
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + f"""
target = {args['target']!r}
s = df[target]
plt.figure(figsize=(6,4))
if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
    plt.hist(s.dropna(), bins=30, color="#4C78A8")
    plt.ylabel("count")
else:
    vc = s.value_counts(dropna=False)
    plt.bar([str(i) for i in vc.index], vc.values, color="#4C78A8")
    plt.ylabel("count")
plt.title(f"Target distribution: {{target}}")
plt.tight_layout()
plt.savefig("plots/target_distribution.png", dpi=110)
print("saved target distribution")
"""
    return _run(args["project_id"], code, "eda_target_distribution.py")


@server.tool("Plot missingness per column (bar chart).", _BASE_SCHEMA)
def create_missingness_plot(args: Dict[str, Any]) -> Dict[str, Any]:
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + """
miss = df.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
plt.figure(figsize=(7,4))
if len(miss) == 0:
    plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
else:
    plt.barh(miss.index[::-1], miss.values[::-1], color="#E45756")
    plt.xlabel("fraction missing")
plt.title("Missing values by column")
plt.tight_layout()
plt.savefig("plots/missingness.png", dpi=110)
print(miss.to_string())
"""
    return _run(args["project_id"], code, "eda_missingness.py")


@server.tool("Plot distributions for numeric features.", {
    **_BASE_SCHEMA, "columns": {"type": "array"}})
def create_feature_distribution_plots(args: Dict[str, Any]) -> Dict[str, Any]:
    cols = args.get("columns") or []
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + f"""
cols = {json.dumps(cols)}
num = [c for c in (cols or df.columns) if pd.api.types.is_numeric_dtype(df[c])][:9]
if num:
    n = len(num)
    ncols = 3
    nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(num):
        axes[i].hist(df[c].dropna(), bins=25, color="#54A24B")
        axes[i].set_title(c, fontsize=9)
    for j in range(len(num), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=110)
    print("plotted", num)
else:
    print("no numeric columns")
"""
    return _run(args["project_id"], code, "eda_feature_distributions.py")


@server.tool("Plot relationships between features and the target.", {
    **_BASE_SCHEMA, "target": {"type": "string", "required": True}, "columns": {"type": "array"}})
def create_feature_target_plots(args: Dict[str, Any]) -> Dict[str, Any]:
    cols = args.get("columns") or []
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + f"""
target = {args['target']!r}
cols = {json.dumps(cols)}
num = [c for c in (cols or df.columns) if c != target and pd.api.types.is_numeric_dtype(df[c])][:6]
is_class = (not pd.api.types.is_numeric_dtype(df[target])) or df[target].nunique() <= 15
if num:
    ncols = 3
    nrows = (len(num)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(num):
        ax = axes[i]
        if is_class:
            groups = df.dropna(subset=[c]).groupby(target)[c]
            ax.boxplot([g.values for _, g in groups], labels=[str(k) for k,_ in groups])
            ax.set_title(f"{{c}} by {{target}}", fontsize=9)
        else:
            ax.scatter(df[c], df[target], s=8, alpha=0.4, color="#B279A2")
            ax.set_title(f"{{c}} vs {{target}}", fontsize=9)
    for j in range(len(num), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("plots/feature_target.png", dpi=110)
    print("plotted feature-target for", num)
else:
    print("no numeric features for feature-target plot")
"""
    return _run(args["project_id"], code, "eda_feature_target.py")


@server.tool("Plot time-series of the target over a time column.", {
    **_BASE_SCHEMA, "time_column": {"type": "string", "required": True},
    "target": {"type": "string", "required": True}})
def create_time_series_plots(args: Dict[str, Any]) -> Dict[str, Any]:
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + f"""
tcol = {args['time_column']!r}
target = {args['target']!r}
d = df[[tcol, target]].copy()
d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
d = d.dropna(subset=[tcol]).sort_values(tcol)
g = d.groupby(d[tcol].dt.to_period("D").dt.to_timestamp())[target].mean()
plt.figure(figsize=(8,4))
plt.plot(g.index, g.values, color="#4C78A8")
plt.title(f"{{target}} over {{tcol}}")
plt.tight_layout()
plt.savefig("plots/time_series.png", dpi=110)
print("plotted time series")
"""
    return _run(args["project_id"], code, "eda_time_series.py")


@server.tool("Compute correlation summary heatmap for numeric features.", _BASE_SCHEMA)
def create_correlation_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + """
num = df.select_dtypes(include=[np.number])
if num.shape[1] >= 2:
    corr = num.corr()
    plt.figure(figsize=(1+0.6*corr.shape[1], 1+0.6*corr.shape[1]))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title("Correlation (numeric features)")
    plt.tight_layout()
    plt.savefig("plots/correlation.png", dpi=110)
    print(corr.round(2).to_string())
else:
    print("not enough numeric columns for correlation")
"""
    return _run(args["project_id"], code, "eda_correlation.py")


@server.tool("Report outliers using the IQR rule for numeric columns.", _BASE_SCHEMA)
def create_outlier_report(args: Dict[str, Any]) -> Dict[str, Any]:
    code = _PREAMBLE.format(csv_path=args["csv_path"]) + """
report = {}
for c in df.select_dtypes(include=[np.number]).columns:
    s = df[c].dropna()
    if len(s) < 5:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    if n_out:
        report[c] = {"n_outliers": n_out, "pct": round(n_out/len(s), 4)}
print(json.dumps(report, indent=2))
"""
    res = _run(args["project_id"], code, "eda_outliers.py")
    return res


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
