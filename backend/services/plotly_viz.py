"""Plotly HTML visualizations for EDA and split diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services import artifact_store
from backend.services.csv_io import load_csv_for_analysis


def _positive_class(y: pd.Series) -> Any:
    """Pick the positive class label for binary classification."""
    vals = y.dropna().unique()
    if len(vals) != 2:
        return vals[0] if len(vals) else None
    for pref in (1, "1", True, "yes", "Yes", "positive", "Positive"):
        if pref in vals:
            return pref
    return sorted(vals, key=lambda x: str(x))[-1]


# Consistent class colors across all EDA subplots (Plotly default palette).
_CLASS_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _class_color(class_idx: int) -> str:
    return _CLASS_COLORS[class_idx % len(_CLASS_COLORS)]


def _class_labels(target: pd.Series) -> List[Any]:
    return sorted(target.dropna().unique(), key=lambda x: str(x))


def _stats_subtitle(series: pd.Series, target: pd.Series) -> str:
    parts: List[str] = []
    for cls in _class_labels(target):
        sub = series[target == cls].dropna()
        if len(sub):
            parts.append(f"{cls}: {sub.mean():.2f}±{sub.std():.2f}")
    return " | ".join(parts)


def _add_continuous_grouped_distribution(
    fig,
    series: pd.Series,
    target: pd.Series,
    row: int,
    legend_added: set,
) -> None:
    """Grouped probability bars per bin — both classes side-by-side (like reference EDA)."""
    import plotly.graph_objects as go

    clean = series.dropna()
    if clean.empty:
        return
    bins = np.histogram_bin_edges(clean, bins=30)
    centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) * 0.42

    for j, cls in enumerate(_class_labels(target)):
        sub = series[target == cls].dropna()
        if sub.empty:
            continue
        counts, _ = np.histogram(sub, bins=bins)
        total = counts.sum()
        prob = counts / total if total else counts.astype(float)
        label = str(cls)
        fig.add_trace(
            go.Bar(
                x=centers,
                y=prob,
                name=label,
                width=bar_width,
                marker=dict(color=_class_color(j), line=dict(width=0.4, color="white")),
                legendgroup="target_class",
                showlegend=label not in legend_added,
            ),
            row=row,
            col=1,
        )
        legend_added.add(label)
    fig.update_yaxes(title_text="probability", row=row, col=1)


def _add_categorical_class_bars(
    fig,
    series: pd.Series,
    target: pd.Series,
    pos_label: Any,
    row: int,
    legend_added: set,
) -> None:
    """Grouped user counts per category — one bar per class, % positive annotated above."""
    import plotly.graph_objects as go

    s_str = series.astype(str)
    y_str = target.astype(str)
    categories = sorted(s_str.dropna().unique(), key=str)
    ct = pd.crosstab(s_str, y_str)

    for j, cls in enumerate(_class_labels(target)):
        cls_str = str(cls)
        counts = [
            int(ct.loc[cat, cls_str]) if cat in ct.index and cls_str in ct.columns else 0
            for cat in categories
        ]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                name=cls_str,
                marker=dict(color=_class_color(j), line=dict(width=0.5, color="white")),
                legendgroup="target_class",
                showlegend=cls_str not in legend_added,
            ),
            row=row,
            col=1,
        )
        legend_added.add(cls_str)

    for cat in categories:
        if cat not in ct.index:
            continue
        row_ct = ct.loc[cat]
        total = float(row_ct.sum())
        if total <= 0:
            continue
        if pos_label is not None and str(pos_label) in row_ct.index:
            pct = 100.0 * float(row_ct[str(pos_label)]) / total
        elif len(row_ct):
            pct = 100.0 * float(row_ct.iloc[-1]) / total
        else:
            pct = 0.0
        cat_max = float(row_ct.max())
        fig.add_annotation(
            x=cat,
            y=cat_max,
            text=f"{pct:.1f}%",
            showarrow=False,
            yshift=12,
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="# users", row=row, col=1)


def generate_eda_plotly(
    project_id: str,
    csv_path: str,
    features: List[str],
    target: Optional[str],
    task_type: str = "",
) -> Dict[str, Any]:
    """Build interactive Plotly HTML — one subplot row per modeling feature."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return {"ok": False, "error": "plotly not installed"}

    if not csv_path or not Path(csv_path).exists():
        return {"ok": False, "error": "csv not found"}

    df = load_csv_for_analysis(csv_path, target_column=target)

    features = [f for f in features if f in df.columns and f != target][:24]
    if not features:
        return {"ok": False, "error": "no features to plot"}

    task = (task_type or "").lower()
    is_regression = task == "regression"
    is_classification = task in ("binary_classification", "multiclass_classification", "classification")
    n_target_classes = 0
    if target and target in df.columns:
        n_target_classes = int(df[target].nunique(dropna=True))
        if not is_regression and not is_classification:
            is_classification = 2 <= n_target_classes <= 50
            is_regression = n_target_classes > 50 and pd.api.types.is_numeric_dtype(df[target])
    is_binary = is_classification and n_target_classes == 2

    pos_label = None
    if is_binary and target and target in df.columns:
        pos_label = _positive_class(df[target])

    subplot_titles: List[str] = []
    conclusions: List[str] = []

    for feat in features:
        s = df[feat]
        title = f"Feature: {feat}"
        if is_binary and target and target in df.columns:
            y = df[target]
            if pd.api.types.is_numeric_dtype(s):
                title = f"{feat}<br>{_stats_subtitle(s, y)}"
            elif not pd.api.types.is_bool_dtype(s):
                pos_name = str(pos_label) if pos_label is not None else "positive"
                title = f"{feat} ({pos_name} % above each category)"
        elif is_classification and not is_binary:
            title = f"{feat} (by class)"
        subplot_titles.append(title)

        if pd.api.types.is_numeric_dtype(s):
            conclusions.append(
                f"**{feat}** (continuous): mean={s.mean():.3f}, std={s.std():.3f}, "
                f"missing={s.isna().mean():.1%}."
            )
        else:
            conclusions.append(
                f"**{feat}** ({s.nunique()} levels): top category "
                f"'{s.mode().iloc[0] if not s.mode().empty else '?'}'."
            )

    n_rows = len(features)
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.04, min(0.10, 1.0 / max(n_rows, 1))),
    )

    legend_added: set = set()

    for i, feat in enumerate(features, start=1):
        s = df[feat]
        row = i

        if is_classification and target and target in df.columns:
            y = df[target]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                _add_continuous_grouped_distribution(fig, s, y, row, legend_added)
            else:
                _add_categorical_class_bars(fig, s, y, pos_label, row, legend_added)
        else:
            if pd.api.types.is_numeric_dtype(s):
                import plotly.graph_objects as go
                clean = s.dropna()
                if not clean.empty:
                    fig.add_trace(
                        go.Histogram(x=clean, name=feat, histnorm="percent", nbinsx=40, showlegend=False),
                        row=row,
                        col=1,
                    )
            else:
                vc = s.astype(str).value_counts().head(20)
                import plotly.graph_objects as go
                fig.add_trace(
                    go.Bar(
                        x=vc.index.tolist(),
                        y=vc.values.tolist(),
                        name=feat,
                        showlegend=False,
                        width=0.55,
                    ),
                    row=row,
                    col=1,
                )

    height = max(520, 420 * n_rows)
    fig.update_layout(
        height=height,
        title_text="EDA — Modeling Features (Pre-Split, Unscaled)",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        legend=dict(
            title="Class",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            xanchor="left",
        ),
    )

    out = artifact_store.plots_dir(project_id) / "eda_features.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)

    summary = "\n".join(f"- {c}" for c in conclusions[:20])
    artifact_store.write_text(project_id, "eda_plotly_conclusions.md", summary)

    return {
        "ok": True,
        "html_path": str(out),
        "html_name": out.name,
        "n_features": len(features),
        "conclusions": conclusions,
    }


def generate_split_target_plotly(
    project_id: str,
    csv_path: str,
    target: str,
    split_meta: Dict[str, Any],
    scaling_method: str = "standard",
) -> Dict[str, Any]:
    """Plot target distribution per train/valid/test split."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return {"ok": False, "error": "plotly not installed"}

    prepared = artifact_store.project_artifact_dir(project_id) / "prepared"
    idx_path = prepared / "split_idx.npz"
    if not idx_path.exists():
        return {"ok": False, "error": "split indices not found"}

    idx = np.load(idx_path)
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        return {"ok": False, "error": f"target {target} not in data"}

    splits = {
        "Train": idx["train"],
        "Validation": idx["valid"],
        "Test": idx["test"],
    }
    report = split_meta or {}
    counts = {
        "Train": report.get("train_rows", len(splits["Train"])),
        "Validation": report.get("valid_rows", len(splits["Validation"])),
        "Test": report.get("test_rows", len(splits["Test"])),
    }
    total = sum(counts.values()) or 1
    ratios = {k: f"{v} ({v/total:.0%})" for k, v in counts.items()}

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        f"Train {ratios['Train']}",
        f"Validation {ratios['Validation']}",
        f"Test {ratios['Test']}",
    ])

    for col_i, (name, indices) in enumerate(splits.items(), start=1):
        y = df.iloc[indices][target]
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            clean = y.dropna()
            fig.add_trace(
                go.Violin(x=clean, name=name, box_visible=True, meanline_visible=True, points=False),
                row=1,
                col=col_i,
            )
        else:
            vc = y.astype(str).value_counts()
            fig.add_trace(
                go.Bar(
                    x=vc.index.tolist(),
                    y=vc.values.tolist(),
                    name=name,
                    showlegend=False,
                    width=0.55,
                ),
                row=1,
                col=col_i,
            )

    fig.update_layout(
        height=420,
        bargap=0.35,
        title_text=f"Target Distribution by Split · Scaling: {scaling_method} (fit on train)",
    )
    out = artifact_store.plots_dir(project_id) / "split_target_distribution.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)

    artifact_store.write_json(project_id, "split_ratios.json", {
        "train_rows": counts["Train"],
        "valid_rows": counts["Validation"],
        "test_rows": counts["Test"],
        "train_pct": round(counts["Train"] / total, 4),
        "valid_pct": round(counts["Validation"] / total, 4),
        "test_pct": round(counts["Test"] / total, 4),
        "scaling_method": scaling_method,
    })

    return {"ok": True, "html_path": str(out), "html_name": out.name, "ratios": ratios}
