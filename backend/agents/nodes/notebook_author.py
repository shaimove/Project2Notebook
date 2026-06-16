"""Notebook Author Agent.

Assembles a clean, readable, reproducible Jupyter notebook from the run
artifacts (not a raw log dump). Uses the notebook MCP tools to set the spec and
export the .ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.services import artifact_store, memory

NB = "notebook-tools"

_FAMILY_IMPORT = {
    "dummy": ("from sklearn.dummy import DummyClassifier, DummyRegressor", "DummyClassifier(strategy='most_frequent')", "DummyRegressor()"),
    "linear": ("from sklearn.linear_model import LogisticRegression, Ridge", "LogisticRegression(max_iter=1000)", "Ridge()"),
    "tree": ("from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor", "DecisionTreeClassifier(max_depth=6, random_state=0)", "DecisionTreeRegressor(max_depth=6, random_state=0)"),
    "random_forest": ("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor", "RandomForestClassifier(n_estimators=200, random_state=0)", "RandomForestRegressor(n_estimators=200, random_state=0)"),
    "boosting": ("from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor", "HistGradientBoostingClassifier(random_state=0)", "HistGradientBoostingRegressor(random_state=0)"),
    "svm": ("from sklearn.svm import SVC, SVR", "SVC(probability=True)", "SVR()"),
}


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    _ = memory.load(project_id)  # read memory before assembling the notebook
    spec = state.get("project_spec") or {}
    audit = state.get("data_audit_report") or {}
    prior = state.get("prior_art_report") or {}
    plan = state.get("preprocessing_plan") or {}
    split = state.get("split_report") or {}
    rows = state.get("model_comparison") or []
    csv_path = (state.get("csv_paths") or [""])[0]

    sections = _build_sections(state, spec, audit, prior, plan, split, rows, csv_path)

    nb_spec = {"title": "Project2Notebook — Final Report", "sections": sections}
    client.call_tool(NB, "update_notebook", {"project_id": project_id, "spec": nb_spec})
    out = client.call_tool(NB, "export_final_notebook", {"project_id": project_id})

    state["notebook_path"] = out.get("notebook_path")
    memory.update(project_id, phase="Notebook Author")
    return state


def _md(text: str) -> Dict[str, str]:
    return {"cell_type": "markdown", "source": text}


def _code(text: str) -> Dict[str, str]:
    return {"cell_type": "code", "source": text}


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "- (none)"


def _read_code(project_id: str, name: str) -> str:
    """Read an authored code file from code/ (the actually executed script)."""
    path = artifact_store.code_dir(project_id) / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _build_sections(state, spec, audit, prior, plan, split, rows, csv_path) -> List[Dict[str, Any]]:
    target = audit.get("target")
    task = spec.get("ml_task_type", "unknown")
    metric = state.get("primary_metric", "")
    plots = (state.get("eda_artifacts") or {}).get("plots", [])

    setup_code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
        "from sklearn.model_selection import train_test_split\n\n"
        f"CSV_PATH = {csv_path!r}  # update if you move the data\n"
        f"TARGET = {target!r}\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "df.shape"
    )

    sections: List[Dict[str, Any]] = []

    sections.append({"title": "Project Summary", "cells": [
        _md(f"**Business goal:** {spec.get('business_goal','(see brief)')}\n\n"
            f"**ML task:** `{task}`  \n**Target:** `{target}`  \n"
            f"**Primary metric:** `{metric}`  \n**Recommended split:** `{spec.get('recommended_split')}`"),
    ]})

    sections.append({"title": "Business & ML Goal Interpretation", "cells": [
        _md(f"- **Unit of prediction:** {spec.get('unit_of_prediction')}\n"
            f"- **Unit of analysis:** {spec.get('unit_of_analysis')}\n"
            f"- **Has time component:** {spec.get('has_time_component')}\n"
            f"- **Success criteria:** {spec.get('success_criteria')}\n\n"
            f"**Assumptions:**\n{_bullets(spec.get('assumptions', []))}\n\n"
            f"**Open questions:**\n{_bullets(spec.get('open_questions', []))}"),
    ]})

    if prior.get("enabled"):
        sections.append({"title": "Prior-Art Summary", "cells": [
            _md(f"_{prior.get('summary','')}_\n\n"
                f"**Common models:**\n{_bullets(prior.get('common_models', []))}\n\n"
                f"**Common feature engineering:**\n{_bullets(prior.get('common_feature_engineering', []))}\n\n"
                f"**Leakage warnings:**\n{_bullets(prior.get('leakage_warnings', []))}\n\n"
                f"**Ideas to test (verify on data!):**\n{_bullets(prior.get('ideas_to_test', []))}\n\n"
                f"> {prior.get('message','')}"),
        ]})

    sections.append({"title": "Dataset Description & Setup", "cells": [
        _md(f"The dataset has **{audit.get('n_rows')} rows × {audit.get('n_cols')} columns**."),
        _code(setup_code),
        _code("df.head()"),
        _code("df.describe(include='all').T"),
    ]})

    sections.append({"title": "Data Audit", "cells": [
        _md(f"- Duplicate rows: **{audit.get('n_duplicate_rows')}**\n"
            f"- Near-empty columns: {audit.get('near_empty_columns')}\n"
            f"- Constant columns: {audit.get('constant_columns')}\n"
            f"- High-cardinality columns: {audit.get('high_cardinality_columns')}\n"
            f"- Time columns: {audit.get('time_columns')}\n"
            f"- Entity/ID columns: {audit.get('entity_columns')}\n"
            f"- Class imbalance ratio: {audit.get('class_imbalance')}\n\n"
            f"**Notes:**\n{_bullets(audit.get('notes', []))}"),
        _code("df.isna().mean().sort_values(ascending=False).head(15)"),
    ]})

    project_id = state["project_id"]
    eda_code = _read_code(project_id, "eda.py")
    eda_cells = [_md(state.get("eda_report") or
                     "EDA was executed via the controlled code runner; key plots are reproduced below.")]
    if eda_code:
        eda_cells.append(_md("**Authored & executed EDA script** (`code/eda.py`):"))
        eda_cells.append(_code(eda_code))
    else:
        eda_cells.append(_code(_eda_repro_code(target)))
    sections.append({"title": "EDA", "cells": eda_cells})

    sections.append({"title": "Data Quality Findings", "cells": [
        _md(_bullets(audit.get("notes", []) or ["No major data-quality issues detected."])),
    ]})

    preproc_code = _read_code(project_id, "preprocessing.py")
    preproc_cells = [
        _md(f"- **Drop columns:** {plan.get('drop_columns')}\n"
            f"- **Numeric columns:** {plan.get('numeric_columns')}\n"
            f"- **Categorical columns:** {plan.get('categorical_columns')}\n"
            f"- **Encoding:** {plan.get('encoding_strategy')}, **scaling:** {plan.get('scaling_strategy')}\n"
            f"- **Missing-value strategy:** {plan.get('missing_value_strategy')}\n\n"
            f"**Leakage mitigations:**\n{_bullets(plan.get('leakage_mitigations', []))}"),
    ]
    if preproc_code:
        preproc_cells.append(_md("**Authored & executed preprocessing script** (`code/preprocessing.py`):"))
        preproc_cells.append(_code(preproc_code))
    else:
        preproc_cells.append(_code(_preprocess_code(plan)))
    sections.append({"title": "Preprocessing Decisions", "cells": preproc_cells})

    sections.append({"title": "Train / Validation / Test Split Logic", "cells": [
        _md(f"Strategy: **{split.get('strategy')}** — {split.get('rationale')}\n\n"
            f"Sizes: train={split.get('train_rows')}, valid={split.get('valid_rows')}, test={split.get('test_rows')}.\n\n"
            "> Preprocessing is fit on the **training split only**; the **test split is held out** until the very end."),
        _code(_split_code(spec, split)),
        _code(_fit_transform_code()),
    ]})

    model_code = _read_code(project_id, "modeling.py")
    baseline_cells = [_md("Baseline model comparison (validation):"), _code(_comparison_code(rows))]
    if model_code:
        baseline_cells.append(_md("**Authored & executed reproducible modeling script** (`code/modeling.py`):"))
        baseline_cells.append(_code(model_code))
    sections.append({"title": "Baseline Models", "cells": baseline_cells})

    sections.append({"title": "First Modeling Conclusion", "cells": [
        _md(state.get("first_conclusion") or "(not available)"),
    ]})

    sections.append({"title": "Iterative Improvements", "cells": [
        _md(state.get("iteration_summary") or "(no iterations run)"),
    ]})

    sections.append({"title": "Final Selected Pipeline", "cells": [
        _md(f"Selected by validation: **`{state.get('best_pipeline_id')}`** "
            f"(validation {metric} = {round(state.get('best_validation_score'),4) if state.get('best_validation_score') is not None else 'n/a'})."),
        _code(_final_model_code(state, task)),
    ]})

    sections.append({"title": "Validation & Final Test Results", "cells": [
        _md(state.get("final_test_report") or "(not available)"),
    ]})

    sections.append({"title": "Error Analysis", "cells": [
        _md("Inspect where the model errs (e.g. confusion matrix for classification, "
            "residuals for regression) to guide the next round of work."),
        _code(_error_analysis_code(task)),
    ]})

    final = state.get("final_test_report_obj") or {}
    sections.append({"title": "Limitations", "cells": [
        _md(_bullets(final.get("limitations", []))),
    ]})
    sections.append({"title": "Recommended Next Steps", "cells": [
        _md(_bullets(final.get("future_directions", []))),
    ]})

    # Appendix: the shared agent memory (decision trail), for transparency.
    mem_md = memory.load_markdown(project_id)
    if mem_md:
        sections.append({"title": "Appendix · Working Context (Agent Memory)", "cells": [_md(mem_md)]})

    return sections


def _eda_repro_code(target) -> str:
    return (
        "num = df.select_dtypes(include=[np.number])\n"
        f"if {target!r} in df.columns:\n"
        f"    s = df[{target!r}]\n"
        "    plt.figure(figsize=(5,3))\n"
        "    (s.value_counts().plot(kind='bar') if s.nunique() <= 15 else plt.hist(s.dropna(), bins=30))\n"
        f"    plt.title('Target distribution: ' + {target!r}); plt.tight_layout(); plt.show()\n"
        "if num.shape[1] >= 2:\n"
        "    corr = num.corr()\n"
        "    plt.figure(figsize=(6,5)); plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1); plt.colorbar()\n"
        "    plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)\n"
        "    plt.title('Correlation'); plt.tight_layout(); plt.show()"
    )


def _preprocess_code(plan) -> str:
    return (
        f"FEATURES = {json.dumps([c for c in plan.get('keep_columns', [])])}\n"
        f"NUMERIC = {json.dumps(plan.get('numeric_columns', []))}\n"
        f"CATEGORICAL = {json.dumps(plan.get('categorical_columns', []))}\n"
        "try:\n"
        "    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
        "except TypeError:\n"
        "    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)\n"
        "pre = ColumnTransformer([\n"
        "    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUMERIC),\n"
        "    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)]), CATEGORICAL),\n"
        "])"
    )


def _split_code(spec, split) -> str:
    strategy = split.get("strategy", "random")
    stratify = "y" if strategy == "stratified" else "None"
    return (
        "data = df.dropna(subset=[TARGET]).reset_index(drop=True)\n"
        "X = data[FEATURES]\n"
        "y = data[TARGET]\n"
        f"# Split strategy: {strategy}\n"
        f"X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify={stratify})\n"
        f"strat2 = y_tmp if {('True' if strategy=='stratified' else 'False')} else None\n"
        "X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=strat2)\n"
        "[a.shape for a in (X_train, X_valid, X_test)]"
    )


def _fit_transform_code() -> str:
    return (
        "# Fit preprocessing on TRAIN ONLY, then transform valid/test\n"
        "Xtr = pre.fit_transform(X_train)\n"
        "Xva = pre.transform(X_valid)\n"
        "Xte = pre.transform(X_test)\n"
        "Xtr.shape, Xva.shape, Xte.shape"
    )


def _comparison_code(rows) -> str:
    return (
        f"comparison = pd.DataFrame({json.dumps(rows)})\n"
        "comparison"
    )


def _final_model_code(state, task) -> str:
    family = state.get("best_model_family") or "boosting"
    params = state.get("best_params") or {}
    imp, clf, reg = _FAMILY_IMPORT.get(family, _FAMILY_IMPORT["boosting"])
    is_clf = task in ("binary_classification", "multiclass_classification", "anomaly_detection")
    ctor = clf if is_clf else reg
    set_params = f"\nmodel.set_params(**{json.dumps(params)})" if params else ""
    return (
        f"{imp}\n"
        f"model = {ctor}{set_params}\n"
        "model.fit(Xtr, y_train)\n"
        "from sklearn.metrics import classification_report, mean_squared_error, r2_score\n"
        + ("print(classification_report(y_test, model.predict(Xte)))" if is_clf else
           "pred = model.predict(Xte)\nprint('RMSE', mean_squared_error(y_test, pred)**0.5, 'R2', r2_score(y_test, pred))")
    )


def _error_analysis_code(task) -> str:
    is_clf = task in ("binary_classification", "multiclass_classification", "anomaly_detection")
    if is_clf:
        return (
            "from sklearn.metrics import confusion_matrix\n"
            "cm = confusion_matrix(y_test, model.predict(Xte))\n"
            "plt.imshow(cm, cmap='Blues'); plt.title('Confusion matrix'); plt.colorbar()\n"
            "plt.xlabel('predicted'); plt.ylabel('actual'); plt.show()\ncm"
        )
    return (
        "resid = y_test - model.predict(Xte)\n"
        "plt.hist(resid, bins=30); plt.title('Residuals'); plt.show()\n"
        "resid.describe()"
    )
