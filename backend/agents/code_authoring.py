"""Code-writing agents.

Only these agents author and execute Python. Planning/review agents call them
(describing *what* code is needed); the code agents build the code, run it
through the **code-tools** MCP server (write → validate → run), inspect the
result, and (on failure) hand off to the **Code Debugger Agent** for bounded
repair. This keeps reasoning separate from execution and code generation safe.

Implemented code agents:
- EDA Code Agent           : writes/runs a consolidated EDA script
- Preprocessing Code Agent : writes/runs a reproducible preprocessing script
- Modeling Code Agent      : writes/runs a reproducible baseline-modeling script
- Iteration Code Agent     : writes a reproducible snippet per accepted change
- Code Debugger Agent      : repairs a failing script and re-runs (LLM-assisted
                             when configured, otherwise a no-op with a clear note)

Note: the *canonical* preprocessing/modeling execution still runs through the
dedicated preprocessing/modeling MCP tool servers (tested engines). The scripts
authored here are standalone, validated, executed reproductions that produce the
table artifacts and power the final notebook. This separation is intentional and
documented in the README.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.mcp_client.client import MCPClient
from backend.services.llm import llm

CODE = "code-tools"


# ---------------------------------------------------------------------------
# Shared loop: validate -> write -> run -> (debug) for one script
# ---------------------------------------------------------------------------
def run_code_agent(
    client: MCPClient,
    project_id: str,
    filename: str,
    code: str,
    max_debug: int = 2,
) -> Dict[str, Any]:
    """Validate, write and run a script; debug-retry on failure."""
    validation = client.call_tool(CODE, "validate_no_shell_commands", {"code": code})
    if not validation.get("ok", True):
        return {"ok": False, "blocked": True, "stderr": f"shell violations: {validation.get('violations')}",
                "filename": filename, "plots": [], "tables": []}

    client.call_tool(CODE, "write_python_file", {
        "project_id": project_id, "filename": filename, "code": code,
    })
    result = client.call_tool(CODE, "run_python_file", {
        "project_id": project_id, "filename": filename,
    })

    attempts = 0
    while not result.get("ok") and not result.get("blocked") and attempts < max_debug:
        fixed = code_debugger(client, project_id, filename, code, result)
        if not fixed:
            break
        code = fixed
        result = client.call_tool(CODE, "run_python_file", {
            "project_id": project_id, "filename": filename,
        })
        attempts += 1

    result["filename"] = filename
    result["debug_attempts"] = attempts
    return result


def code_debugger(
    client: MCPClient,
    project_id: str,
    filename: str,
    code: str,
    run_result: Dict[str, Any],
) -> Optional[str]:
    """Code Debugger Agent: try to repair a failing script and re-write it.

    Uses the LLM when configured. Without an LLM it returns None (no blind
    edits), and the failure is surfaced honestly to the caller/UI.
    """
    stderr = run_result.get("stderr", "")
    if not llm.enabled:
        return None
    system = (
        "You are a Python debugging agent. You are given a script and the error "
        "it produced. Return ONLY the full corrected script (no prose, no shell "
        "commands, no subprocess). Keep it self-contained and matplotlib-only."
    )
    prompt = f"Script `{filename}`:\n```python\n{code}\n```\nError:\n{stderr[:1500]}\nReturn the corrected full script."
    fixed = llm.complete(system, prompt, max_tokens=1500)
    if not fixed:
        return None
    fixed = _strip_fences(fixed)
    # Re-validate + persist via the tool layer.
    validation = client.call_tool(CODE, "validate_no_shell_commands", {"code": fixed})
    if not validation.get("ok", True):
        return None
    client.call_tool(CODE, "write_python_file", {
        "project_id": project_id, "filename": filename, "code": fixed,
    })
    return fixed


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Code templates
# ---------------------------------------------------------------------------
_HEADER = """import json, warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)
df = pd.read_csv({csv_path!r})
"""


def build_eda_code(csv_path: str, target: Optional[str], numeric_cols: List[str],
                   time_col: Optional[str]) -> str:
    parts = [_HEADER.format(csv_path=csv_path)]
    parts.append(f"TARGET = {target!r}\nNUMERIC = {json.dumps(numeric_cols)}\nTIME_COL = {time_col!r}\n")
    parts.append(
        "summary = {}\n"
        "# 1) missingness table + plot\n"
        "miss = df.isna().mean().sort_values(ascending=False)\n"
        "miss.rename('frac_missing').to_csv('tables/missingness.csv')\n"
        "m = miss[miss > 0]\n"
        "plt.figure(figsize=(7,4))\n"
        "(plt.barh(m.index[::-1], m.values[::-1]) if len(m) else plt.text(0.5,0.5,'No missing values',ha='center'))\n"
        "plt.title('Missing values by column'); plt.tight_layout(); plt.savefig('plots/missingness.png', dpi=110); plt.close()\n"
        "summary['n_missing_cols'] = int((miss>0).sum())\n"
    )
    parts.append(
        "# 2) target distribution\n"
        "if TARGET and TARGET in df.columns:\n"
        "    s = df[TARGET]\n"
        "    plt.figure(figsize=(6,4))\n"
        "    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:\n"
        "        plt.hist(s.dropna(), bins=30, color='#4C78A8')\n"
        "    else:\n"
        "        vc = s.value_counts(dropna=False); plt.bar([str(i) for i in vc.index], vc.values, color='#4C78A8')\n"
        "        vc.rename('count').to_csv('tables/target_distribution.csv')\n"
        "    plt.title('Target distribution: '+str(TARGET)); plt.tight_layout(); plt.savefig('plots/target_distribution.png', dpi=110); plt.close()\n"
    )
    parts.append(
        "# 3) numeric feature distributions\n"
        "num = [c for c in (NUMERIC or df.select_dtypes(include=[np.number]).columns) if c in df.columns][:9]\n"
        "if num:\n"
        "    ncols=3; nrows=(len(num)+ncols-1)//ncols\n"
        "    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,3*nrows)); axes=np.array(axes).reshape(-1)\n"
        "    for i,c in enumerate(num): axes[i].hist(df[c].dropna(), bins=25, color='#54A24B'); axes[i].set_title(c, fontsize=9)\n"
        "    for j in range(len(num), len(axes)): axes[j].axis('off')\n"
        "    plt.tight_layout(); plt.savefig('plots/feature_distributions.png', dpi=110); plt.close()\n"
        "    df[num].describe().T.to_csv('tables/numeric_summary.csv')\n"
    )
    parts.append(
        "# 4) feature-target relationships\n"
        "if TARGET and TARGET in df.columns:\n"
        "    is_class = (not pd.api.types.is_numeric_dtype(df[TARGET])) or df[TARGET].nunique() <= 15\n"
        "    fnum = [c for c in num if c != TARGET][:6]\n"
        "    if fnum:\n"
        "        ncols=3; nrows=(len(fnum)+ncols-1)//ncols\n"
        "        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,3*nrows)); axes=np.array(axes).reshape(-1)\n"
        "        for i,c in enumerate(fnum):\n"
        "            ax=axes[i]\n"
        "            if is_class:\n"
        "                g=df.dropna(subset=[c]).groupby(TARGET)[c]; ax.boxplot([v.values for _,v in g], labels=[str(k) for k,_ in g])\n"
        "            else:\n"
        "                ax.scatter(df[c], df[TARGET], s=8, alpha=0.4)\n"
        "            ax.set_title(c, fontsize=9)\n"
        "        for j in range(len(fnum), len(axes)): axes[j].axis('off')\n"
        "        plt.tight_layout(); plt.savefig('plots/feature_target.png', dpi=110); plt.close()\n"
    )
    parts.append(
        "# 5) correlation\n"
        "numdf = df.select_dtypes(include=[np.number])\n"
        "if numdf.shape[1] >= 2:\n"
        "    corr = numdf.corr(); corr.to_csv('tables/correlation.csv')\n"
        "    plt.figure(figsize=(1+0.6*corr.shape[1],1+0.6*corr.shape[1]))\n"
        "    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1); plt.colorbar(fraction=0.046)\n"
        "    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8); plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)\n"
        "    plt.title('Correlation (numeric)'); plt.tight_layout(); plt.savefig('plots/correlation.png', dpi=110); plt.close()\n"
    )
    parts.append(
        "# 6) outliers (IQR)\n"
        "rows=[]\n"
        "for c in numdf.columns:\n"
        "    s=numdf[c].dropna()\n"
        "    if len(s)<5: continue\n"
        "    q1,q3=s.quantile(0.25),s.quantile(0.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr\n"
        "    n=int(((s<lo)|(s>hi)).sum())\n"
        "    if n: rows.append({'column':c,'n_outliers':n,'pct':round(n/len(s),4)})\n"
        "pd.DataFrame(rows).to_csv('tables/outliers.csv', index=False)\n"
    )
    parts.append(
        "# 7) time series of target\n"
        "if TIME_COL and TARGET and TIME_COL in df.columns and TARGET in df.columns:\n"
        "    d=df[[TIME_COL,TARGET]].copy(); d[TIME_COL]=pd.to_datetime(d[TIME_COL], errors='coerce')\n"
        "    d=d.dropna(subset=[TIME_COL]).sort_values(TIME_COL)\n"
        "    g=d.groupby(d[TIME_COL].dt.to_period('D').dt.to_timestamp())[TARGET].mean()\n"
        "    plt.figure(figsize=(8,4)); plt.plot(g.index,g.values); plt.title(str(TARGET)+' over '+str(TIME_COL))\n"
        "    plt.tight_layout(); plt.savefig('plots/time_series.png', dpi=110); plt.close()\n"
    )
    parts.append("print('EDA_SUMMARY', json.dumps(summary))\n")
    return "\n".join(parts)


def build_preprocessing_code(csv_path: str, plan: Dict[str, Any], spec: Dict[str, Any]) -> str:
    target = (spec.get("targets") or [None])[0]
    keep = [c for c in plan.get("keep_columns", [])]
    strategy = spec.get("recommended_split", "random")
    stratify = strategy == "stratified"
    return _HEADER.format(csv_path=csv_path) + f"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

TARGET = {target!r}
FEATURES = {json.dumps(keep)}
data = df.dropna(subset=[TARGET]).reset_index(drop=True)
X, y = data[FEATURES], data[TARGET]
NUMERIC = [c for c in FEATURES if pd.api.types.is_numeric_dtype(data[c])]
CATEG = [c for c in FEATURES if c not in NUMERIC]
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUMERIC),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)]), CATEG),
])
strat = y if {stratify!r} else None
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
strat2 = y_tmp if {stratify!r} else None
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=strat2)
# Fit on TRAIN ONLY (leakage-safe), then transform
Xtr = pre.fit_transform(X_train); Xva = pre.transform(X_valid); Xte = pre.transform(X_test)
pd.DataFrame([
    {{'split':'train','rows':len(X_train),'features':Xtr.shape[1]}},
    {{'split':'valid','rows':len(X_valid),'features':Xva.shape[1]}},
    {{'split':'test','rows':len(X_test),'features':Xte.shape[1]}},
]).to_csv('tables/split_sizes.csv', index=False)
print('PREPROCESS_OK', Xtr.shape, Xva.shape, Xte.shape)
"""


def build_modeling_code(csv_path: str, plan: Dict[str, Any], spec: Dict[str, Any]) -> str:
    target = (spec.get("targets") or [None])[0]
    keep = [c for c in plan.get("keep_columns", [])]
    strategy = spec.get("recommended_split", "random")
    stratify = strategy == "stratified"
    task = spec.get("ml_task_type", "binary_classification")
    is_clf = task in ("binary_classification", "multiclass_classification", "anomaly_detection")
    return _HEADER.format(csv_path=csv_path) + f"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

IS_CLF = {is_clf!r}
TARGET = {target!r}
FEATURES = {json.dumps(keep)}
data = df.dropna(subset=[TARGET]).reset_index(drop=True)
X, y = data[FEATURES], data[TARGET]
if IS_CLF:
    y = y.astype('category').cat.codes
NUMERIC = [c for c in FEATURES if pd.api.types.is_numeric_dtype(data[c])]
CATEG = [c for c in FEATURES if c not in NUMERIC]
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUMERIC),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)]), CATEG),
])
strat = y if ({stratify!r} and IS_CLF) else None
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
strat2 = y_tmp if ({stratify!r} and IS_CLF) else None
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=strat2)
Xtr = pre.fit_transform(X_train); Xva = pre.transform(X_valid)

def score(model):
    model.fit(Xtr, y_train)
    if IS_CLF:
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        pred = model.predict(Xva)
        out = {{'accuracy': round(accuracy_score(y_valid, pred),4)}}
        try:
            import numpy as _np
            if len(_np.unique(y_train))==2 and hasattr(model,'predict_proba'):
                out['roc_auc']=round(roc_auc_score(y_valid, model.predict_proba(Xva)[:,1]),4)
            out['f1']=round(f1_score(y_valid, pred, average='binary' if len(_np.unique(y_train))==2 else 'macro'),4)
        except Exception: pass
        return out
    from sklearn.metrics import mean_squared_error, r2_score
    pred = model.predict(Xva)
    return {{'rmse': round(mean_squared_error(y_valid, pred)**0.5,4), 'r2': round(r2_score(y_valid, pred),4)}}

rows=[]
if IS_CLF:
    rows.append({{'model':'dummy', **score(DummyClassifier(strategy='most_frequent'))}})
    rows.append({{'model':'logistic', **score(LogisticRegression(max_iter=1000))}})
    rows.append({{'model':'random_forest', **score(RandomForestClassifier(n_estimators=200, random_state=0))}})
else:
    rows.append({{'model':'dummy', **score(DummyRegressor())}})
    rows.append({{'model':'ridge', **score(Ridge())}})
    rows.append({{'model':'random_forest', **score(RandomForestRegressor(n_estimators=200, random_state=0))}})
pd.DataFrame(rows).to_csv('tables/model_comparison_repro.csv', index=False)
print('MODELING_OK', json.dumps(rows))
"""


def build_iteration_code(iteration: int, model_name: str, params: Dict[str, Any], hypothesis: str) -> str:
    return (
        f"# Iteration {iteration} — reproducible change snippet (executed via modeling-tools in the run)\n"
        f"# Hypothesis: {hypothesis}\n"
        f"# Re-train the '{model_name}' family with overridden params and compare on validation.\n"
        f"PARAMS = {json.dumps(params)}\n"
        f"# model.set_params(**PARAMS); model.fit(Xtr, y_train); evaluate on (Xva, y_valid)\n"
        f"print('ITERATION_{iteration}', {json.dumps(model_name)}, PARAMS)\n"
    )
