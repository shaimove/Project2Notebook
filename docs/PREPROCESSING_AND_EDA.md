# Preprocessing, EDA Review, and Data Quality

This document describes how Project2Notebook turns raw CSV + brief into a
leakage-aware preprocessing plan, how the **Data Quality Review agent** cleans
raw data, how the **EDA Review agent** interprets plots and tables, and how
checkpoints enable resume.

---

## Pipeline placement

```
Project Understanding → Prior Art → **Data Quality Review**
  → Data Audit → EDA Planning → EDA Code Agent → **EDA Review**
  → Preprocessing Planner → Modeling → …
```

Checkpoints (SQLite) are saved **after every step** so a failed run can resume
from the last successful checkpoint (`POST /api/run` with `"resume": true`).

---

## Data Quality Review agent

### Role

**Review + remediation agent** that runs on the **raw uploaded CSV** before Data
Audit and EDA. It normalizes column names, fixes inconsistent categoricals,
coerces mixed numeric strings, drops index artifacts and duplicate rows, and
writes `cleaned_*.csv` for all downstream steps.

### MCP server: `data-quality-tools`

| Tool | Purpose |
|------|---------|
| `scan_data_quality` | Column names, value consistency, categories, missingness, duplicates |
| `apply_remediation` | Apply plan and write `cleaned_{original}.csv` under project uploads |

### Agent outputs

- `data_quality_report.json`, `data_quality.md`
- Updates `state.csv_paths` to the cleaned file
- Preserves `state.original_csv_paths` for reproducibility
- Updates working memory `data_quality_findings`

### Handling strategy

| Severity | Action |
|----------|--------|
| **Critical** | Drop row-identifier columns (never declared targets) |
| **Warning** | Rename, normalize categories, coerce types, drop duplicates |
| **Info** | Column name normalization suggestions |

---

## Preprocessing

### Goal

Build a **leakage-safe** train/valid/test split and a preprocessor fit **only on
train**, then transform valid/test for modeling tools.

### MCP server: `preprocessing-tools`

| Tool | Purpose |
|------|---------|
| `create_preprocessing_plan` | Heuristic plan: drop/keep columns, encoding, scaling, FE ideas |
| `check_preprocessing_leakage` | Flags risky columns / fit-on-train checks |
| `build_train_valid_test_split` | Creates split + writes prepared arrays |
| `fit_preprocessor_on_train` | Fits imputer/encoder/scaler on **train only** |
| `transform_valid_test` | Applies fitted preprocessor to valid/test |

### Plan inputs

1. **Column profile** from Data Audit (`data_audit_report.columns`)
2. **Project spec** (`project_spec.json`) — task type, targets, leakage risks, split strategy
3. **EDA findings** (`eda_findings.json`) — drops, important columns, engineering ideas

### Heuristic rules (before EDA merge)

- Drop: constant columns, >60% missing, ID-like names, leakage-prone names (`future`, `post_`, …), raw date/time columns used for splitting
- Keep: remaining feature columns; classify numeric vs categorical
- Missing values: median (numeric), most frequent (categorical)
- Encoding: one-hot; scaling: standard
- Leakage mitigations: fit on train only; hold out test until final evaluation

### EDA findings merge

When `eda_findings` is present, `create_preprocessing_plan`:

- Adds `features_to_drop` from EDA (never un-drops target)
- Ensures `important_columns` stay in `keep_columns`
- Appends `features_to_engineer` to `feature_engineering` notes
- Copies `preprocessing_implications` into plan `notes`

### Outputs

- `preprocessing_plan.json`, `split_report.json`, `preprocessing_decisions.md`
- Prepared data under `storage/artifacts/{project_id}/` (splits, preprocessor joblib)

---

## EDA Review agent

### Role

**Review-only agent** (no code execution). Bridges EDA artifacts → preprocessing
and working memory. LLMs are strongest at reading **figures**; tables supply
reproducible numeric evidence.

### MCP server: `eda-review-tools`

| Tool | Purpose |
|------|---------|
| `analyze_eda_tables` | Deterministic analysis of EDA CSV tables |
| `list_eda_plot_inventory` | Maps plot filenames → semantic descriptions + paths |

### Inputs

- `eda_artifacts.json` (plot/table paths from EDA Code Agent)
- `data_audit_report.json`, `project_spec.json`
- Plot PNGs under `artifacts/{project_id}/plots/`
- Tables under `artifacts/{project_id}/tables/`

### Analysis flow

1. **Tier 1 — Tables** (`analyze_eda_tables`)
   - `correlation.csv` → target correlation, multicollinear groups (|r| ≥ 0.9)
   - `outliers.csv` → clip/robust recommendations
   - `missingness.csv` → drop sparse columns
   - `numeric_summary.csv` → skew → log1p engineering ideas
   - `target_distribution.csv` + audit imbalance → stratification note

2. **Tier 2 — LLM vision** (when `OPENAI_API_KEY` set)
   - Each plot reviewed via `llm.complete_with_images`
   - Extracts: summary, observations, feature names mentioned
   - Plot types: target distribution, missingness, feature distributions, feature-target, correlation heatmap, time series

3. **Tier 3 — LLM synthesis** (optional)
   - Consolidates table + plot findings into final JSON
   - Must not contradict leakage drops from audit

### Decisions the agent can make

| Decision | Examples |
|----------|----------|
| **Feature selection** | `important_columns`, `features_to_drop`, `features_to_watch` |
| **Feature engineering** | `features_to_engineer`: `{base_column, transform, rationale}` e.g. log1p for skew |
| **Preprocessing hints** | scaling/clipping/stratification notes in `preprocessing_implications` |
| **Target insights** | imbalance, distribution shape |
| **Multicollinearity** | `multicollinear_groups`; drop redundant column (keep highest target correlation) |
| **Open questions** | Ambiguous patterns needing domain confirmation |

Each insight includes `evidence`, `evidence_source`, `confidence`, `recommended_action`.

### Outputs

- `eda_findings.json`, `eda_findings.md`
- Updates `working_context.json` / `working_context.md` (important columns, drops, FE ideas)
- Fed into `create_preprocessing_plan` via Preprocessing Planner node

### Offline behavior

Without LLM: table analysis still runs; plot inventory uses heuristic descriptions only.

---

## Checkpoints and resume (A1–A2)

### Storage

SQLite database: `backend/storage/checkpoints.db`

- `run_sessions` — one row per pipeline execution
- `checkpoints` — serialized `DataScientist` state after each step

### API

```json
POST /api/run
{
  "project_id": "...",
  "resume": true,
  "from_step": null
}
```

- `resume: true` — continue latest incomplete run from last checkpoint
- `from_step: N` — restart from step N (loads checkpoint N−1)
- Response includes `run_id`, `resumed_from_step`

```http
GET /api/projects/{id}/runs
GET /api/projects/{id}/runs/{run_id}/checkpoints
```

---

## Related files

| Area | Path |
|------|------|
| Data Quality agent | `backend/agents/nodes/data_quality.py` |
| Data quality MCP | `mcp_servers/data_quality_server.py` |
| EDA Review agent | `backend/agents/nodes/eda_review.py` |
| EDA review MCP | `mcp_servers/eda_review_server.py` |
| Preprocessing MCP | `mcp_servers/preprocessing_server.py` |
| Schemas | `backend/schemas/data_quality.py`, `backend/schemas/eda_findings.py` |
| Checkpoints | `backend/services/checkpoint_store.py` |
| Graph | `backend/agents/graph.py` |
