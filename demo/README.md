# Demo: Subscription Churn

This folder contains a small, self-contained demo project for Project2Notebook.

## Files
- `project_description.md` — the project brief (business + ML goals, columns,
  constraints, evaluation preferences).
- `sample_dataset.csv` — a synthetic but realistic churn dataset (1,200 rows).
- `_generate_dataset.py` — the reproducible generator for the CSV.

## What it exercises
- **Binary classification** (`churned`).
- **Class (im)balance** (~1:2 churn ratio).
- **Categorical** (`platform`, `region`, `subscription_tier`) and **numeric**
  features.
- A **time column** (`timestamp`) → time-aware / grouped split discussion.
- An intentional **leakage-prone future column** (`num_sessions_next_7d`) to test
  leakage detection.
- Missing values in `avg_session_duration`.
- Feature-engineering opportunities and model comparison.

## Run it
From the repository root:

```bash
python -m backend.cli demo
```

This runs the full agentic pipeline and prints the timeline, tool-call count,
summary, and the path to the generated notebook + artifacts.

To regenerate the dataset:

```bash
python demo/_generate_dataset.py
```
