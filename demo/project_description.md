# Project Brief: Subscription Churn Prediction

## Business problem
Our subscription product is losing customers. The growth team wants to **predict
which customers are likely to churn** so they can target retention offers before
the customer leaves. A false negative (missing a churner) is more costly than a
false positive (offering a discount to someone who would have stayed), so recall
on churners matters, but we also care about overall ranking quality.

## Project goals
- Build a model that predicts whether a customer will churn.
- Make the modeling **leakage-aware**: we must not use any activity that happens
  *after* the prediction point (future activity) as a feature.
- Produce a clear, reproducible analysis that another data scientist can review.

## Available data
A single CSV, `sample_dataset.csv`, with one row per customer at the prediction
time.

## Column meaning
- `customer_id` — unique customer identifier (not a feature).
- `timestamp` — the date the snapshot/prediction is made.
- `platform` — the customer's primary platform (`android`, `ios`, `web`).
- `region` — customer region (`NA`, `EU`, `APAC`, `LATAM`).
- `num_sessions_last_7d` — sessions in the 7 days before the prediction point.
- `num_sessions_last_30d` — sessions in the 30 days before the prediction point.
- `avg_session_duration` — average session duration (minutes); may be missing.
- `num_errors` — number of errors the customer hit recently.
- `subscription_tier` — `basic`, `standard`, or `premium`.
- `num_sessions_next_7d` — sessions in the 7 days **after** the prediction point.
  **WARNING: this is future information and must NOT be used as a feature
  (leakage).** It is included to test whether the system detects leakage.
- `churned` — **target**: 1 if the customer churned, 0 otherwise.

## Target variable
`churned` (binary classification).

## Known constraints
- Do not leak future activity (e.g. `num_sessions_next_7d`).
- The same customer should not appear in both train and test (here each customer
  appears once, but the split should still be leakage-aware).
- Keep a held-out test set for a single final evaluation.

## Expected deliverables
- A reproducible Jupyter notebook containing project understanding, EDA,
  preprocessing, baseline models, an iterative improvement loop, and a final
  test evaluation.

## Evaluation preferences
- Primary metric: **ROC-AUC** (ranking quality).
- Also report F1 and recall on churners.
