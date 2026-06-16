"""Reproducible generator for the synthetic churn demo dataset.

Run:  python demo/_generate_dataset.py
Produces demo/sample_dataset.csv (a realistic, imbalanced churn problem with
categorical + numeric features, a time column, missing values, and an
intentionally leakage-prone future column).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200

platforms = rng.choice(["android", "ios", "web"], size=n, p=[0.45, 0.4, 0.15])
regions = rng.choice(["NA", "EU", "APAC", "LATAM"], size=n, p=[0.4, 0.3, 0.2, 0.1])
tiers = rng.choice(["basic", "standard", "premium"], size=n, p=[0.5, 0.35, 0.15])
s30 = rng.poisson(12, n) + 1
s7 = np.clip((s30 * rng.uniform(0.1, 0.45, n)).round().astype(int), 0, None)
avg_dur = np.round(rng.gamma(2.0, 3.0, n) + 1, 2)
errors = rng.poisson(1.2, n)

tier_eff = pd.Series(tiers).map({"basic": 0.9, "standard": 0.2, "premium": -0.6}).to_numpy()
plat_eff = pd.Series(platforms).map({"android": 0.2, "ios": -0.1, "web": 0.3}).to_numpy()
logit = (
    -0.4 - 0.12 * s7 - 0.03 * s30 + 0.25 * errors - 0.05 * avg_dur
    + tier_eff + plat_eff + rng.normal(0, 0.5, n)
)
p = 1 / (1 + np.exp(-logit))
churned = (rng.uniform(size=n) < p).astype(int)

# Intentionally leakage-prone column: activity AFTER the prediction point.
next7 = np.where(churned == 1, rng.poisson(0.3, n), rng.poisson(4, n))

ts = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D")

avg_dur_missing = avg_dur.copy()
miss_idx = rng.choice(n, size=int(0.05 * n), replace=False)
avg_dur_missing[miss_idx] = np.nan

df = pd.DataFrame(
    {
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "timestamp": ts.strftime("%Y-%m-%d"),
        "platform": platforms,
        "region": regions,
        "num_sessions_last_7d": s7,
        "num_sessions_last_30d": s30,
        "avg_session_duration": avg_dur_missing,
        "num_errors": errors,
        "subscription_tier": tiers,
        "num_sessions_next_7d": next7,  # leakage-prone (future) column on purpose
        "churned": churned,
    }
)

out = Path(__file__).resolve().parent / "sample_dataset.csv"
df.to_csv(out, index=False)
print("wrote", out, "| rows", len(df), "| churn rate", round(float(churned.mean()), 3))
