"""Tests for large CSV I/O helpers."""
from __future__ import annotations

from backend.services import file_store
from backend.services.csv_io import (
    chunked_missing_fractions,
    estimate_row_count,
)
from mcp_servers.data_quality_server import scan_data_quality


def test_estimate_row_count_and_chunked_missing(isolated_storage):
    pid = "csv-io-1"
    lines = ["a,b,c", "1,2,3", "1,,3", "2,3,4", "2,,4"]
    path = file_store.save_bytes(pid, "small.csv", "\n".join(lines).encode())
    assert estimate_row_count(str(path)) == 4
    miss = chunked_missing_fractions(str(path))
    assert miss["b"] == 0.5


def test_scan_uses_full_file_missing_on_sample(isolated_storage, monkeypatch):
    pid = "csv-io-2"
    lines = ["target,region,ok"] + ["1,EU,1" for _ in range(60)] + ["0,,1" for _ in range(40)]
    path = file_store.save_bytes(pid, "med.csv", "\n".join(lines).encode())
    monkeypatch.setattr("backend.services.csv_io.settings.large_csv_bytes", 10)
    result = scan_data_quality({"csv_path": str(path), "spec": {"targets": ["target"]}})
    by_name = {p["name"]: p for p in result["column_profiles"]}
    assert by_name["region"]["role"] == "drop"
    assert by_name["region"]["pct_missing"] == 0.4
