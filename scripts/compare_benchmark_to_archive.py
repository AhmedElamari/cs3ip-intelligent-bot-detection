#!/usr/bin/env python3
"""Compare two benchmark ``model_comparison.csv`` files (e.g. archive vs new run)."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any


METRIC_KEYS = ("ACCURACY", "PRECISION", "RECALL", "F1", "ROC_AUC", "MCC")


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            name = (row.get("Model") or "").strip()
            if not name:
                continue
            rows[name] = row
        return rows


def _float_cell(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare archive vs new model_comparison.csv")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("Archive_results/results_cli_check/benchmark_20260306_160812/model_comparison.csv"),
        help="Baseline CSV",
    )
    parser.add_argument(
        "--new",
        type=Path,
        required=True,
        help="New run CSV",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"Archive CSV not found: {args.archive.resolve()}")
    if not args.new.is_file():
        raise SystemExit(f"New CSV not found: {args.new.resolve()}")

    base = _read_rows(args.archive)
    new = _read_rows(args.new)
    models = sorted(set(base.keys()) | set(new.keys()))

    print("Benchmark comparison: archive (baseline) vs new run")
    print(f"  Baseline: {args.archive.resolve()}")
    print(f"  New:      {args.new.resolve()}")
    print()

    for model in models:
        print(f"## {model}")
        if model not in base:
            print("  (not in baseline — new model only)\n")
            continue
        if model not in new:
            print("  (not in new run — dropped)\n")
            continue
        b, n = base[model], new[model]
        for key in METRIC_KEYS:
            bv = _float_cell(b, key)
            nv = _float_cell(n, key)
            if bv is None and nv is None:
                continue
            if bv is None:
                print(f"  {key}: n/a -> {nv:.6f}")
                continue
            if nv is None:
                print(f"  {key}: {bv:.6f} -> n/a")
                continue
            delta = nv - bv
            sign = "+" if delta >= 0 else ""
            print(f"  {key}: {bv:.6f} -> {nv:.6f}  (delta {sign}{delta:.6f})")
        bt = _float_cell(b, "Training Time (s)")
        nt = _float_cell(n, "Training Time (s)")
        if bt is not None and nt is not None:
            print(f"  Training Time (s): {bt:.3f} -> {nt:.3f}")
        print()


if __name__ == "__main__":
    main()
