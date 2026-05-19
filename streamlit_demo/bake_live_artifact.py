"""Bake RandomForest + medians for Streamlit Tab 3 (live prediction)."""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from DataLoader import TwiBotDataLoader, load_twibot_splits_as_dict
from FeatureEngineering import BotFeatureExtractor, derive_reference_date
from Preprocessing import BotDetector

_SCHEMA = "LivePredictorV1"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HPO_JSON = _REPO_ROOT / "demo_assets" / "rf_hpo_defaults.json"


def _normalize_label_value(value: Any) -> int | None:
    """Match DataLoader embedded-label rules (0/1, bot/human strings)."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("0", "1"):
            return int(normalized)
        if normalized in ("bot", "fake"):
            return 1
        if normalized in ("human", "real"):
            return 0
        numeric = pd.to_numeric(normalized, errors="coerce")
        if numeric in (0, 1):
            return int(numeric)
        return None
    if isinstance(value, numbers.Integral):
        return int(value) if value in (0, 1) else None
    if isinstance(value, numbers.Real):
        if value in (0, 1) and float(value).is_integer():
            return int(value)
    return None


def _normalize_label_column(series: pd.Series) -> pd.Series:
    mapped = series.map(_normalize_label_value)
    if mapped.isna().any():
        bad = int(mapped.isna().sum())
        raise ValueError(f"{bad} label value(s) could not be normalized to 0/1.")
    return mapped.astype(int)

def _merge_labels(df: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    lab = pd.read_csv(labels_path)
    id_col = next((c for c in lab.columns if str(c).strip().lower() == "id"), None)
    if id_col is None:
        id_col = next((c for c in lab.columns if str(c).strip().upper() == "ID"), None)
    if id_col is None:
        raise ValueError(f"No id/ID column in {labels_path}")
    if "label" not in lab.columns:
        raise ValueError(f"No label column in {labels_path}")
    merge_df = lab[[id_col, "label"]].copy()
    merge_df[id_col] = merge_df[id_col].astype(str).str.strip()
    out = df.copy()
    if "user_id" not in out.columns:
        raise ValueError("Expected user_id column for label merge")
    out["_merge_key"] = out["user_id"].astype(str).str.strip()
    merged = out.merge(
        merge_df.rename(columns={id_col: "_merge_key"}),
        on="_merge_key",
        how="inner",
    )
    merged = merged.drop(columns=["_merge_key"], errors="ignore")
    return merged


def _numeric_feature_names(train_df: pd.DataFrame) -> list[str]:
    names = (
        train_df.drop(columns=["label"])
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    if not names:
        raise ValueError("No numeric features after preprocessing.")
    return names


def _sklearn_rf_kwargs(hpo_best: dict[str, Any], random_state: int) -> dict[str, Any]:
    allowed = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    }
    merged: dict[str, Any] = {k: v for k, v in hpo_best.items() if k in allowed}
    merged["random_state"] = random_state
    merged["class_weight"] = "balanced"
    merged["n_jobs"] = -1
    merged["oob_score"] = True
    return merged


def _load_hpo_bundle(hpo_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(hpo_json.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "HPOResultV1":
        raise ValueError(f"Expected HPOResultV1 in {hpo_json}")
    bp = raw.get("best_params") or {}
    try:
        rel = hpo_json.relative_to(_REPO_ROOT)
        rel_s = str(rel).replace("\\", "/")
    except ValueError:
        rel_s = str(hpo_json)
    provenance: dict[str, Any] = {
        "hpo_artifact_relpath": rel_s,
        "best_val_f1": float(raw.get("best_score", 0.0)),
        "trial_count": int(raw.get("trial_count", 0)),
        "metric": str(raw.get("metric", "val_f1")),
        "search_space_version": raw.get("search_space_version"),
    }
    return _sklearn_rf_kwargs(bp, random_state=int(raw.get("seed", 2112))), provenance


def _resolve_rf_kwargs(
    hpo_json: Path | None,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hpo_path = hpo_json or _DEFAULT_HPO_JSON
    if hpo_path.is_file():
        return _load_hpo_bundle(hpo_path)
    from config import load_config

    print(
        f"Warning: HPO JSON not found at {hpo_path}; using config random_forest defaults.",
        file=sys.stderr,
    )
    cfg = load_config()
    params = dict(cfg.get("models.random_forest.params") or {})
    params["random_state"] = random_state
    provenance = {
        "hpo_artifact_relpath": "(config defaults)",
        "best_val_f1": None,
        "trial_count": 0,
        "metric": "config",
        "search_space_version": None,
    }
    return _sklearn_rf_kwargs(params, random_state=random_state), provenance


def bake(
    out_path: Path,
    *,
    data_path: Path | None = None,
    labels_path: Path | None = None,
    train_split_dir: Path | None = None,
    hpo_json: Path | None = None,
    random_state: int = 2112,
    benchmark_xai_relpath: str = "results/benchmark_20260517_091938",
) -> None:
    if train_split_dir is not None:
        splits = load_twibot_splits_as_dict(train_split_dir)
        train_df = splits["train"]
    else:
        if data_path is None:
            raise ValueError("Provide data_path or train_split_dir.")
        loader = TwiBotDataLoader(json_path=data_path)
        df = loader.load()
        if "label" not in df.columns:
            if labels_path is None:
                raise ValueError("Dataset has no label column; provide --labels CSV.")
            df = _merge_labels(df, labels_path)
        df = df.dropna(subset=["label"])
        if df.empty:
            raise ValueError("No labeled rows.")
        try:
            train_df, _rest = train_test_split(
                df,
                test_size=0.2,
                random_state=random_state,
                stratify=df["label"],
            )
        except ValueError:
            train_df = df

    train_df = train_df.dropna(subset=["label"])
    if train_df.empty:
        raise ValueError("No labeled training rows.")
    train_df = train_df.copy()
    train_df["label"] = _normalize_label_column(train_df["label"])

    rf_kwargs, hpo_meta = _resolve_rf_kwargs(hpo_json, random_state)
    rf_kwargs["random_state"] = random_state

    reference_date = derive_reference_date(train_df)

    extractor = BotFeatureExtractor(reference_date=reference_date)
    train_fe = extractor.extract_all_features(train_df.copy())

    detector = BotDetector()
    detector.data = train_fe
    train_fe = detector.preprocess()

    feature_order = _numeric_feature_names(train_fe)
    X = train_fe[feature_order].astype(np.float64)
    y = train_fe["label"].astype(int)

    medians = np.median(X.values, axis=0).astype(np.float64)

    model = RandomForestClassifier(**rf_kwargs)
    model.fit(X.values, y.values)

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "feature_order": feature_order,
        "medians": medians,
        "model": model,
        "hpo_provenance": {**hpo_meta, "benchmark_xai_relpath": benchmark_xai_relpath},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)

    loaded = joblib.load(out_path)
    assert loaded["schema_version"] == _SCHEMA
    chk = model.predict_proba(X.values[:1])[0, 1]
    chk2 = loaded["model"].predict_proba(X.values[:1])[0, 1]
    assert np.isclose(chk, chk2), "Round-trip model probability mismatch"
    print(f"Wrote {out_path} ({len(feature_order)} features, train n={len(X)})")
    print(f"HPO: {hpo_meta.get('hpo_artifact_relpath')} val_f1={hpo_meta.get('best_val_f1')}")
    print(f"Sanity check predict_proba(first row): {chk:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Bake live predictor artifact for VIVA Tab 3.")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to TwiBot-20 JSON (single file). Ignored when --train-split-dir is set.",
    )
    p.add_argument(
        "--train-split-dir",
        type=Path,
        default=None,
        help="Directory with train.json/dev.json/test.json (uses train split; matches benchmark).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels CSV (id + label) when JSON has no embedded labels",
    )
    p.add_argument(
        "--hpo-json",
        type=Path,
        default=None,
        help="HPOResultV1 JSON for Random Forest (default: demo_assets/rf_hpo_defaults.json).",
    )
    p.add_argument(
        "--benchmark-xai-rel",
        type=str,
        default="results/benchmark_20260517_091938",
        help="Relative path to RF explainability benchmark folder (provenance only).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "demo_assets" / "live_predictor.joblib",
        help="Output joblib path",
    )
    args = p.parse_args()
    train_dir = args.train_split_dir
    data_path = args.data
    if train_dir is None and data_path is None:
        default_data = _REPO_ROOT / "data"
        if (default_data / "train.json").is_file():
            train_dir = default_data
        else:
            data_path = _REPO_ROOT / "TwiBot-20_sample.json"
    bake(
        args.out,
        data_path=data_path,
        labels_path=args.labels,
        train_split_dir=train_dir,
        hpo_json=args.hpo_json,
        benchmark_xai_relpath=args.benchmark_xai_rel,
    )


if __name__ == "__main__":
    main()
