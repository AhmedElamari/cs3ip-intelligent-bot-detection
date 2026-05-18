"""Bake RandomForest + medians for Streamlit Tab 3 (live prediction)."""

from __future__ import annotations

import argparse
import json
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

    hpo_path = hpo_json or (
        _REPO_ROOT / "results/hpo_cache/random_forest"
        / "e076c10b28698ab7ae24b52bbf79b89fbc22d4a1286ea8db8d5bdcf91529162f.json"
    )
    if not hpo_path.is_file():
        raise FileNotFoundError(f"HPO JSON not found: {hpo_path}")
    rf_kwargs, hpo_meta = _load_hpo_bundle(hpo_path)
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
        help="HPOResultV1 JSON for Random Forest (default: repo cached RF study).",
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
