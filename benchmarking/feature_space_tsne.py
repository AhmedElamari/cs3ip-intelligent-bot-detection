"""t-SNE visual evidence for clean and adversarial engineered feature space."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from adversarial import RealisticPerturbationEngine
from benchmarking.poster_figures import SCENARIO_LABELS, poster_style


SCENARIO_ORDER = ("human_clean", "bot_clean", "bot_cheap_only", "bot_realistic_mixed")
SCENARIO_DISPLAY = {
    "human_clean": "Clean humans",
    "bot_clean": "Clean true bots",
    "bot_cheap_only": "Bots after cheap attacks",
    "bot_realistic_mixed": "Bots after mixed realistic attacks",
}


def save_attack_feature_space_tsne(
    benchmark: Any,
    feature_names: Sequence[str],
    output_dir: Path,
    profiles: Sequence[str] = ("cheap_only", "realistic_mixed"),
    perplexity: float = 30.0,
    random_state: int = 2112,
) -> Optional[Path]:
    """Save t-SNE scatter, coordinates, and caption for attack-space evidence."""
    if benchmark.base_train_inputs is None or benchmark.base_test_inputs is None:
        return None
    if benchmark.base_y_train is None or benchmark.y_test is None:
        return None

    feature_names = list(feature_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _to_frame(benchmark.base_train_inputs, feature_names)
    test = _to_frame(benchmark.base_test_inputs, feature_names)
    y_train = np.asarray(benchmark.base_y_train)
    y_test = np.asarray(benchmark.y_test)
    if test.empty or y_test.size != len(test):
        return None

    engine = RealisticPerturbationEngine(
        feature_names=feature_names,
        X_train=train,
        y_train=y_train,
    )
    stacked = _stack_scenarios(test, y_test, engine, profiles)
    if len(stacked) <= 2:
        return None

    effective_perplexity = min(float(perplexity), max(1.0, float(len(stacked) - 1) / 3.0))
    scaler = StandardScaler()
    scaler.fit(train)
    embedded = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
    ).fit_transform(scaler.transform(stacked[feature_names]))

    coords = stacked[["scenario", "true_label", "source_test_row_index"]].copy()
    coords.insert(0, "tsne_y", embedded[:, 1])
    coords.insert(0, "tsne_x", embedded[:, 0])
    coords.to_csv(output_dir / "attack_feature_space_tsne.csv", index=False)
    _save_plot(coords, output_dir / "attack_feature_space_tsne.png")
    _write_caption(output_dir / "attack_feature_space_tsne_caption.md", profiles, effective_perplexity, random_state)
    return output_dir / "attack_feature_space_tsne.png"


def _to_frame(data: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.loc[:, list(feature_names)].copy()
    return pd.DataFrame(data, columns=feature_names)


def _stack_scenarios(
    test: pd.DataFrame,
    y_test: np.ndarray,
    engine: RealisticPerturbationEngine,
    profiles: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for scenario, mask in (("human_clean", y_test == 0), ("bot_clean", y_test == 1)):
        chunk = test.loc[mask].copy()
        if chunk.empty:
            continue
        chunk["scenario"] = scenario
        chunk["true_label"] = y_test[mask].astype(int)
        chunk["source_test_row_index"] = test.index[mask].astype(int)
        rows.append(chunk)

    bot_mask = y_test == 1
    bots = test.loc[bot_mask].copy()
    if not bots.empty:
        for profile in profiles:
            result = engine.apply_profile(bots, profile)
            if not result.applied:
                continue
            chunk = result.data.copy()
            chunk["scenario"] = f"bot_{profile}"
            chunk["true_label"] = 1
            chunk["source_test_row_index"] = chunk.index.astype(int)
            rows.append(chunk)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _save_plot(coords: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "human_clean": "#9AA0A6",
        "bot_clean": "#4C72B0",
        "bot_cheap_only": "#DD8452",
        "bot_realistic_mixed": "#C44E52",
    }
    markers = {
        "human_clean": "o",
        "bot_clean": "o",
        "bot_cheap_only": "^",
        "bot_realistic_mixed": "s",
    }
    with poster_style():
        fig, ax = plt.subplots(figsize=(8, 6))
        for scenario in SCENARIO_ORDER:
            subset = coords[coords["scenario"].eq(scenario)]
            if subset.empty:
                continue
            ax.scatter(
                subset["tsne_x"],
                subset["tsne_y"],
                s=36,
                alpha=0.72,
                marker=markers.get(scenario, "o"),
                color=colors.get(scenario),
                label=SCENARIO_DISPLAY.get(scenario, scenario),
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.set_title("Clean vs adversarial engineered feature space")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _write_caption(
    output_path: Path,
    profiles: Sequence[str],
    perplexity: float,
    random_state: int,
) -> None:
    profile_labels = ", ".join(SCENARIO_LABELS.get(profile, profile) for profile in profiles)
    text = (
        "Qualitative 2-D t-SNE embedding of train-standardised engineered account features for "
        f"clean humans, clean true bots, and true bots under {profile_labels}. "
        "This visual is qualitative evidence only and is not a statistical test; apparent clusters "
        "or shifts should be interpreted as exploratory feature-space structure. "
        f"Settings: perplexity={perplexity:g}, init=PCA, learning_rate=auto, random_state={random_state}, "
        "with StandardScaler fit on the training split."
    )
    output_path.write_text(text + "\n", encoding="utf-8")
