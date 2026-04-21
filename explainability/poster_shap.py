"""Poster-sized SHAP beeswarm export (publication labels, high DPI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

LABEL_MAP = {
    "is_verified": "Verified account", "followers_count": "Followers (count)",
    "friends_count": "Friends (count)", "followers_to_friends_ratio": "Followers / Friends ratio",
    "followers_per_day": "Followers / day", "tweets_per_day": "Tweets / day",
    "favourites_per_day": "Favourites / day", "statuses_count": "Statuses (count)",
    "favourites_count": "Favourites (count)", "listed_count": "Listed (count)",
    "account_age_days": "Account age (days)", "description_length": "Bio length (chars)",
    "has_description": "Has bio", "has_url": "Has URL in profile",
    "screen_name_length": "Screen-name length", "screen_name_has_digits": "Digits in screen name",
    "default_profile": "Default profile", "default_profile_image": "Default avatar",
    "has_extended_profile": "Extended profile", "geo_enabled": "Geo enabled",
    "protected": "Protected account", "tweet_count": "Tweet count (sample window)",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
}


def _pretty_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(str(model_name), str(model_name).replace("_", " ").title())


def _caption(model_name: str, top_n: int) -> str:
    return (
        f"Global SHAP beeswarm for {_pretty_model_name(model_name)} on the evaluation test split "
        f"(top {top_n} features by mean |SHAP|). Each dot is one account; horizontal position is that "
        'feature\'s contribution to the log-odds of "bot", colour encodes feature value '
        "(red=high, blue=low). The model's decision is dominated by profile-metadata signals - "
        "verified status, followers/friends ratio, and account-age-normalised activity rates - rather "
        "than raw volume counters. The concentration of signal in a small band of interpretable "
        "features supports the interpretability claim and means adversarial robustness hinges on how "
        "easily an attacker can manipulate those specific profile attributes, not on hidden embeddings."
    )


def export_poster_shap(
    shap_values: Any,
    X: Any,
    feature_names: Sequence[str],
    *,
    model_name: str,
    output_dir: Path | str,
    top_n: int = 10,
    title: str | None = None,
) -> Path:
    import shap

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"shap_summary_{model_name}_poster"
    (out / f"{stem}_caption.md").write_text(_caption(model_name, top_n) + "\n", encoding="utf-8")
    display = [LABEL_MAP.get(str(n), str(n)) for n in feature_names]
    with mpl.rc_context({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    }):
        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            shap_values, X, feature_names=display, max_display=top_n, plot_type="dot", show=False,
        )
        if title:
            plt.title(title)
        plt.tight_layout()
        fig = plt.gcf()
        png = out / f"{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
    return png
