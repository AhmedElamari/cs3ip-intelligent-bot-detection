"""Tab 3 — Live prediction (RF + SHAP, gauge iframe animation)."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from FeatureEngineering import BotFeatureExtractor
from streamlit_demo.data import (
    DEMO_ENGINEERED_FEATURE_COUNT,
    DEMO_LIVE_MEDIAN_FILLED_COUNT,
    LIVE_UI_EDITABLE_FEATURE_COUNT,
)

SHAP_DISPLAY_FEATURES: tuple[str, ...] = (
    "is_verified",
    "tweets_per_day",
    "followers_to_friends_ratio",
    "account_age_days",
    "default_profile_image",
    "description_length",
    "screen_name_has_digits",
    "followers_count",
    "friends_count",
    "statuses_count",
)

LIVE_SHAP_LABELS: dict[str, str] = {
    "followers_to_friends_ratio": "ff_ratio",
    "default_profile_image": "default_avatar",
    "screen_name_has_digits": "name_has_digits",
}

LIVE_TOGGLE_LABELS: dict[str, str] = {
    "is_verified": "Verified",
    "default_profile_image": "Default avatar",
    "screen_name_has_digits": "Digits in name",
}


def format_shap_feature_label(name: str) -> str:
    return LIVE_SHAP_LABELS.get(name, name)


def _hpo_provenance(pred: Any) -> dict[str, Any] | None:
    """Safe read for cached or legacy predictor objects missing ``hpo_provenance``."""
    prov = getattr(pred, "hpo_provenance", None)
    return cast(dict[str, Any], prov) if isinstance(prov, dict) else None


LIVE_DEFAULTS: dict[str, int | float | bool] = {
    "followers_count": 50,
    "friends_count": 2000,
    "account_age_days": 3,
    "statuses_count": 600,
    "description_length": 0,
    "is_verified": False,
    "default_profile_image": True,
    "screen_name_has_digits": True,
}


def compute_derived_pair(
    followers: float,
    friends: float,
    age_days: float,
    statuses: float,
) -> tuple[float, float]:
    age_safe = max(float(age_days), 1.0)
    tpd = float(statuses) / age_safe
    tpd = min(tpd, float(BotFeatureExtractor.TWEETS_PER_DAY_CAP))
    ratio = float(followers) / (float(friends) + 1.0)
    ratio = min(ratio, float(BotFeatureExtractor.FOLLOWERS_FRIENDS_RATIO_CAP))
    return tpd, ratio


def format_derived_display(tpd: float, ff_ratio: float) -> tuple[str, str]:
    tpd_s = f"{tpd:.1f}" if math.isfinite(tpd) else "\u221e"
    ff_s = f"{ff_ratio:.3f}" if math.isfinite(ff_ratio) else "\u221e"
    return tpd_s, ff_s


@dataclass
class LivePredictor:
    model: Any
    feature_order: list[str]
    medians: np.ndarray
    hpo_provenance: dict[str, Any] | None = None
    _explainer: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._idx = {n: i for i, n in enumerate(self.feature_order)}
        if len(self.medians) != len(self.feature_order):
            raise ValueError("medians length must match feature_order")

    def assemble_row(self, ui: Mapping[str, Any]) -> np.ndarray:
        row = np.array(self.medians, dtype=np.float64, copy=True)
        followers = float(ui["followers_count"])
        friends = float(ui["friends_count"])
        age_days = float(ui["account_age_days"])
        statuses = float(ui["statuses_count"])
        tpd, ff_ratio = compute_derived_pair(followers, friends, age_days, statuses)
        overrides = {
            "followers_count": followers,
            "friends_count": friends,
            "account_age_days": age_days,
            "statuses_count": statuses,
            "tweets_per_day": tpd,
            "followers_to_friends_ratio": ff_ratio,
            "description_length": float(ui["description_length"]),
            "is_verified": float(ui["is_verified"]),
            "default_profile_image": float(ui["default_profile_image"]),
            "screen_name_has_digits": float(ui["screen_name_has_digits"]),
        }
        for name, val in overrides.items():
            i = self._idx.get(name)
            if i is not None:
                row[i] = val
        return row

    def predict(self, ui: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        row = self.assemble_row(ui).reshape(1, -1)
        bot_score = float(self.model.predict_proba(row)[0, 1])
        explainer = self._ensure_explainer()
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            vec = np.asarray(sv[1], dtype=np.float64).ravel()
        else:
            arr = np.asarray(sv, dtype=np.float64)
            if arr.ndim == 3:
                vec = arr[0, :, 1].ravel()
            elif arr.ndim == 2:
                vec = arr[0].ravel()
            else:
                vec = arr.ravel()
        if vec.size != len(self.feature_order):
            raise RuntimeError(
                f"SHAP length {vec.size} != features {len(self.feature_order)}"
            )
        full = dict(zip(self.feature_order, vec))
        display = {k: float(full[k]) for k in SHAP_DISPLAY_FEATURES if k in full}
        return bot_score, display

    def _ensure_explainer(self) -> Any:
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer


def load_live_predictor(path: Path) -> LivePredictor | None:
    try:
        raw = joblib.load(path)
    except Exception:
        return None

    try:
        if isinstance(raw, LivePredictor):
            prov = _hpo_provenance(raw)
            return LivePredictor(
                model=raw.model,
                feature_order=list(raw.feature_order),
                medians=np.asarray(raw.medians, dtype=np.float64),
                hpo_provenance=prov,
            )

        if not isinstance(raw, dict):
            return None
        if raw.get("schema_version") != "LivePredictorV1":
            return None
        fo = list(raw["feature_order"])
        med = np.asarray(raw["medians"], dtype=np.float64)
        prov = raw.get("hpo_provenance")
        if prov is not None and not isinstance(prov, dict):
            prov = None
        return LivePredictor(
            model=raw["model"],
            feature_order=fo,
            medians=med,
            hpo_provenance=prov,
        )
    except Exception:
        return None


def build_fallback_predictor_for_demo() -> LivePredictor:
    """Train a small RF on synthetic rows when no baked joblib exists (Tab 3 still works)."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from Preprocessing import BotDetector

    rng = np.random.RandomState(2112)
    n = 160
    ref = pd.Timestamp("2022-06-15")
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        created = ref - pd.Timedelta(days=int(rng.randint(10, 900)))
        rows.append(
            {
                "account_creation_date": created,
                "is_verified": int(rng.randint(0, 2)),
                "followers_count": int(rng.randint(1, 20000)),
                "friends_count": int(rng.randint(1, 8000)),
                "listed_count": int(rng.randint(0, 100)),
                "statuses_count": int(rng.randint(0, 8000)),
                "favourites_count": int(rng.randint(0, 3000)),
                "default_profile": int(rng.randint(0, 2)),
                "default_profile_image": int(rng.randint(0, 2)),
                "has_extended_profile": int(rng.randint(0, 2)),
                "geo_enabled": int(rng.randint(0, 2)),
                "protected": int(rng.randint(0, 2)),
                "description": "x" * int(rng.randint(0, 140)),
                "url": None,
                "screen_name": f"user_{rng.randint(1000, 99999)}",
                "label": int(rng.randint(0, 2)),
            }
        )
    df = pd.DataFrame(rows)
    ext = BotFeatureExtractor(reference_date=ref)
    fe = ext.extract_all_features(df)
    det = BotDetector()
    det.data = fe
    fe = det.preprocess()
    names = fe.drop(columns=["label"]).select_dtypes(include=[np.number]).columns.tolist()
    if not names:
        raise RuntimeError("Fallback feature extraction produced no numeric columns")
    x_arr = fe[names].astype(np.float64).values
    y_arr = fe["label"].astype(int).values
    med = np.median(x_arr, axis=0).astype(np.float64)
    model = RandomForestClassifier(n_estimators=100, random_state=2112, n_jobs=-1)
    model.fit(x_arr, y_arr)
    return LivePredictor(model=model, feature_order=list(names), medians=med, hpo_provenance=None)


@st.cache_resource
def cached_live_predictor(path_str: str) -> tuple[LivePredictor | None, str]:
    path = Path(path_str)
    loaded = load_live_predictor(path)
    if loaded is not None:
        return loaded, "artifact"
    try:
        return build_fallback_predictor_for_demo(), "fallback"
    except Exception:
        return None, "error"


def build_live_intro_html() -> str:
    return (
        '<div class="demo-live-intro-row">'
        "<div>"
        '<span class="demo-section-label">Section 03</span>'
        '<h1 class="demo-h1 demo-live-h1">Live prediction</h1>'
        '<p class="demo-lede demo-live-lede">'
        f"Adjust the {LIVE_UI_EDITABLE_FEATURE_COUNT} most impactful features. "
        f"The remaining {DEMO_LIVE_MEDIAN_FILLED_COUNT} are filled with training-set medians."
        "</p>"
        "</div>"
        f'<div class="demo-live-meta">Random Forest · {DEMO_ENGINEERED_FEATURE_COUNT} features</div>'
        "</div>"
    )


def build_idle_panel_html() -> str:
    return (
        '<div class="demo-result-card demo-idle-card">'
        '<div class="demo-idle-inner">'
        '<p class="demo-idle-text">Configure parameters on the left<br/>'
        "and run the prediction.</p>"
        "</div></div>"
    )


def build_artifact_missing_html(asset_relpath: str) -> str:
    cmd = (
        "python -m streamlit_demo.bake_live_artifact "
        "--train-split-dir data "
        f"--out {asset_relpath}"
    )
    return (
        '<div class="demo-result-card demo-artifact-missing">'
        '<span class="demo-section-label">Artifact missing</span>'
        '<p class="demo-artifact-missing-text">Train and save the live predictor:</p>'
        f'<pre class="demo-artifact-cmd">{html.escape(cmd)}</pre>'
        "</div>"
    )


def build_verdict_html(bot_score: float) -> str:
    is_bot = bot_score >= 0.5
    conf = bot_score if is_bot else 1.0 - bot_score
    pct = int(round(conf * 100))
    title = "Bot detected" if is_bot else "Human"
    color = "#C95444" if is_bot else "#4DA87A"
    return (
        '<div class="demo-verdict-block">'
        f'<div class="demo-verdict-title" style="color:{color};">{html.escape(title)}</div>'
        f'<div class="demo-verdict-sub">{pct}% confidence · Random Forest</div>'
        "</div>"
    )


def build_shap_waterfall_html(contributions: Sequence[tuple[str, float]]) -> str:
    rows = [(n, float(v)) for n, v in contributions if abs(float(v)) > 1e-12]
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    rows = rows[:5]
    if not rows:
        return (
            '<div class="demo-shap-waterfall-card demo-shap-waterfall-card--compact">'
            '<span class="demo-section-label">Feature contributions (SHAP waterfall)</span>'
            '<p class="demo-shap-empty">No strong signals at current values.</p>'
            "</div>"
        )
    mx = max(abs(v) for _, v in rows)
    parts: list[str] = [
        '<div class="demo-shap-waterfall-card demo-shap-waterfall-card--compact">',
        '<span class="demo-section-label">Feature contributions (SHAP waterfall)</span>',
        '<div class="demo-shap-rows">',
    ]
    for name, val in rows:
        w = (abs(val) / mx) * 46.0 if mx > 0 else 0.0
        pos = val > 0
        col = "#C95444" if pos else "#4DA87A"
        side_class = "demo-shap-fill--pos" if pos else "demo-shap-fill--neg"
        sign = "+" if pos else ""
        disp = f"{sign}{val:.2f}"
        parts.append(
            '<div class="demo-shap-row">'
            f'<div class="demo-shap-name">{html.escape(format_shap_feature_label(name))}</div>'
            '<div class="demo-shap-track">'
            '<div class="demo-shap-center-line"></div>'
            f'<div class="demo-shap-fill {side_class}" '
            f'style="width:{w:.2f}%;background:{col};"></div>'
            "</div>"
            f'<span class="demo-shap-val" style="color:{col};">{disp}</span>'
            "</div>"
        )
    parts.append("</div>")
    parts.append(
        '<div class="demo-shap-legend">'
        '<span><span class="demo-shap-legend-red">\u2192</span> pushes toward bot</span>'
        '<span><span class="demo-shap-legend-green">\u2190</span> pushes toward human</span>'
        "</div></div>"
    )
    return "".join(parts)


def build_gauge_html(bot_score: float, confidence_pct: int, version: int) -> str:
    gid = f"gArc_{version}"
    score_js = f"{float(bot_score):.8f}"
    conf_js = int(confidence_pct)
    ver_js = int(version)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
:root {{
  --green:#4DA87A; --amber:#C8A44A; --red:#C95444; --accent:#6CC8BE;
  --bg:#101218; --text-dim:#6a7186;
}}
body {{ margin:0; background:#181b23; border-radius:10px; border:1px solid rgba(255,255,255,0.08);
  padding:12px 16px; box-sizing:border-box; min-height:100px; }}
svg {{ display:block; margin:0 auto; overflow:visible; }}
</style></head><body>
<svg id="gSvg" width="236" height="130" viewBox="0 0 280 155">
<defs>
<linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="oklch(0.68 0.14 150)"/>
<stop offset="50%" stop-color="oklch(0.75 0.12 75)"/>
<stop offset="100%" stop-color="oklch(0.65 0.17 20)"/>
</linearGradient>
</defs>
<path id="gTrack" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="10" stroke-linecap="round"/>
<path id="gTint" fill="none" stroke="url(#{gid})" stroke-width="10" stroke-linecap="round" opacity="0.2"/>
<path id="gFill" fill="none" stroke-width="10" stroke-linecap="round" opacity="0.85"/>
<line id="gNeedle" stroke-width="2" stroke-linecap="round"/>
<circle id="gHubO" cx="140" cy="125" r="7" fill="var(--bg)" stroke-width="1.5"/>
<circle id="gHubI" cx="140" cy="125" r="2.5"/>
<text x="25" y="142" font-family="JetBrains Mono, monospace" font-size="10" fill="var(--green)" text-anchor="middle">HUMAN</text>
<text x="255" y="142" font-family="JetBrains Mono, monospace" font-size="10" fill="var(--red)" text-anchor="middle">BOT</text>
<text id="gPct" x="140" y="112" font-family="JetBrains Mono, monospace" font-size="24" font-weight="600" text-anchor="middle"></text>
<text x="140" y="128" font-family="JetBrains Mono, monospace" font-size="9" fill="var(--text-dim)" text-anchor="middle">CONFIDENCE</text>
</svg>
<script>
(function() {{
  const cx = 140, cy = 125, r = 95;
  const storageKey = 'demo-gauge-prev-angle';
  const botScore = {score_js};
  const confPct = {conf_js};
  const version = {ver_js};
  function needleColor(score) {{
    if (score < 0.5) return 'var(--green)';
    if (score < 0.75) return 'var(--amber)';
    return 'var(--red)';
  }}
  function arcPath(s, e) {{
    const toRad = d => (d / 180) * Math.PI;
    const x1 = cx + r * Math.cos(Math.PI - toRad(s));
    const y1 = cy - r * Math.sin(Math.PI - toRad(s));
    const x2 = cx + r * Math.cos(Math.PI - toRad(e));
    const y2 = cy - r * Math.sin(Math.PI - toRad(e));
    const large = e - s > 180 ? 1 : 0;
    return 'M ' + x1 + ' ' + y1 + ' A ' + r + ' ' + r + ' 0 ' + large + ' 1 ' + x2 + ' ' + y2;
  }}
  const track = document.getElementById('gTrack');
  const tint = document.getElementById('gTint');
  const fill = document.getElementById('gFill');
  const needle = document.getElementById('gNeedle');
  const hubO = document.getElementById('gHubO');
  const hubI = document.getElementById('gHubI');
  const pctEl = document.getElementById('gPct');
  const fullArc = arcPath(0, 180);
  track.setAttribute('d', fullArc);
  tint.setAttribute('d', fullArc);
  pctEl.textContent = confPct + '%';
  const col = needleColor(botScore);
  fill.setAttribute('stroke', col);
  needle.setAttribute('stroke', col);
  hubO.setAttribute('stroke', col);
  hubI.setAttribute('fill', col);
  const endAngle = botScore * 180;
  let startAngle = parseFloat(sessionStorage.getItem(storageKey) || '0');
  if (!Number.isFinite(startAngle)) startAngle = 0;
  const dur = 1000;
  const t0 = performance.now();
  function needleCoords(angle) {{
    const rad = (angle / 180) * Math.PI;
    const nx = cx + (r - 10) * Math.cos(Math.PI - rad);
    const ny = cy - (r - 10) * Math.sin(Math.PI - rad);
    return {{ nx, ny }};
  }}
  function tick(now) {{
    const t = Math.min((now - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const ang = startAngle + (endAngle - startAngle) * ease;
    fill.setAttribute('d', arcPath(0, ang));
    const nc = needleCoords(ang);
    needle.setAttribute('x1', cx);
    needle.setAttribute('y1', cy);
    needle.setAttribute('x2', nc.nx);
    needle.setAttribute('y2', nc.ny);
    if (t < 1) requestAnimationFrame(tick);
    else sessionStorage.setItem(storageKey, String(endAngle));
  }}
  requestAnimationFrame(tick);
}})();
</script>
</body></html>"""


def _live_toggle(label: str, default_yes: bool) -> bool:
    key = f"live_seg_{label}"
    display = LIVE_TOGGLE_LABELS.get(label, label.replace("_", " "))
    sel = st.segmented_control(
        display,
        ["Yes", "No"],
        default="Yes" if default_yes else "No",
        key=key,
    )
    if isinstance(sel, (list, tuple)):
        sel = sel[0] if sel else "No"
    return str(sel or "No") == "Yes"


def live_header_status(pred: LivePredictor | None, source: str) -> str:
    if pred is None and source == "error":
        return "Predictor unavailable"
    if pred is not None:
        return "Model loaded"
    return "Demo"


def live_header_hint_from_path(path: Path) -> str:
    return "Model loaded" if path.is_file() else "Demo"


def render_live_prediction(
    predictor_path: Path,
    *,
    bundle: tuple[LivePredictor | None, str] | None = None,
) -> None:
    for k, v in LIVE_DEFAULTS.items():
        sk = f"live_val_{k}"
        if sk not in st.session_state:
            st.session_state[sk] = v
    if "live_predicted" not in st.session_state:
        st.session_state.live_predicted = False
    if "live_run_count" not in st.session_state:
        st.session_state.live_run_count = 0
    if "live_last_bot_score" not in st.session_state:
        st.session_state.live_last_bot_score = 0.5
    if "live_last_contribs" not in st.session_state:
        st.session_state.live_last_contribs = []

    pred, source = (
        bundle
        if bundle is not None
        else cached_live_predictor(str(predictor_path.resolve()))
    )

    st.markdown('<div class="demo-live-page-body">', unsafe_allow_html=True)
    st.markdown(build_live_intro_html(), unsafe_allow_html=True)

    st.markdown('<div class="demo-live-main-row-anchor" aria-hidden="true"></div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([0.28, 0.72], gap="small")

    with col_left:
        with st.container(border=True):
            st.markdown(
                '<span class="demo-section-label">Account parameters</span>',
                unsafe_allow_html=True,
            )
            followers = st.slider(
                "followers_count",
                0,
                100_000,
                value=int(st.session_state.live_val_followers_count),
                step=10,
                key="live_followers",
            )
            friends = st.slider(
                "friends_count",
                0,
                100_000,
                value=int(st.session_state.live_val_friends_count),
                step=10,
                key="live_friends",
            )
            age_days = st.slider(
                "account_age_days",
                1,
                5000,
                value=int(st.session_state.live_val_account_age_days),
                step=1,
                key="live_age",
            )
            statuses = st.slider(
                "statuses_count",
                0,
                100_000,
                value=int(st.session_state.live_val_statuses_count),
                step=10,
                key="live_statuses",
            )
            tpd, ff = compute_derived_pair(followers, friends, age_days, statuses)
            tpd_s, ff_s = format_derived_display(tpd, ff)
            st.markdown(
                f'<div class="demo-derived-pair"><div><div class="demo-derived-lbl">tweets_per_day</div>'
                f'<div class="demo-derived-val">{html.escape(tpd_s)}</div></div>'
                f'<div><div class="demo-derived-lbl">ff_ratio</div>'
                f'<div class="demo-derived-val">{html.escape(ff_s)}</div></div></div>',
                unsafe_allow_html=True,
            )
            desc_len = st.slider(
                "description_length",
                0,
                160,
                value=int(st.session_state.live_val_description_length),
                step=1,
                key="live_desc",
            )
            t_col1, t_col2, t_col3 = st.columns(3, gap="small")
            with t_col1:
                verified = _live_toggle("is_verified", False)
            with t_col2:
                default_img = _live_toggle("default_profile_image", True)
            with t_col3:
                name_digits = _live_toggle("screen_name_has_digits", True)

            run = st.button("\u25b6  Run prediction", type="primary", key="live_run_btn")

    ui = {
        "followers_count": followers,
        "friends_count": friends,
        "account_age_days": age_days,
        "statuses_count": statuses,
        "description_length": desc_len,
        "is_verified": 1 if verified else 0,
        "default_profile_image": 1 if default_img else 0,
        "screen_name_has_digits": 1 if name_digits else 0,
    }

    if run and pred is not None:
        bs, contribs = pred.predict(ui)
        st.session_state.live_predicted = True
        st.session_state.live_run_count += 1
        st.session_state.live_last_bot_score = bs
        st.session_state.live_last_contribs = list(contribs.items())

    with col_right:
        if pred is None and source == "error":
            st.markdown(
                build_artifact_missing_html("demo_assets/live_predictor.joblib"),
                unsafe_allow_html=True,
            )
        elif not st.session_state.live_predicted:
            st.markdown(build_idle_panel_html(), unsafe_allow_html=True)
        else:
            bs = float(st.session_state.live_last_bot_score)
            is_bot = bs >= 0.5
            conf = bs if is_bot else 1.0 - bs
            pct = int(round(conf * 100))
            rc = int(st.session_state.live_run_count)
            st.markdown('<div class="demo-live-results-panel">', unsafe_allow_html=True)
            with st.container(border=True):
                components.html(
                    build_gauge_html(bs, pct, rc),
                    height=158,
                    scrolling=False,
                )
                st.markdown(build_verdict_html(bs), unsafe_allow_html=True)
            contribs = list(st.session_state.live_last_contribs)
            with st.container(border=True):
                st.markdown(build_shap_waterfall_html(contribs), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
