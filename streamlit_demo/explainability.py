"""Tab 2 — Explainability: classed HTML (article, SHAP slot, feature bars, appendix)."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Sequence

from streamlit_demo.data import (
    DEMO_ENGINEERED_FEATURE_COUNT,
    FEATURE_IMPORTANCE_ROWS,
    RESILIENCE_ROWS,
    SHAP_SUMMARY_RF_PATH,
)
from streamlit_demo.model_arena import ACCENT

SHAP_SUMMARY_FILENAME = SHAP_SUMMARY_RF_PATH.name


def top_feature_names(rows: Sequence[tuple[str, float]], n: int = 3) -> list[str]:
    return [r[0] for r in rows[:n]]


def format_resilience_rows(rows: Sequence[tuple[str, str, float, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for feature, cost, flip, note in rows:
        pct = int(round(flip * 100))
        out.append(
            {
                "Feature": feature,
                "Evasion Cost": cost,
                "Flip Rate": f"{pct}%",
                "Note": note,
            }
        )
    return out


def shap_image_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_shap_placeholder_html(filename: str) -> str:
    return (
        '<div class="demo-shap-placeholder">'
        f'<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="4" y="4" width="32" height="32" rx="4" stroke="rgba(255,255,255,0.2)" '
        f'stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<line x1="20" y1="12" x2="20" y2="28" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>'
        f'<line x1="12" y1="20" x2="28" y2="20" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>'
        f"</svg>"
        f'<div class="demo-shap-placeholder-text">Drop <span style="color:{ACCENT};">{html.escape(filename)}</span> here</div>'
        '<div class="demo-shap-placeholder-hint">Regenerate with '
        "<code>python run_benchmark.py --models random_forest --explain</code> "
        f"then copy <code>shap_summary_random_forest.png</code> to <code>demo_assets/</code>.</div></div>"
    )


def _figure_heading(fig_label: str, subtitle: str) -> str:
    return (
        '<div class="demo-figure-head">'
        f'<span class="demo-figure-label">{html.escape(fig_label)}</span>'
        f'<span class="demo-figure-subtitle">{html.escape(subtitle)}</span>'
        "</div>"
        '<div class="demo-figure-rule"></div>'
    )


def _feature_bars_html(rows: Sequence[tuple[str, float]]) -> str:
    vals = [r[1] for r in rows]
    max_v = max(vals) if vals else 1.0
    parts: list[str] = ['<div class="demo-feature-bars">']
    for i, (name, val) in enumerate(rows):
        pct = min(100.0, (val / max_v) * 100.0) if max_v > 0 else 0.0
        top = i < 3
        ncls = "demo-feature-name demo-feature-name--top" if top else "demo-feature-name"
        vcls = "demo-feature-val demo-feature-val--top" if top else "demo-feature-val"
        fill = ACCENT if top else "rgba(255,255,255,0.15)"
        parts.append(
            '<div class="demo-feature-row">'
            f'<div class="{ncls}">{html.escape(name)}</div>'
            '<div class="demo-bar-track">'
            f'<div class="demo-bar-fill" style="width:{pct:.1f}%;background:{fill};"></div>'
            "</div>"
            f'<div class="{vcls}">{val:.3f}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _resilience_table_html(rows: Sequence[tuple[str, str, float, str]]) -> str:
    cells = format_resilience_rows(rows)
    th = (
        "<thead><tr>"
        "<th>Feature</th><th>Evasion Cost</th><th>Flip Rate</th><th>Note</th>"
        "</tr></thead>"
    )
    body = ["<tbody>"]
    for c in cells:
        body.append(
            "<tr>"
            f'<td class="demo-resilience-feature">{html.escape(c["Feature"])}</td>'
            f"<td>{html.escape(c['Evasion Cost'])}</td>"
            f"<td>{html.escape(c['Flip Rate'])}</td>"
            f"<td>{html.escape(c['Note'])}</td>"
            "</tr>"
        )
    body.append("</tbody>")
    return f'<table class="demo-resilience-table">{th}{"".join(body)}</table>'


def _aside_html(top_three: Sequence[str]) -> str:
    names = list(top_three)
    while len(names) < 3:
        names.append("\u2014")
    a, b, c = names[:3]
    return (
        '<div class="demo-aside-column">'
        '<div class="demo-aside-note">'
        '<span class="demo-aside-label">Note</span>'
        "<p style=\"margin-top:8px;\">SHAP (SHapley Additive exPlanations) attributes each prediction to the features "
        "that produced it. Unlike aggregate feature importance, SHAP values are computed "
        "<em>per-prediction</em>, enabling us to explain individual classifications.</p>"
        '<p style="margin-top:16px;">The top 3 features — '
        f'<span style="color:{ACCENT};">{html.escape(a)}</span>, '
        f'<span style="color:{ACCENT};">{html.escape(b)}</span>, and '
        f'<span style="color:{ACCENT};">{html.escape(c)}</span> — align with the '
        "Random Forest importance profile in Fig. 2 below.</p>"
        "</div></div>"
    )


def build_explainability_html(
    feature_rows: Sequence[tuple[str, float]],
    resilience_rows: Sequence[tuple[str, str, float, str]],
    *,
    shap_image_exists: bool,
    shap_image_src: str,
) -> str:
    top3 = top_feature_names(feature_rows, n=3)
    if shap_image_exists and shap_image_src:
        shap_block = (
            '<div class="demo-shap-slot">'
            f'<img class="demo-shap-img" src="{html.escape(shap_image_src, quote=True)}" alt="SHAP summary" />'
            "</div>"
        )
    else:
        shap_block = build_shap_placeholder_html(SHAP_SUMMARY_FILENAME)

    return (
        '<div class="demo-explain-grid">'
        '<article class="demo-explain-article">'
        '<span class="demo-section-label">Section 02</span>'
        '<h2 class="demo-h1" style="font-size:32px;">Why did it predict that?</h2>'
        '<p class="demo-lede" style="max-width:640px;margin-top:12px;">A model\'s predictions are only useful if we '
        "can interrogate them. The SHAP summary plot shows how individual features push predictions toward bot or "
        "human, and the bar chart shows which features the Random Forest emphasised in this benchmark export.</p>"
        '<div style="margin-top:48px;"></div>'
        f"{_figure_heading('Fig. 1', f'SHAP summary — Random Forest, top 10 of {DEMO_ENGINEERED_FEATURE_COUNT} features')}"
        f"{shap_block}"
        f'<p class="demo-figure-caption">Fig. 1 — Bot-class SHAP values on up to 500 held-out test rows; '
        f"the beeswarm shows the top 10 of {DEMO_ENGINEERED_FEATURE_COUNT} engineered numeric features "
        "(by mean |SHAP|).</p>"
        '<div style="margin-top:56px;"></div>'
        f"{_figure_heading('Fig. 2', f'Feature importance — Top 10 of {DEMO_ENGINEERED_FEATURE_COUNT}')}"
        f"{_feature_bars_html(feature_rows)}"
        '<p class="demo-fig2-foot">Fig. 2 — Random Forest feature importance from the benchmark export. '
        f'<span style="color:{ACCENT};">Highlighted</span> features are the top three in this snapshot.</p>'
        '<div class="demo-appendix">'
        '<div class="demo-appendix-head">'
        '<span class="demo-appendix-label">Appendix</span>'
        '<span class="demo-appendix-title">Adversarial robustness</span>'
        '<span class="demo-appendix-tag">Bonus</span>'
        "</div>"
        '<div class="demo-figure-rule" style="margin-bottom:16px;"></div>'
        '<details class="demo-appendix-details">'
        "<summary>Adversarial Robustness — Which features can a bot operator cheaply fake?</summary>"
        '<div class="demo-appendix-body">'
        f"{_resilience_table_html(resilience_rows)}"
        "</div></details></div>"
        "</article>"
        f"{_aside_html(top3)}"
        "</div>"
    )


def explainability_body_html() -> str:
    exists = SHAP_SUMMARY_RF_PATH.is_file()
    src = shap_image_data_uri(SHAP_SUMMARY_RF_PATH) if exists else ""
    return build_explainability_html(
        FEATURE_IMPORTANCE_ROWS,
        RESILIENCE_ROWS,
        shap_image_exists=exists,
        shap_image_src=src,
    )


def render_explainability() -> None:
    import streamlit as st

    st.markdown(explainability_body_html(), unsafe_allow_html=True)
