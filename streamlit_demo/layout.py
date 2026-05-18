"""Custom demo shell: top nav and query-param tab keys."""

from __future__ import annotations

import html
from typing import Any


def normalize_tab(raw: str | None) -> str:
    if raw is None or not str(raw).strip():
        return "arena"
    key = str(raw).strip().lower()
    if key in ("arena", "model-arena", "tab1", "1"):
        return "arena"
    if key in ("explainability", "explain", "tab2", "2"):
        return "explainability"
    if key in ("live", "prediction", "tab3", "3"):
        return "live"
    return "arena"


def first_query_param(st: Any, key: str) -> str | None:
    v = st.query_params.get(key)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else None
    return str(v)


def active_tab_from_session(st: Any) -> str:
    return normalize_tab(first_query_param(st, "tab"))


def build_header_html(active_tab: str, status_text: str = "Demo") -> str:
    """Top nav matching v3: brand left, tabs center, status right."""
    arena_cls = "demo-tab-link demo-tab-active" if active_tab == "arena" else "demo-tab-link"
    exp_cls = (
        "demo-tab-link demo-tab-active" if active_tab == "explainability" else "demo-tab-link"
    )
    live_cls = "demo-tab-link demo-tab-active" if active_tab == "live" else "demo-tab-link"
    return (
        '<header class="demo-topnav" data-demo-active="'
        + html.escape(active_tab, quote=True)
        + '">'
        '<div class="demo-topnav-inner">'
        '<div class="demo-brand">'
        '<span class="demo-brand-mark">BD</span>'
        '<div class="demo-brand-text">'
        '<div class="demo-brand-title">Bot Detection</div>'
        '<div class="demo-brand-sub">TwiBot-20 · Dissertation Demo</div>'
        "</div></div>"
        '<nav class="demo-tabs" aria-label="Demo sections">'
        f'<a href="?tab=arena" class="{arena_cls}">01 Model Arena</a>'
        f'<a href="?tab=explainability" class="{exp_cls}">02 Explainability</a>'
        f'<a href="?tab=live" class="{live_cls}">03 Live Prediction</a>'
        "</nav>"
        '<div class="demo-status">'
        '<span class="demo-status-dot" aria-hidden="true"></span>'
        f'<span class="demo-status-text">{html.escape(status_text)}</span>'
        "</div></div></header>"
    )


def render_header(st: Any, active_tab: str) -> None:
    st.markdown(build_header_html(active_tab), unsafe_allow_html=True)
