"""Streamlit entry: VIVA demo (Tabs 1-3, query-param navigation)."""

from __future__ import annotations

import streamlit as st

from streamlit_demo.data import DEMO_DATASET_META, LIVE_PREDICTOR_JOBLIB_PATH, MODEL_ARENA_ROWS
from streamlit_demo.explainability import explainability_body_html
from streamlit_demo.layout import active_tab_from_session, build_header_html
from streamlit_demo.live_prediction import (
    cached_live_predictor,
    live_header_hint_from_path,
    live_header_status,
    render_live_prediction,
)
from streamlit_demo.model_arena import arena_provenance_footnote_html, build_model_arena_html
from streamlit_demo.styles import apply_theme


def main() -> None:
    st.set_page_config(
        page_title="Bot Detection — VIVA Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme(st)
    tab = active_tab_from_session(st)
    if tab == "live":
        live_bundle = cached_live_predictor(str(LIVE_PREDICTOR_JOBLIB_PATH.resolve()))
        status = live_header_status(live_bundle[0], live_bundle[1])
    else:
        live_bundle = None
        status = live_header_hint_from_path(LIVE_PREDICTOR_JOBLIB_PATH)
    header = build_header_html(tab, status)
    if tab == "live":
        # Do not split <div class="demo-shell"> across Streamlit blocks — each block is a
        # sibling, so an opened shell never wraps widgets and causes a full-viewport layout gap.
        st.markdown(header, unsafe_allow_html=True)
        render_live_prediction(LIVE_PREDICTOR_JOBLIB_PATH, bundle=live_bundle)
        return
    body = (
        build_model_arena_html(
            MODEL_ARENA_ROWS,
            DEMO_DATASET_META,
            footnote_html=arena_provenance_footnote_html(),
        )
        if tab == "arena"
        else explainability_body_html()
    )
    st.markdown(
        f'<div class="demo-shell">{header}{body}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
