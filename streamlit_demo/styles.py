"""Global Streamlit theme + VIVA demo shell (prototype v3 tokens)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import streamlit as st_mod

GLOBAL_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
  --bg: #101218;
  --bg2: #16181f;
  --bg3: #1c1f28;
  --card: #181b23;
  --text: #e2e5ec;
  --text-mid: #949bb2;
  --text-dim: #6a7186;
  --accent: #6CC8BE;
  --green: #4DA87A;
  --amber: #C8A44A;
  --red: #C95444;
  --border: rgba(255,255,255,0.08);
  --border-accent: rgba(108,200,190,0.25);
}

html, body, [data-testid="stApp"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg); }

#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important;
}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

footer { visibility: hidden; }

/* Fixed demo nav clears this offset (all tabs). */
section.main,
[data-testid="stMain"] {
  padding-top: 56px !important;
}

/* Flush inner column padding (Streamlit adds padding above block-container). */
section.main > div,
[data-testid="stMain"] > div {
  padding-top: 0 !important;
}
.block-container {
  padding: 0 48px 64px 48px !important;
  max-width: 100% !important;
}
[data-testid="stVerticalBlock"] > div [data-testid="stMarkdownContainer"] {
  margin-top: 0 !important;
}

@keyframes demoDotPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

/* --- Demo shell (single-markdown tabs 1–2 only; min-height avoids double scroll with main padding) --- */
.demo-shell {
  background: var(--bg);
  min-height: calc(100vh - 56px);
  padding-top: 0;
  box-sizing: border-box;
}

.demo-topnav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 999;
  height: 56px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  margin: 0;
  padding: 0 48px;
  box-sizing: border-box;
}
.demo-topnav-inner {
  max-width: 1400px;
  margin: 0 auto;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}

.demo-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.demo-brand-mark {
  width: 36px; height: 36px; border-radius: 8px;
  background: linear-gradient(135deg, rgba(108,200,190,0.2), rgba(108,200,190,0.05));
  border: 1px solid var(--border-accent);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: var(--accent);
  flex-shrink: 0;
}
.demo-brand-title { font-family: 'DM Sans', sans-serif; font-size: 15px; font-weight: 600; color: var(--text); }
.demo-brand-sub {
  font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); margin-top: 2px;
}

.demo-tabs {
  display: flex; align-items: center; justify-content: center; gap: 4px;
  flex: 1;
}
.demo-tab-link {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-dim);
  text-decoration: none;
  padding: 8px 14px;
  border-radius: 6px;
  white-space: nowrap;
}
.demo-tab-link:hover { color: var(--text-mid); background: rgba(255,255,255,0.04); }
.demo-tab-active {
  color: var(--text) !important;
  font-weight: 600;
  background: var(--bg3) !important;
}
.demo-tab-disabled {
  cursor: default;
  opacity: 0.45;
  pointer-events: none;
}

.demo-status {
  display: flex; align-items: center; gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-mid);
  flex-shrink: 0;
}
.demo-status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green);
  animation: demoDotPulse 2s ease-in-out infinite;
}

/* --- Page body (Tab 1) --- */
.demo-page-body { padding: 12px 0 0 0; display: flex; flex-direction: column; gap: 28px; }

.demo-arena-intro {
  display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
  flex-wrap: wrap;
}
.demo-section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent);
}
.demo-h1 {
  font-family: 'DM Sans', sans-serif; font-size: 32px; font-weight: 700; line-height: 1.15;
  margin: 8px 0 0 0; letter-spacing: -0.02em; color: var(--text);
}
.demo-lede {
  font-family: 'DM Sans', sans-serif; font-size: 15px; color: var(--text-mid); margin-top: 8px;
  max-width: 480px; line-height: 1.55;
}
.demo-meta-right {
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--text-dim); line-height: 1.8;
}
.demo-meta-ratio { color: var(--amber); }

.demo-winner-card {
  background: var(--card);
  border: 1px solid var(--border-accent);
  border-radius: 10px;
  padding: 22px 28px;
  display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
}
.demo-winner-badge {
  width: 48px; height: 48px; border-radius: 10px;
  background: linear-gradient(135deg, rgba(108,200,190,0.15), rgba(108,200,190,0.04));
  border: 1px solid var(--border-accent);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: var(--accent);
  flex-shrink: 0;
}
.demo-winner-copy { flex: 1; min-width: 200px; }
.demo-winner-title {
  font-family: 'DM Sans', sans-serif; font-size: 18px; font-weight: 600; color: var(--text);
}
.demo-winner-title-accent { color: var(--accent); }
.demo-winner-sub {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); margin-top: 4px;
}
.demo-winner-chips { display: flex; gap: 16px; flex-wrap: wrap; }
.demo-winner-chip {
  text-align: center; padding: 10px 16px;
  background: rgba(255,255,255,0.02);
  border-radius: 8px; border: 1px solid var(--border);
}
.demo-winner-chip-val {
  font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600;
}
.demo-winner-chip-lbl {
  font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text-dim); margin-top: 3px;
}

.demo-metric-table {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.demo-metric-header-row {
  display: grid;
  grid-template-columns: 200px repeat(5, 1fr);
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.015);
}
.demo-metric-th {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-dim);
}
.demo-metric-row {
  display: grid;
  grid-template-columns: 200px repeat(5, 1fr);
  padding: 14px 24px;
  align-items: center;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  transition: background 0.15s;
}
.demo-metric-row:last-child { border-bottom: none; }
.demo-metric-row:hover { background: rgba(255,255,255,0.02) !important; }
.demo-metric-row--champion { background: rgba(108,200,190,0.03); }

.demo-model-name-wrap { display: flex; align-items: center; gap: 8px; }
.demo-pulse-dot {
  display: inline-block;
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent);
  animation: demoDotPulse 2s ease-in-out infinite;
}
.demo-model-name { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.demo-model-name--strong { font-weight: 600; color: var(--accent); }
.demo-model-name--muted { font-weight: 400; color: var(--text); }
.demo-best-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; color: var(--accent);
  padding: 2px 7px; border-radius: 4px;
  border: 1px solid rgba(108,200,190,0.27);
  background: rgba(108,200,190,0.07);
  letter-spacing: 0.08em; text-transform: uppercase;
}

.demo-metric-cell { display: flex; align-items: center; gap: 8px; padding-right: 16px; }
.demo-bar-track {
  flex: 1; height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden;
}
.demo-bar-fill { height: 100%; border-radius: 2px; }
.demo-metric-val {
  font-family: 'JetBrains Mono', monospace; font-size: 12px; min-width: 44px; text-align: right;
}
.demo-metric-val--best { font-weight: 600; }

.demo-f1-note {
  background: var(--card);
  border: 1px solid rgba(200,175,100,0.15);
  border-radius: 10px;
  padding: 18px 24px;
  display: flex; gap: 16px; align-items: flex-start;
}
.demo-f1-icon {
  font-family: 'JetBrains Mono', monospace; font-size: 16px; color: var(--amber); margin-top: 1px;
}
.demo-f1-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--amber);
}
.demo-f1-copy {
  font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--text-mid); line-height: 1.65; margin-top: 6px;
}

/* --- Tab 2 explainability --- */
.demo-explain-grid {
  display: grid;
  grid-template-columns: 1fr 220px;
  gap: 48px;
  padding-top: 12px;
  align-items: start;
}
@media (max-width: 900px) {
  .demo-explain-grid { grid-template-columns: 1fr; }
}

.demo-explain-article { min-width: 0; }

.demo-figure-head {
  display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
}
.demo-figure-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent);
}
.demo-figure-subtitle { font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--text-mid); }
.demo-figure-rule { height: 1px; background: var(--text); width: 100%; margin-bottom: 20px; opacity: 0.12; }

.demo-shap-slot { margin-bottom: 8px; }
.demo-shap-placeholder {
  width: 100%;
  aspect-ratio: 16/9;
  border: 1px dashed rgba(255,255,255,0.15);
  border-radius: 6px;
  background: rgba(255,255,255,0.02);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 24px;
}
.demo-shap-placeholder-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
  text-align: center;
  line-height: 1.7;
}
.demo-shap-placeholder-hint {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--text-dim);
  opacity: 0.5;
}
.demo-shap-img {
  width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block;
}
.demo-figure-caption {
  font-family: 'DM Sans', sans-serif; font-size: 12px; color: var(--text-dim); margin-top: 8px;
}

.demo-feature-bars { display: flex; flex-direction: column; gap: 10px; margin-top: 4px; }
.demo-feature-row {
  display: grid;
  grid-template-columns: 200px 1fr 56px;
  gap: 12px;
  align-items: center;
}
.demo-feature-name {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-mid);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.demo-feature-name--top { color: var(--accent); font-weight: 500; }
.demo-feature-val {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-mid); text-align: right;
}
.demo-feature-val--top { color: var(--accent); font-weight: 600; }

.demo-fig2-foot {
  font-family: 'DM Sans', sans-serif; font-size: 13px; color: var(--text-dim); font-style: italic; margin-top: 16px;
}

.demo-appendix { margin-top: 56px; }
.demo-appendix-head {
  display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
}
.demo-appendix-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--amber);
}
.demo-appendix-title { font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--text-mid); }
.demo-appendix-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; color: var(--amber);
  padding: 2px 7px; border-radius: 4px;
  border: 1px solid rgba(200,175,100,0.27);
  background: rgba(200,175,100,0.07);
  letter-spacing: 0.08em; text-transform: uppercase;
}

.demo-appendix-details {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.demo-appendix-details > summary {
  cursor: pointer;
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
  padding: 14px 18px;
  list-style: none;
}
.demo-appendix-details > summary::-webkit-details-marker { display: none; }
.demo-appendix-details[open] > summary { border-bottom: 1px solid var(--border); }
.demo-appendix-body { padding: 0 18px 18px 18px; }

.demo-resilience-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  margin-top: 12px;
}
.demo-resilience-table th {
  text-align: left;
  color: var(--text-dim);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 10px 12px 8px 0;
  border-bottom: 1px solid var(--border);
}
.demo-resilience-table td {
  padding: 12px 12px 12px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--text-mid);
  vertical-align: top;
}
.demo-resilience-table tr:last-child td { border-bottom: none; }
.demo-resilience-feature { color: var(--text); font-weight: 500; }

.demo-aside-column { padding-top: 140px; }
@media (max-width: 900px) { .demo-aside-column { padding-top: 24px; } }
.demo-aside-note {
  position: sticky;
  top: 72px;
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  color: var(--text-dim);
  line-height: 1.65;
  border-left: 2px solid var(--accent);
  padding-left: 16px;
}
.demo-aside-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent);
}

/* Tab 3 live prediction — responsive, one viewport (no zoom) */
.block-container:has(.demo-live-page-body) {
  padding-left: 32px !important;
  padding-right: 32px !important;
  padding-bottom: 16px !important;
  --demo-nav-h: 56px;
  --demo-live-intro-h: clamp(56px, 12dvh, 88px);
  --demo-live-pad: 16px;
  --demo-live-main-h: calc(100dvh - var(--demo-nav-h) - var(--demo-live-intro-h) - var(--demo-live-pad));
}
@supports not (height: 100dvh) {
  .block-container:has(.demo-live-page-body) {
    --demo-live-main-h: calc(100vh - var(--demo-nav-h) - var(--demo-live-intro-h) - var(--demo-live-pad));
  }
}
.demo-live-page-body {
  padding-top: 6px;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlock"] {
  gap: 0.4rem !important;
}
.demo-live-intro-row {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  max-height: var(--demo-live-intro-h);
  overflow: hidden;
}
.block-container:has(.demo-live-page-body) .demo-live-h1 {
  font-size: clamp(22px, 3.2vw, 28px) !important;
  margin: 4px 0 0 0 !important;
  line-height: 1.15 !important;
}
.demo-live-lede {
  max-width: 480px;
  font-size: clamp(13px, 1.8vw, 15px);
  line-height: 1.45;
  margin: 6px 0 0 0;
}
.demo-live-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-dim);
  text-align: right;
  line-height: 1.6;
  padding-bottom: 2px;
}
.demo-live-main-row-anchor {
  display: none;
}
/* Main two-column row (anchor = stable hook after intro) */
.block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"],
.block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]),
.block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stLayoutWrapper"]) {
  gap: 16px !important;
  align-items: stretch !important;
  max-height: var(--demo-live-main-h) !important;
  box-sizing: border-box;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] {
  padding: 12px 14px !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
  min-height: 0;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"],
.block-container:has(.demo-live-page-body) [data-testid="stLayoutWrapper"] [data-testid="stVerticalBlock"] {
  flex: 1 1 auto;
  min-height: 0;
}
/* Streamlit 1.3x border containers use stLayoutWrapper (not stVerticalBlockBorderWrapper) */
.block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:first-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
  min-height: 0;
  box-sizing: border-box;
}
@media (min-width: 768px) {
  .block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"],
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]),
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stLayoutWrapper"]) {
    overflow: hidden !important;
  }
  .block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:first-child [data-testid="stVerticalBlockBorderWrapper"],
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div:first-child [data-testid="stVerticalBlockBorderWrapper"],
  .block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:first-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
    max-height: var(--demo-live-main-h);
    height: 100%;
    overflow: visible !important;
  }
  .block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child,
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div:last-child {
    max-height: var(--demo-live-main-h);
    overflow: hidden !important;
    box-sizing: border-box;
  }
  .block-container:has(.demo-live-page-body) [data-testid="column"]:last-child [data-testid="stVerticalBlock"],
  .block-container:has(.demo-live-page-body) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlock"] {
    max-height: var(--demo-live-main-h);
    overflow: hidden !important;
    box-sizing: border-box;
  }
}
@media (max-width: 767px) {
  .block-container:has(.demo-live-page-body) {
    --demo-live-intro-h: clamp(48px, 10dvh, 72px);
    padding-left: 16px !important;
    padding-right: 16px !important;
  }
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {
    flex-direction: column !important;
    max-height: var(--demo-live-main-h) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    -webkit-overflow-scrolling: touch;
  }
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div {
    width: 100% !important;
    flex: 0 0 auto !important;
    max-width: 100% !important;
    max-height: none !important;
  }
  .block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] {
    max-height: none;
    overflow-y: visible;
  }
  .block-container:has(.demo-live-page-body) [data-testid="column"]:last-child [data-testid="stVerticalBlock"],
  .block-container:has(.demo-live-page-body) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div:last-child {
    max-height: none;
    overflow-y: visible;
  }
  .demo-live-meta { display: none; }
  .demo-shap-waterfall-card--compact {
    max-height: min(160px, 24dvh);
  }
  .block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] {
    position: sticky;
    bottom: 0;
    z-index: 2;
    background: var(--card);
    padding-top: 6px;
    margin-top: auto;
  }
}
@media (max-width: 767px) {
  .block-container:has(.demo-live-results-panel) {
    --demo-live-intro-h: clamp(44px, 8dvh, 64px);
  }
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child [data-testid="stVerticalBlock"],
  .block-container:has(.demo-live-results-panel) [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > div:last-child {
    max-height: min(50dvh, 430px) !important;
  }
  .block-container:has(.demo-live-results-panel) .demo-verdict-title {
    font-size: 18px !important;
  }
  .block-container:has(.demo-live-results-panel) .demo-verdict-block {
    min-height: 56px;
  }
  .block-container:has(.demo-live-results-panel) .demo-shap-waterfall-card--compact {
    max-height: min(140px, 20dvh);
    padding: 8px 10px;
  }
}
@media (max-height: 900px) {
  .block-container:has(.demo-live-page-body) .demo-live-h1 {
    font-size: 22px !important;
    margin-top: 2px !important;
  }
  .demo-live-lede { font-size: 13px; margin-top: 4px; }
  .demo-live-intro-row { margin-bottom: 4px; }
}
.block-container:has(.demo-live-page-body) [data-testid="stSlider"] {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  margin-bottom: 0 !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stSlider"] > div {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  min-height: 0 !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stSlider"] [data-testid="stThumbValue"] {
  display: none;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
  gap: 0.35rem !important;
}
.block-container:has(.demo-live-page-body)
  [data-testid="stElementContainer"]:has(.demo-derived-pair) {
  margin-bottom: 8px !important;
}
.block-container:has(.demo-live-page-body)
  [data-testid="stElementContainer"]:has(.demo-derived-pair)
  + [data-testid="stElementContainer"] {
  margin-top: 6px !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="slider"] {
  height: 14px !important;
  min-height: 14px !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]:has([data-baseweb="button-group"]) {
  gap: 8px !important;
  margin-top: 2px !important;
  margin-bottom: 2px !important;
  align-items: flex-end !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]:has([data-baseweb="button-group"]) > div {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-end !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]:has([data-baseweb="button-group"]) [data-testid="stWidgetLabel"],
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]:has([data-baseweb="button-group"]) label {
  min-height: 2.4em !important;
  display: flex !important;
  align-items: flex-end !important;
  margin-bottom: 2px !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
  margin-top: 6px !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]:has([data-baseweb="button-group"]) [data-baseweb="button-group"] button {
  padding: 4px 6px !important;
  min-height: 28px !important;
  font-size: 10px !important;
}
.block-container:has(.demo-live-page-body) [data-testid="stSlider"] label {
  line-height: 1.25 !important;
  margin-bottom: 2px !important;
}
.block-container:has(.demo-live-page-body) [data-baseweb="button-group"] {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
}

.demo-derived-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 6px 8px;
  background: rgba(108,200,190,0.04);
  border-radius: 8px;
  border: 1px solid rgba(108,200,190,0.12);
  margin: 0;
}
.demo-derived-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--text-dim);
  letter-spacing: 0.06em;
}
.demo-derived-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  font-weight: 600;
  color: var(--accent);
  margin-top: 2px;
}

.demo-live-page-body .stSlider label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim) !important;
}
.demo-live-page-body [data-testid="stSlider"] [data-testid="stTickBarMin"],
.demo-live-page-body [data-testid="stSlider"] [data-testid="stTickBarMax"] {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--accent);
}

.demo-live-page-body [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
}

.demo-live-page-body [data-testid="stHorizontalBlock"] [data-baseweb="button-group"] button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
}

.demo-live-page-body [data-testid="stButton"] > button[kind="primary"] {
  width: 100%;
  margin-top: 4px;
  padding: 9px !important;
  border-radius: 8px !important;
}

.demo-result-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.demo-idle-card {
  padding: 28px 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: var(--demo-live-main-h);
  max-height: var(--demo-live-main-h);
  box-sizing: border-box;
}
.demo-idle-text {
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  color: var(--text-dim);
  line-height: 1.6;
  text-align: center;
  margin: 0;
}

/* Results: stacked cards (gauge+verdict, then SHAP) — VIVA mockup */
.demo-live-results-panel {
  width: 100%;
  box-sizing: border-box;
}
.block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"],
.block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
  display: grid !important;
  grid-template-rows: 0 minmax(0, 58fr) minmax(0, 42fr) !important;
  gap: 12px !important;
  height: var(--demo-live-main-h) !important;
  max-height: var(--demo-live-main-h) !important;
  min-height: 0 !important;
  box-sizing: border-box;
  overflow: hidden !important;
}
.block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
.block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
}
@media (min-width: 768px) {
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"],
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
    overflow: hidden !important;
  }
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlockBorderWrapper"]:first-of-type,
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child [data-testid="stVerticalBlockBorderWrapper"]:first-of-type {
    flex: 1 1 58%;
    min-height: 0;
    max-height: none !important;
    overflow: hidden !important;
    padding: 14px 16px 10px 16px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlockBorderWrapper"]:last-of-type,
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child [data-testid="stVerticalBlockBorderWrapper"]:last-of-type {
    flex: 1 1 42%;
    min-height: 0;
    max-height: none !important;
    overflow: hidden !important;
    padding: 12px 14px !important;
    display: flex;
    flex-direction: column;
  }
  .block-container:has(.demo-live-results-panel) .demo-shap-waterfall-card--compact {
    flex: 1 1 auto;
    min-height: 0;
    max-height: none !important;
    overflow: hidden !important;
    margin-top: 0;
    display: flex;
    flex-direction: column;
  }
  .block-container:has(.demo-live-results-panel) .demo-shap-rows {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }
  /* stLayoutWrapper build: row 2 = gauge, row 3 = SHAP (grid on parent) */
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
    min-height: 0 !important;
    max-height: none !important;
    overflow: hidden !important;
  }
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type,
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: calc(var(--demo-live-main-h) * 0.55) !important;
  }
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] > div:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-of-type,
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-of-type {
    max-height: calc(var(--demo-live-main-h) * 0.42) !important;
  }
}
.block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child [data-testid="stIFrame"],
.block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] [data-testid="stIFrame"] {
  margin: 0 auto !important;
  width: 100% !important;
  max-width: 300px !important;
}
.block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child .demo-verdict-block,
.block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] .demo-verdict-block {
  width: 100%;
  margin-top: 2px;
}
.demo-verdict-block {
  text-align: center;
  margin: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 0;
  padding: 0 8px 4px 8px;
}
.demo-verdict-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.01em;
  line-height: 1.2;
}
.demo-verdict-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
}

.demo-shap-waterfall-card {
  padding: 16px 18px;
}
.demo-shap-waterfall-card--compact {
  padding: 12px 14px;
  margin-top: 4px;
  box-sizing: border-box;
  flex-shrink: 1;
}
@media (max-width: 767px) {
  .demo-shap-waterfall-card--compact {
    max-height: min(160px, 24dvh);
    overflow-y: auto;
  }
}
@media (max-height: 900px) and (min-width: 768px) {
  .block-container:has(.demo-live-results-panel) [data-testid="column"]:last-child [data-testid="stIFrame"],
  .block-container:has(.demo-live-results-panel) .demo-live-main-row-anchor ~ [data-testid="stHorizontalBlock"] [data-testid="stIFrame"] {
    height: 128px !important;
  }
  .demo-shap-waterfall-card--compact .demo-shap-rows {
    gap: 4px;
  }
  .demo-shap-waterfall-card--compact .demo-shap-row {
    gap: 8px;
  }
  .demo-shap-waterfall-card--compact .demo-shap-name {
    font-size: 10px;
  }
}
.demo-shap-waterfall-card--compact .demo-shap-rows {
  margin-top: 8px;
  gap: 6px;
}
.demo-shap-rows {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.demo-shap-empty {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  color: var(--text-dim);
  margin-top: 16px;
}
.demo-shap-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.demo-shap-name {
  width: 210px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-mid);
  text-align: right;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.demo-shap-track {
  flex: 1;
  position: relative;
  height: 18px;
}
.demo-shap-center-line {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(255,255,255,0.08);
}
.demo-shap-fill {
  position: absolute;
  top: 1px;
  bottom: 1px;
  border-radius: 2px;
  transition: width 0.4s;
}
.demo-shap-fill--pos { left: 50%; }
.demo-shap-fill--neg { right: 50%; }
.demo-shap-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  width: 48px;
  text-align: right;
  font-weight: 500;
  flex-shrink: 0;
}
.demo-shap-legend {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-dim);
}
.demo-shap-legend-red { color: var(--red); }
.demo-shap-legend-green { color: var(--green); }

.demo-artifact-missing {
  padding: 24px;
}
.demo-artifact-missing-text {
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  color: var(--text-mid);
  margin-top: 12px;
}
.demo-artifact-cmd {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--accent);
  background: rgba(255,255,255,0.03);
  padding: 12px;
  border-radius: 8px;
  border: 1px dashed rgba(255,255,255,0.12);
  overflow-x: auto;
  margin-top: 12px;
}

/* Streamlit embeds (gauge iframe): remove extra vertical gap */
[data-testid="stIFrame"] {
  margin: 0 !important;
}

/* Widgets we may still use sparingly */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
}
[data-testid="stButton"] > button {
  background: rgba(108,200,190,0.1) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  font-family: 'DM Sans', sans-serif;
  font-weight: 600;
  border-radius: 8px;
}
"""


def apply_theme(st: Any) -> None:
    st.markdown(f"<style>{GLOBAL_THEME_CSS}</style>", unsafe_allow_html=True)
