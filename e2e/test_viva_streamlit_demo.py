"""Playwright E2E checks for the Streamlit VIVA demo (structure + Tab 3 layout)."""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import Page, expect

from streamlit_demo.data import SHAP_SUMMARY_RF_PATH


pytestmark = pytest.mark.e2e


def _wait_demo_shell(page: Page) -> None:
    """Streamlit serves a shell first; wait for custom demo markup from st.markdown."""
    expect(page.locator(".demo-topnav")).to_be_visible(timeout=120_000)


def test_default_tab_model_arena(viva_page: Page, streamlit_base_url: str) -> None:
    viva_page.goto(f"{streamlit_base_url}/", wait_until="commit")
    _wait_demo_shell(viva_page)
    expect(viva_page.get_by_text("01 Model Arena", exact=True)).to_be_visible()
    expect(viva_page.get_by_text("Random Forest", exact=False).first).to_be_visible()


def test_explainability_tab(viva_page: Page, streamlit_base_url: str) -> None:
    viva_page.goto(f"{streamlit_base_url}/?tab=explainability", wait_until="commit")
    _wait_demo_shell(viva_page)
    expect(viva_page.get_by_text("02 Explainability", exact=True)).to_be_visible()
    expect(viva_page.locator(".demo-explain-article")).to_be_visible()
    expect(viva_page.get_by_text("Why did it predict", exact=False)).to_be_visible()


@pytest.mark.skipif(
    not SHAP_SUMMARY_RF_PATH.is_file(),
    reason="demo_assets/shap_summary_random_forest.png not present (placeholder UI tested in unit tests)",
)
def test_explainability_shap_summary_image_present(viva_page: Page, streamlit_base_url: str) -> None:
    """When demo_assets/shap_summary_random_forest.png exists, Tab 2 must render it (not placeholder)."""
    viva_page.goto(f"{streamlit_base_url}/?tab=explainability", wait_until="commit")
    _wait_demo_shell(viva_page)
    img = viva_page.locator("img.demo-shap-img")
    expect(img).to_be_visible(timeout=120_000)
    expect(img).to_have_attribute("alt", "SHAP summary")
def test_live_tab_controls_and_intro(viva_page: Page, streamlit_base_url: str) -> None:
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    expect(viva_page.get_by_text("Live prediction", exact=True)).to_be_visible()
    expect(viva_page.get_by_text("Section 03", exact=True)).to_be_visible()
    expect(viva_page.get_by_text("remaining 16", exact=False)).to_be_visible()
    expect(viva_page.get_by_text("24 features", exact=False)).to_be_visible()
    expect(viva_page.get_by_text("Model loaded", exact=True)).to_be_visible()
    expect(viva_page.get_by_text("Account parameters", exact=True)).to_be_visible()
    expect(viva_page.get_by_role("slider", name=re.compile(r"followers", re.I)).first).to_be_visible()
    expect(viva_page.get_by_role("button", name=re.compile(r"Run prediction"))).to_be_visible()


def test_live_tab_two_column_width_ratio(viva_page: Page, streamlit_base_url: str) -> None:
    """Guardrail: left (sliders) column should be materially narrower than the right column."""
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    viva_page.get_by_role("slider", name=re.compile(r"followers", re.I)).first.wait_for(state="visible")
    dims = viva_page.evaluate(
        """
        () => {
          const cols = document.querySelectorAll('[data-testid="column"]');
          if (cols.length >= 2) {
            const r0 = cols[0].getBoundingClientRect();
            const r1 = cols[1].getBoundingClientRect();
            return { w0: r0.width, w1: r1.width, mode: "column-testid" };
          }
          const blocks = [...document.querySelectorAll('[data-testid="stHorizontalBlock"]')];
          for (const b of blocks) {
            const br = b.getBoundingClientRect();
            if (br.width < 400) continue;
            const kids = [...b.children].filter((c) => c.getBoundingClientRect().width > 120);
            if (kids.length >= 2) {
              const r0 = kids[0].getBoundingClientRect();
              const r1 = kids[1].getBoundingClientRect();
              return { w0: r0.width, w1: r1.width, mode: "horizontal-children" };
            }
          }
          return null;
        }
        """
    )
    assert dims is not None, "Could not resolve two-column layout in the DOM"
    assert dims["w0"] > 80 and dims["w1"] > 80, dims
    assert dims["w0"] < dims["w1"] * 0.85, dims


def test_live_tab_gauge_iframe_present_after_run(viva_page: Page, streamlit_base_url: str) -> None:
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    viva_page.get_by_role("button", name=re.compile(r"Run prediction")).click()
    gauge = _locate_gauge_svg(viva_page)
    expect(gauge).to_be_visible(timeout=30_000)
    expect(gauge).to_have_attribute("viewBox", "0 0 280 155")


def _locate_gauge_svg(page: Page):
    """Return a locator for #gSvg inside the components iframe (not main frame)."""
    end = time.monotonic() + 45.0
    while time.monotonic() < end:
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            loc = frame.locator("#gSvg")
            try:
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        page.wait_for_timeout(200)
    raise AssertionError("Gauge #gSvg not found in any iframe after Run prediction")


def test_live_tab_adheres_to_denser_layout_spec(viva_page: Page, streamlit_base_url: str) -> None:
    """User asked for Tab 3 to feel less oversized — assert the implemented CSS/layout contract.

    Checks (see ``streamlit_demo/styles.py`` + ``live_prediction.py``):
    - Live page horizontal padding tightened to 32px (was 48px on generic shell).
    - Main two-column row ~30/70 (not ~34/66) and horizontal gap 16px.
    - Components gauge iframe height capped (Streamlit ``height=188``).
    - Verdict headline uses reduced typography (26px).
    """
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    viva_page.get_by_role("slider", name=re.compile(r"followers", re.I)).first.wait_for(state="visible")
    viva_page.wait_for_function(
        """() => {
          const cols = document.querySelectorAll('[data-testid="column"]');
          if (cols.length >= 2) return true;
          const blocks = [...document.querySelectorAll('[data-testid="stHorizontalBlock"]')];
          for (const b of blocks) {
            if (b.getBoundingClientRect().width < 400) continue;
            const kids = [...b.children].filter((c) => c.getBoundingClientRect().width > 120);
            if (kids.length >= 2) return true;
          }
          return false;
        }""",
        timeout=90_000,
    )

    pad = viva_page.evaluate(
        """
        () => {
          const bc = document.querySelector('[data-testid="stAppViewContainer"] .block-container')
            || document.querySelector('.block-container');
          if (!bc) return null;
          const s = getComputedStyle(bc);
          return { paddingLeft: s.paddingLeft, paddingRight: s.paddingRight };
        }
        """
    )
    assert pad is not None, "block-container not found"
    for side in ("paddingLeft", "paddingRight"):
        px = float(str(pad[side]).replace("px", ""))
        assert 28 <= px <= 36, (
            f"Expected live-tab horizontal padding ~32px (denser); got {side}={pad[side]!r}. "
            "If styles did not apply, confirm ?tab=live and streamlit_demo/styles.css rules."
        )

    layout = viva_page.evaluate(
        """
        () => {
          const cols = document.querySelectorAll('[data-testid="column"]');
          if (cols.length >= 2) {
            const r0 = cols[0].getBoundingClientRect();
            const r1 = cols[1].getBoundingClientRect();
            const parent = cols[0].closest('[data-testid="stHorizontalBlock"]');
            const gap = parent ? getComputedStyle(parent).gap : '';
            const total = r0.width + r1.width;
            const leftShare = total > 0 ? r0.width / total : 0;
            return { w0: r0.width, w1: r1.width, leftShare, gap, mode: 'column-testid' };
          }
          const blocks = [...document.querySelectorAll('[data-testid="stHorizontalBlock"]')];
          for (const b of blocks) {
            const br = b.getBoundingClientRect();
            if (br.width < 400) continue;
            const kids = [...b.children].filter((c) => c.getBoundingClientRect().width > 120);
            if (kids.length >= 2) {
              const r0 = kids[0].getBoundingClientRect();
              const r1 = kids[1].getBoundingClientRect();
              const gap = getComputedStyle(b).gap || '';
              const total = r0.width + r1.width;
              const leftShare = total > 0 ? r0.width / total : 0;
              return { w0: r0.width, w1: r1.width, leftShare, gap, mode: 'horizontal-children' };
            }
          }
          return null;
        }
        """
    )
    assert layout is not None, "Could not read column layout"
    assert 0.24 <= layout["leftShare"] <= 0.44, (
        f"Expected ~30/70 column split (left share ~0.30); got leftShare={layout['leftShare']:.3f} "
        f"w0={layout['w0']:.0f} w1={layout['w1']:.0f}"
    )
    gap_px = float(str(layout["gap"] or "0").replace("px", ""))
    assert 12 <= gap_px <= 22, (
        f"Expected horizontal block gap ~16px (tighter than old 28px); got gap={layout['gap']!r}"
    )

    viva_page.get_by_role("button", name=re.compile(r"Run prediction")).click()
    gauge = _locate_gauge_svg(viva_page)
    expect(gauge).to_be_visible(timeout=30_000)

    iframe_h = viva_page.evaluate(
        """
        () => {
          for (const f of document.querySelectorAll('iframe')) {
            let doc = null;
            try { doc = f.contentDocument; } catch (e) { continue; }
            if (!doc || !doc.querySelector('#gSvg')) continue;
            return f.getBoundingClientRect().height;
          }
          return null;
        }
        """
    )
    assert iframe_h is not None, "Gauge iframe not found in main document"
    assert iframe_h <= 160, (
        f"Expected compact gauge iframe (components height=142px + slack); got {iframe_h:.0f}px"
    )

    stacked = viva_page.evaluate(
        """
        () => {
          let iframeBottom = null;
          for (const f of document.querySelectorAll('iframe')) {
            try {
              if (f.contentDocument && f.contentDocument.querySelector('#gSvg')) {
                iframeBottom = f.getBoundingClientRect().bottom;
                break;
              }
            } catch (e) { continue; }
          }
          const verdict = document.querySelector('.demo-verdict-title');
          if (!iframeBottom || !verdict) return null;
          const vr = verdict.getBoundingClientRect();
          return { iframeBottom, verdictTop: vr.top, stacked: vr.top >= iframeBottom - 4 };
        }
        """
    )
    assert stacked is not None and stacked["stacked"], (
        f"Expected verdict below gauge (mockup layout); got {stacked}"
    )

    title_fs = viva_page.evaluate(
        """
        () => {
          const el = document.querySelector('.demo-verdict-title');
          if (!el) return null;
          return getComputedStyle(el).fontSize;
        }
        """
    )
    assert title_fs is not None, "Verdict title not in DOM after run"
    fs = float(str(title_fs).replace("px", ""))
    assert 22 <= fs <= 28, (
        f"Expected verdict title font-size ~26px; got {title_fs!r}"
    )


def test_live_tab_run_button_fully_visible(viva_page: Page, streamlit_base_url: str) -> None:
    """Run prediction must not be clipped at 1280x800 (idle and after prediction)."""
    viva_page.set_viewport_size({"width": 1280, "height": 800})
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    run_btn = viva_page.get_by_role("button", name=re.compile(r"Run prediction"))
    run_btn.wait_for(state="visible")

    def _live_controls_fully_visible() -> dict:
        return viva_page.evaluate(
            """
            () => {
              function clippedByHiddenOverflow(el) {
                const r = el.getBoundingClientRect();
                let p = el.parentElement;
                while (p && p !== document.documentElement) {
                  const s = getComputedStyle(p);
                  if (s.overflowY === 'hidden' || s.overflow === 'hidden') {
                    const pr = p.getBoundingClientRect();
                    if (r.bottom > pr.bottom + 0.5 || r.top < pr.top - 0.5) return true;
                  }
                  p = p.parentElement;
                }
                return false;
              }
              function fullyVisible(el) {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                const vh = window.innerHeight;
                if (r.height < 4) return false;
                if (r.top < 56 || r.bottom > vh - 4) return false;
                return !clippedByHiddenOverflow(el);
              }
              const btn = [...document.querySelectorAll('button')].find(
                (b) => /run prediction/i.test(b.textContent || '')
              );
              const toggles = [...document.querySelectorAll('[data-baseweb="button-group"]')];
              return {
                run_button: fullyVisible(btn),
                toggles: toggles.map((t) => fullyVisible(t)),
                no_page_scroll: document.documentElement.scrollHeight <= window.innerHeight + 2,
              };
            }
            """
        )

    before = _live_controls_fully_visible()
    assert before["run_button"], f"Run button clipped before run: {before}"
    assert len(before["toggles"]) >= 3 and all(before["toggles"]), (
        f"Toggle clipped before run: {before}"
    )
    assert before["no_page_scroll"], f"Page scrolls before run: {before}"
    run_btn.click()
    _locate_gauge_svg(viva_page)
    after = _live_controls_fully_visible()
    assert after["run_button"], f"Run button clipped after run: {after}"
    assert len(after["toggles"]) >= 3 and all(after["toggles"]), (
        f"Toggle clipped after run: {after}"
    )
    assert after["no_page_scroll"], f"Page scrolls after run: {after}"


def test_live_tab_post_run_fits_viewport(viva_page: Page, streamlit_base_url: str) -> None:
    """After run at 1280x800: mockup stack visible, no page scroll, gauge card taller than SHAP."""
    viva_page.set_viewport_size({"width": 1280, "height": 800})
    viva_page.goto(f"{streamlit_base_url}/?tab=live", wait_until="commit")
    _wait_demo_shell(viva_page)
    viva_page.get_by_role("button", name=re.compile(r"Run prediction")).click()
    _locate_gauge_svg(viva_page)
    expect(viva_page.locator(".demo-verdict-title")).to_be_visible(timeout=30_000)
    expect(viva_page.locator(".demo-shap-legend")).to_be_visible()

    layout = viva_page.evaluate(
        """
        () => {
          function mainRowEl() {
            const anchor = document.querySelector('.demo-live-main-row-anchor');
            const next = anchor?.nextElementSibling;
            if (next?.getAttribute('data-testid') === 'stHorizontalBlock') return next;
            const blocks = [...document.querySelectorAll('[data-testid="stHorizontalBlock"]')];
            for (const b of blocks) {
              const kids = [...b.children].filter((c) => c.getBoundingClientRect().width > 120);
              if (kids.length >= 2) return b;
            }
            return null;
          }
          function rightBorderWraps() {
            const row = mainRowEl();
            if (!row) return [];
            const kids = [...row.children].filter((c) => c.getBoundingClientRect().width > 80);
            const right = kids[kids.length - 1];
            if (!right) return [];
            const wraps = [...right.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]')];
            if (wraps.length) return wraps;
            return [...right.querySelectorAll('[data-testid="stLayoutWrapper"]')];
          }
          const mainRow = mainRowEl();
          const mainRowOy = mainRow ? getComputedStyle(mainRow).overflowY : '';
          const wrapHeights = rightBorderWraps().map((w) => w.getBoundingClientRect().height);
          const legend = document.querySelector('.demo-shap-legend');
          const lr = legend ? legend.getBoundingClientRect() : null;
          const legendVisible = lr && lr.top >= 56 && lr.bottom <= window.innerHeight - 4;
          return {
            scrollH: document.documentElement.scrollHeight,
            vh: window.innerHeight,
            mainRowOy,
            wrapHeights,
            gaugeTaller:
              wrapHeights.length >= 2 &&
              wrapHeights[0] >= wrapHeights[1] * 0.95,
            legendVisible,
          };
        }
        """
    )
    assert layout["scrollH"] <= layout["vh"] + 2, layout
    assert layout["mainRowOy"] not in ("auto", "scroll"), layout
    assert layout["gaugeTaller"], f"Gauge card should be taller than SHAP; got {layout}"
    assert layout["legendVisible"], layout
