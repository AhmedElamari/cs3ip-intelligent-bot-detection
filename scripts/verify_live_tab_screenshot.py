"""Capture Tab 3 screenshots and assert layout (idle + after run) at 1280x800."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "live_tab_screenshots"
URL = "http://localhost:8501/?tab=live"
VIEWPORT = {"width": 1280, "height": 800}

CHECK_JS = """
(afterRun) => {
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
  const verdict = document.querySelector('.demo-verdict-title');
  const shapLegend = document.querySelector('.demo-shap-legend');
  let gauge = false;
  for (const f of document.querySelectorAll('iframe')) {
    try {
      if (f.contentDocument && f.contentDocument.querySelector('#gSvg')) gauge = true;
    } catch (e) {}
  }
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
  const rightWraps = rightBorderWraps();
  const wrapHeights = rightWraps.map((w) => w.getBoundingClientRect().height);
  const gaugeTaller =
    !afterRun || window.innerWidth < 768 || wrapHeights.length < 2
      ? true
      : wrapHeights[0] >= wrapHeights[1] * 0.95;
  function derivedNotOverlappingDesc() {
    const derived = document.querySelector('.demo-derived-pair');
    if (!derived) return false;
    const dr = derived.getBoundingClientRect();
    const sliders = [...document.querySelectorAll('[data-testid="stSlider"]')];
    const desc = sliders.find((s) => /description_length/i.test(s.textContent || ''));
    if (!desc) return true;
    const label = desc.querySelector('label') || desc;
    const lr = label.getBoundingClientRect();
    return dr.bottom <= lr.top + 2;
  }
  function shapNamesNotWrapped() {
    const names = [...document.querySelectorAll('.demo-shap-name')];
    if (!names.length) return true;
    return names.every((n) => n.scrollHeight <= n.clientHeight + 1);
  }
  const derivedClear = derivedNotOverlappingDesc();
  const shapOk = afterRun ? shapNamesNotWrapped() : true;
  const ok =
    fullyVisible(btn) &&
    toggles.length >= 3 &&
    toggles.every((t) => fullyVisible(t)) &&
    document.documentElement.scrollHeight <= window.innerHeight + 2 &&
    derivedClear &&
    (afterRun ? fullyVisible(verdict) && fullyVisible(shapLegend) && gauge && gaugeTaller && shapOk : true) &&
    (window.innerWidth >= 768 ? mainRowOy !== 'auto' && mainRowOy !== 'scroll' : true);
  return {
    ok,
    afterRun,
    scrollH: document.documentElement.scrollHeight,
    vh: window.innerHeight,
    mainRowOy,
    gaugeTaller,
    wrapHeights,
    derivedClear,
    shapOk,
  };
}
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        for after_run, suffix in ((False, "idle"), (True, "after_run")):
            page.goto(URL, wait_until="commit", timeout=120_000)
            page.locator(".demo-topnav").wait_for(state="visible", timeout=120_000)
            page.get_by_role("slider").first.wait_for(state="visible", timeout=90_000)
            if after_run:
                page.get_by_role("button", name="Run prediction").click()
                end = time.monotonic() + 45.0
                while time.monotonic() < end:
                    found = any(
                        f.locator("#gSvg").count() > 0
                        for f in page.frames
                        if f != page.main_frame
                    )
                    if found:
                        break
                    time.sleep(0.2)
                page.locator(".demo-verdict-title").wait_for(state="visible", timeout=30_000)
            info = page.evaluate(CHECK_JS, after_run)
            png = OUT_DIR / f"1280x800_{suffix}.png"
            page.screenshot(path=str(png), full_page=False)
            print(f"{suffix}: {info} -> {png}")
            if not info.get("ok"):
                failed = True
        browser.close()

    proof = ROOT / "results" / "live_tab_verify_pass.png"
    after_png = OUT_DIR / "1280x800_after_run.png"
    if after_png.exists():
        proof.write_bytes(after_png.read_bytes())
        print(f"proof: {proof}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
