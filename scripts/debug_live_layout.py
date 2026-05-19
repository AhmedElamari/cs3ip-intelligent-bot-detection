"""Debug Tab 3 layout at multiple viewports (no page scroll; controls + results visible)."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "live_tab_screenshots"

VIEWPORTS = [
    ("1280x800", 1280, 800),
    ("390x844", 390, 844),
    ("1366x768", 1366, 768),
    ("1920x1080", 1920, 1080),
]

VISIBILITY_JS = """
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
  function intersectsViewport(el, topMin = 56) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight;
    if (r.height < 4 || r.width < 4) return false;
    if (r.bottom <= topMin || r.top >= vh - 4) return false;
    return true;
  }
  function fullyVisible(el, topMin = 56) {
    if (!intersectsViewport(el, topMin)) return false;
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
  function mainLeftCol() {
    const blocks = [...document.querySelectorAll('[data-testid="stHorizontalBlock"]')];
    for (const b of blocks) {
      if (!b.querySelector('[data-testid="stVerticalBlockBorderWrapper"]')) continue;
      const kids = [...b.children].filter((c) => c.getBoundingClientRect().width > 80);
      if (kids.length) return kids[0];
    }
    return null;
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
  let gaugeTaller = true;
  if (afterRun && window.innerWidth >= 768 && wrapHeights.length >= 2) {
    gaugeTaller = wrapHeights[0] >= wrapHeights[1] * 0.95;
  }
  const desktopNoRowScroll =
    window.innerWidth < 768 || (mainRowOy !== 'auto' && mainRowOy !== 'scroll');
  const checks = {
    run_button: intersectsViewport(btn),
    toggle_groups: toggles.map((t) => intersectsViewport(t)),
    no_page_scroll: document.documentElement.scrollHeight <= window.innerHeight + 2,
    desktop_no_row_scroll: desktopNoRowScroll,
    verdict: afterRun ? fullyVisible(verdict) : true,
    shap_legend: afterRun
      ? (window.innerHeight <= 850 ? intersectsViewport(shapLegend) : fullyVisible(shapLegend))
      : true,
    gauge: afterRun ? gauge : true,
    gauge_taller_than_shap: gaugeTaller,
  };
  const ok =
    checks.run_button &&
    toggles.length >= 3 &&
    checks.toggle_groups.every(Boolean) &&
    checks.no_page_scroll &&
    checks.desktop_no_row_scroll &&
    checks.verdict &&
    checks.shap_legend &&
    checks.gauge &&
    checks.gauge_taller_than_shap;
  return { ok, checks, vh: window.innerHeight, scrollH: document.documentElement.scrollHeight, wrapHeights };
}
"""


def wait_tcp(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Nothing listening on 127.0.0.1:{port} after {timeout_s}s")


def pick_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def check_viewport(page: Page, url: str, label: str, w: int, h: int) -> list[dict]:
    out: list[dict] = []
    page.set_viewport_size({"width": w, "height": h})
    for after_run, suffix in ((False, "idle"), (True, "after_run")):
        page.goto(f"{url}/?tab=live", wait_until="commit", timeout=120_000)
        page.locator(".demo-topnav").wait_for(timeout=120_000)
        page.get_by_role("slider").first.wait_for(timeout=90_000)
        page.wait_for_function(
            "() => document.querySelectorAll('[data-baseweb=\"button-group\"]').length >= 3",
            timeout=60_000,
        )
        if after_run:
            page.get_by_role("button", name="Run prediction").click()
            end = time.monotonic() + 45.0
            while time.monotonic() < end:
                found = False
                for frame in page.frames:
                    if frame == page.main_frame:
                        continue
                    try:
                        if frame.locator("#gSvg").count() > 0:
                            found = True
                            break
                    except Exception:
                        pass
                if found:
                    break
                page.wait_for_timeout(200)
            page.locator(".demo-verdict-title").wait_for(state="visible", timeout=30_000)
            if w < 768:
                for sel in (".demo-verdict-title", ".demo-shap-waterfall-card"):
                    loc = page.locator(sel)
                    if loc.count():
                        loc.first.scroll_into_view_if_needed()
                page.wait_for_timeout(400)
        info = page.evaluate(VISIBILITY_JS, after_run)
        info["viewport"] = label
        info["state"] = suffix
        png = OUT_DIR / f"{label}_{suffix}.png"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(png), full_page=False)
        info["screenshot"] = str(png)
        out.append(info)
    return out


def main() -> None:
    port = pick_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_tcp(port)
    url = f"http://127.0.0.1:{port}"
    failed = False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            all_info: list[dict] = []
            for label, w, h in VIEWPORTS:
                all_info.extend(check_viewport(page, url, label, w, h))
            browser.close()
        for info in all_info:
            print(info)
            if not info.get("ok"):
                failed = True
        legacy = ROOT / "results" / "live_tab_debug.png"
        proof = ROOT / "results" / "live_tab_verify_pass.png"
        idle_1280 = OUT_DIR / "1280x800_idle.png"
        after_1280 = OUT_DIR / "1280x800_after_run.png"
        if idle_1280.exists():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(idle_1280.read_bytes())
            print(f"legacy screenshot: {legacy}")
        if after_1280.exists():
            proof.write_bytes(after_1280.read_bytes())
            print(f"proof screenshot: {proof}")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
