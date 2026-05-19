"""Capture Tab 3 screenshots at multiple viewports (idle + after Run)."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "live_tab_screenshots"

VIEWPORTS: list[tuple[str, int, int]] = [
    ("1280x800", 1280, 800),
    ("390x844", 390, 844),
    ("1366x768", 1366, 768),
    ("1920x1080", 1920, 1080),
]

CHECK_JS = """
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
  function inViewport(el, topMin = 56) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight;
    if (r.height < 4 || r.width < 4) return false;
    if (r.bottom <= topMin || r.top >= vh - 4) return false;
    return true;
  }
  const btn = [...document.querySelectorAll('button')].find(
    (b) => /run prediction/i.test(b.textContent || '')
  );
  const toggles = [...document.querySelectorAll('[data-baseweb="button-group"]')];
  const verdict = document.querySelector('.demo-verdict-title');
  const shap = document.querySelector('.demo-shap-waterfall-card');
  let gauge = false;
  for (const f of document.querySelectorAll('iframe')) {
    try {
      if (f.contentDocument && f.contentDocument.querySelector('#gSvg')) {
        gauge = true;
        break;
      }
    } catch (e) { /* cross-origin */ }
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
  const leftCol = mainLeftCol();
  const zoomLeft = (() => {
    if (!leftCol) return '1';
    const z = getComputedStyle(leftCol).zoom;
    return z === 'normal' ? '1' : z;
  })();
  return {
    run_button: inViewport(btn),
    toggles: toggles.map((t) => inViewport(t)),
    verdict: inViewport(verdict),
    shap: inViewport(shap),
    gauge_present: gauge,
    no_page_scroll: document.documentElement.scrollHeight <= window.innerHeight + 2,
    left_zoom: zoomLeft,
  };
}
"""


def pick_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def wait_tcp(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Nothing listening on 127.0.0.1:{port} after {timeout_s}s")


def wait_live_ready(page: Page) -> None:
    page.locator(".demo-topnav").wait_for(timeout=120_000)
    page.get_by_role("slider").first.wait_for(state="visible", timeout=90_000)


def wait_gauge(page: Page, timeout_ms: int = 45_000) -> None:
    end = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < end:
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            try:
                if frame.locator("#gSvg").count() > 0:
                    return
            except Exception:
                pass
        page.wait_for_timeout(200)
    raise RuntimeError("Gauge #gSvg not found after Run prediction")


def run_capture(
    page: Page,
    base_url: str,
    label: str,
    width: int,
    height: int,
    *,
    after_run: bool,
) -> dict:
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"{base_url}/?tab=live", wait_until="commit", timeout=120_000)
    wait_live_ready(page)
    if after_run:
        page.get_by_role("button", name="Run prediction").click()
        wait_gauge(page)
        page.locator(".demo-verdict-title").wait_for(state="visible", timeout=30_000)
        if width < 768:
            for sel in (".demo-verdict-title", ".demo-shap-waterfall-card"):
                loc = page.locator(sel)
                if loc.count():
                    loc.first.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
    info = page.evaluate(CHECK_JS)
    suffix = "after_run" if after_run else "idle"
    path = OUT_DIR / f"{label}_{suffix}.png"
    page.screenshot(path=str(path), full_page=False)
    info["screenshot"] = str(path)
    info["viewport"] = label
    info["state"] = suffix
    ok = info["no_page_scroll"] and info["run_button"]
    if after_run:
        ok = ok and info["gauge_present"] and info["verdict"] and info["shap"]
    else:
        ok = ok and len(info["toggles"]) >= 3 and all(info["toggles"])
    if float(info.get("left_zoom", "1")) < 0.99:
        ok = False
        info["zoom_rejected"] = True
    info["ok"] = ok
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture live tab screenshots.")
    parser.add_argument("--url", default="", help="Base URL (default: start Streamlit)")
    parser.add_argument("--port", type=int, default=0, help="Port when starting Streamlit")
    parser.add_argument(
        "--viewports",
        default="all",
        help="Comma-separated labels or 'all' (default: all)",
    )
    parser.add_argument(
        "--states",
        default="idle,after_run",
        help="Comma-separated: idle, after_run",
    )
    args = parser.parse_args()
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    wanted = (
        {v[0] for v in VIEWPORTS}
        if args.viewports.strip().lower() == "all"
        else {s.strip() for s in args.viewports.split(",")}
    )
    viewports = [v for v in VIEWPORTS if v[0] in wanted]
    if not viewports:
        print("No matching viewports selected. Use --viewports=all or valid labels.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = None
    port = args.port or pick_port()
    base_url = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{port}"

    if not args.url:
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

    results: list[dict] = []
    failed = False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            for label, w, h in viewports:
                if "idle" in states:
                    r = run_capture(page, base_url, label, w, h, after_run=False)
                    results.append(r)
                    print(r)
                    if not r.get("ok"):
                        failed = True
                if "after_run" in states:
                    r = run_capture(page, base_url, label, w, h, after_run=True)
                    results.append(r)
                    print(r)
                    if not r.get("ok"):
                        failed = True
            if not failed and "after_run" in states:
                golden = ROOT / "results" / "live_tab_verify_pass.png"
                src = OUT_DIR / f"{viewports[0][0]}_after_run.png"
                if src.exists():
                    golden.parent.mkdir(parents=True, exist_ok=True)
                    golden.write_bytes(src.read_bytes())
                    print(f"golden: {golden.resolve()}")
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=15)

    print(f"\nScreenshots: {OUT_DIR.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
