"""Session-scoped Streamlit server for Playwright E2E tests."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from playwright.sync_api import expect

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Streamlit hydrates after first paint; default expect timeout is too tight.
expect.set_options(timeout=90_000)


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _host, port = s.getsockname()
    s.close()
    return int(port)


def _wait_tcp(port: int, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Nothing listening on 127.0.0.1:{port} after {timeout_s}s")


def _start_streamlit_on_port(port: int) -> subprocess.Popen[str]:
    cmd = [
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
    ]
    return subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_streamlit(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def streamlit_base_url() -> Generator[str, None, None]:
    proc: subprocess.Popen[str] | None = None
    port = 0
    last_err = ""
    for _attempt in range(3):
        port = _pick_free_port()
        proc = _start_streamlit_on_port(port)
        try:
            _wait_tcp(port, timeout_s=45.0)
            break
        except TimeoutError:
            last_err = proc.stderr.read() if proc.stderr else ""
            _stop_streamlit(proc)
            proc = None
    if proc is None:
        raise TimeoutError(
            f"Streamlit did not listen on 127.0.0.1 after retries; last stderr tail:\n{last_err[-2000:]}"
        )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        _stop_streamlit(proc)
        err = proc.stderr.read() if proc.stderr else ""
        if proc.returncode not in (0, -15, 1) and err.strip():
            sys.stderr.write(f"[e2e] streamlit stderr tail:\n{err[-4000:]}\n")


@pytest.fixture
def viva_page(page, streamlit_base_url):
    """Playwright page with sane defaults for the demo."""
    page.set_default_timeout(90_000)
    page.set_default_navigation_timeout(90_000)
    yield page
