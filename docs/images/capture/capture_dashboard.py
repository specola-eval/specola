"""Re-capture the two README dashboard screenshots.

Serves a saved recording with `specola view`, drives headless Chrome over the
DevTools protocol (websockets is already in the robocasa env; Chrome ships the
protocol, so nothing new is installed), and writes JPEGs at the committed
1512x790 geometry.

Usage:
  PYTHONNOUSERSITE=1 python capture_dashboard.py \
      --recording results/<run_id>/recording_000 --out-dir docs/images
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

VIEWPORT = (1512, 790)
SEEK_STEP = 900


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 120.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry until the server is up
            last = exc
            time.sleep(0.5)
    raise SystemExit("timed out waiting for {}: {}".format(url, last))


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    async def send(self, method, **params):
        import json as _json

        self._id += 1
        message_id = self._id
        await self.ws.send(
            _json.dumps({"id": message_id, "method": method,
                         "params": params})
        )
        while True:
            raw = _json.loads(await self.ws.recv())
            if raw.get("id") == message_id:
                if "error" in raw:
                    raise SystemExit("{} failed: {}".format(method,
                                                            raw["error"]))
                return raw.get("result", {})

    async def evaluate(self, expression, await_promise=False):
        result = await self.send(
            "Runtime.evaluate", expression=expression,
            returnByValue=True, awaitPromise=await_promise,
        )
        if "exceptionDetails" in result:
            raise SystemExit("JS error: {}".format(
                result["exceptionDetails"].get("text")
            ))
        return result["result"].get("value")

    async def poll(self, expression, timeout=90.0, label=""):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await self.evaluate(expression):
                return True
            await asyncio.sleep(0.25)
        raise SystemExit("timed out waiting for {}: {}".format(
            label or expression, expression
        ))

    async def screenshot(self, path: Path):
        result = await self.send(
            "Page.captureScreenshot", format="jpeg", quality=90,
            captureBeyondViewport=False,
        )
        path.write_bytes(base64.b64decode(result["data"]))


async def capture(page_url: str, out_dir: Path, chrome: str, seek_step: int):
    import websockets

    profile = Path(tempfile.mkdtemp(prefix="specola-shot-"))
    port = free_port()
    proc = None
    try:
        proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--remote-debugging-port={}".format(port),
                "--user-data-dir={}".format(profile),
                "--window-size={},{}".format(*VIEWPORT),
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        targets = json.loads(
            wait_http("http://127.0.0.1:{}/json/list".format(port), 60)
        )
        ws_url = next(
            t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"
        )
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send(
                "Emulation.setDeviceMetricsOverride",
                width=VIEWPORT[0], height=VIEWPORT[1],
                deviceScaleFactor=1, mobile=False,
            )
            await cdp.send("Page.navigate", url=page_url)
            await cdp.poll(
                "!!document.getElementById('run-id') && "
                "document.getElementById('run-id').textContent.length > 0",
                label="run metadata",
            )
            await cdp.poll(
                "document.querySelectorAll('#tree li').length > 0",
                label="binding tree",
            )

            # Seek mid-episode so the plot has history and the transitions
            # table is populated, then let the replay settle.
            await cdp.evaluate(
                "(() => {{ const s = document.getElementById('scrub');"
                " s.value = {}; "
                " s.dispatchEvent(new Event('input', {{bubbles: true}}));"
                " s.dispatchEvent(new Event('change', {{bubbles: true}}));"
                " }})()".format(seek_step)
            )
            await asyncio.sleep(4)
            await cdp.poll(
                "document.querySelectorAll('#transitions tbody tr').length > 0",
                label="transitions",
            )
            await cdp.poll(
                "document.getElementById('camera-pane')"
                ".classList.contains('has-frames')",
                label="camera frame",
            )

            # Open a binding inspector, as the committed screenshot shows.
            await cdp.evaluate(
                "(() => { const rows = document.querySelectorAll("
                "'#tree li[data-kind=\"binding\"]');"
                " if (!rows.length) throw new Error('no binding rows');"
                " rows[3 < rows.length ? 3 : 0].click(); })()"
            )
            await asyncio.sleep(3)

            replay = out_dir / "dashboard-replay.jpg"
            await cdp.screenshot(replay)
            print("wrote {} ({} bytes)".format(replay, replay.stat().st_size))

            # Camera grid: toggle the tile view over every recorded camera.
            await cdp.evaluate(
                "document.getElementById('camera-grid-toggle').click()"
            )
            await asyncio.sleep(5)
            tiles = await cdp.evaluate(
                "document.querySelectorAll('#camera-grid .tile').length"
            )
            if not tiles:
                raise SystemExit("camera grid did not open")
            grid = out_dir / "dashboard-camera-grid.jpg"
            await cdp.screenshot(grid)
            print("wrote {} ({} bytes, {} tiles)".format(
                grid, grid.stat().st_size, tiles
            ))
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--specola", default=os.path.expanduser(
        "~/miniconda3/envs/robocasa/bin/specola"
    ))
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--seek-step", type=int, default=SEEK_STEP)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    server = subprocess.Popen(
        [args.specola, "view", args.recording, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env,
        preexec_fn=os.setsid,
    )
    try:
        wait_http("http://127.0.0.1:{}/".format(port), 120)
        asyncio.run(capture(
            "http://127.0.0.1:{}/".format(port), args.out_dir,
            args.chrome, args.seek_step,
        ))
    finally:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
