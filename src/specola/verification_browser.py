"""Rendered installation checks using the existing workbench CDP approach.

The transport is adapted from Specola's workbench_browser_check.py. Owned
targets, process groups, overall deadlines and retained screenshots cover
the installation boundary rather than a workstation's existing browser.
"""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
import time
import urllib.request
from urllib.parse import urlparse


ERROR_TRAP = r"""
window.__specolaErrors = [];
window.addEventListener('error', e => window.__specolaErrors.push((e.filename || '') + ': ' + e.message));
window.addEventListener('unhandledrejection', e => window.__specolaErrors.push(String(e.reason)));
window.__verificationLabels = [];
window.__verificationDraws = [];
const originalClear = CanvasRenderingContext2D.prototype.clearRect;
CanvasRenderingContext2D.prototype.clearRect = function(...args) {
  if (this.canvas.id === 'plot') {
    window.__verificationLabels = [];
    window.__verificationDraws = [];
  }
  return originalClear.apply(this, args);
};
const prototype = CanvasRenderingContext2D.prototype;
const begin = prototype.beginPath;
prototype.beginPath = function(...args) {
  if (this.canvas.id === 'plot') this.__verificationPath = [];
  return begin.apply(this, args);
};
for (const method of ['moveTo', 'lineTo']) {
  const original = prototype[method];
  prototype[method] = function(x, y, ...rest) {
    if (this.canvas.id === 'plot') (this.__verificationPath ||= []).push([x, y]);
    return original.call(this, x, y, ...rest);
  };
}
for (const method of ['stroke', 'fill']) {
  const original = prototype[method];
  prototype[method] = function(...args) {
    const result = original.apply(this, args);
    if (this.canvas.id === 'plot') window.__verificationDraws.push({
      kind: method, color: method === 'stroke' ? this.strokeStyle : this.fillStyle,
      width: this.lineWidth, alpha: this.globalAlpha,
      path: (this.__verificationPath || []).map(point => point.slice())
    });
    return result;
  };
}
const originalText = CanvasRenderingContext2D.prototype.fillText;
CanvasRenderingContext2D.prototype.fillText = function(text, x, y, ...rest) {
  if (this.canvas.id === 'plot') {
    const m = this.measureText(String(text));
    const ascent = m.actualBoundingBoxAscent, descent = m.actualBoundingBoxDescent;
    window.__verificationLabels.push({text: String(text),
      x0: x - m.actualBoundingBoxLeft, x1: x + m.actualBoundingBoxRight,
      y0: y - ascent, y1: y + descent});
  }
  return originalText.call(this, text, x, y, ...rest);
};
"""


OBSERVATION_PROBE = r"""
(() => {
  if (window.__verificationObservationProbe) return true;
  window.__verificationObservationProbe = true;
  window.__verificationObservation = {serial: 0};
  window.__verificationRequest = 0;
  window.__verificationModelState = () => ({epoch: model.epoch, seq: model.latestSeq,
    step: model.step, time: model.simTime, camera: model.camera, grid: model.gridOpen,
    frame: model.frameStep, bitmap: Boolean(model.bitmap), decode: model.decodeToken});
  const original = ingestBundle;
  ingestBundle = async function(bundle, acknowledge) {
    const request = window.__verificationRequest;
    const eligible = bundle.epoch > model.epoch ||
      (bundle.epoch === model.epoch && bundle.seq > model.latestSeq);
    const received = {request, epoch: bundle.epoch, seq: bundle.seq, step: bundle.step,
      time: bundle.sim_time, camera: model.camera, frame: bundle.frames?.[model.camera]?.step,
      before: window.__verificationModelState()};
    window.__verificationLastBundle = received;
    await original(bundle, acknowledge);
    received.after = window.__verificationModelState();
    if (!eligible || model.epoch !== bundle.epoch || model.latestSeq !== bundle.seq) return;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (request !== window.__verificationRequest ||
          model.epoch !== bundle.epoch || model.latestSeq !== bundle.seq) return;
      window.__verificationObservation = {
        serial: window.__verificationObservation.serial + 1, request,
        step: bundle.step, time: bundle.sim_time, epoch: bundle.epoch, seq: bundle.seq,
        bindings: (bundle.specs || []).flatMap(s => (s.bindings || []).map(b =>
          ({id: bindingId(s.name, b.key), rho: b.rho}))),
        frames: Object.fromEntries(Object.entries(bundle.frames || {}).map(([name, frame]) => [name, frame.step]))
      };
    }));
  };
  return true;
})()
"""


def scrub_condition(step, request, *, camera=False):
    """Require a new accepted bundle and its painted binding/step data."""
    return f"""(() => {{
      const observed = window.__verificationObservation;
      const state = window.__verificationScrub = {{requested_step: {int(step)}, request: {int(request)},
        model: window.__verificationModelState(), received: window.__verificationLastBundle,
        observed: observed && {{request: observed.request, step: observed.step, time: observed.time,
          epoch: observed.epoch, seq: observed.seq, frame: observed.frames?.[model.camera]}},
        first_failed: null}};
      const fail = condition => {{state.first_failed = condition; return false;}};
      if (!observed || observed.request !== {int(request)}) return fail('response-request');
      if (observed.step !== {int(step)}) return fail('response-step');
      if (observed.epoch !== model.epoch || observed.seq !== model.latestSeq) return fail('response-identity');
      const label = state.displayed_step = document.getElementById('step-label').textContent;
      if (!label.startsWith('step {int(step)}/')) return fail('painted-step');
      if (model.simTime !== observed.time) return fail('simulation-time');
      const visible = [...document.querySelectorAll('#tree li[data-kind="binding"]')];
      const binding = observed.bindings.find(b => visible.some(node =>
        node.dataset.key === b.id && node.querySelector('.rho').textContent === formatRho(b.rho)));
      const candidate = binding || observed.bindings.find(b => visible.some(node => node.dataset.key === b.id));
      if (candidate) state.binding = {{...candidate,
        painted: visible.find(node => node.dataset.key === candidate.id).querySelector('.rho').textContent}};
      if (!binding) return fail('painted-binding');
      if ({str(camera).lower()}) {{
        if (!model.bitmap) return fail('camera-bitmap');
        if (observed.frames[model.camera] === undefined) return fail('camera-identity');
        if (model.frameStep !== observed.frames[model.camera]) return fail('camera-frame');
      }}
      return {{step: observed.step, request: observed.request, time: observed.time,
               binding, frames: observed.frames, displayed_step: label}};
    }})()"""


def scrub(tab, step, *, camera=False):
    """Drive the actual range input and await response-derived painted data."""
    tab.evaluate(OBSERVATION_PROBE)
    request = tab.evaluate(f"""(() => {{
      const request = ++window.__verificationRequest;
      const control = document.getElementById('scrub');
      control.value = {json.dumps(str(step))};
      for (const event of ['input', 'change', 'pointerup'])
        control.dispatchEvent(new Event(event, {{bubbles: true}}));
      return request;
    }})()""")
    return tab.wait(scrub_condition(step, request, camera=camera),
                    "Scrub did not receive and paint the requested recorded observation")

class CDP:
    """Match protocol replies without extending deadlines for incoming events."""

    def __init__(self, url, *, timeout: float = 15):
        import websocket
        self.timeout = timeout
        self.socket = websocket.create_connection(url, timeout=timeout, suppress_origin=True)
        self.next_id = 0

    def send(self, method, params=None):
        self.next_id += 1
        identifier = self.next_id
        deadline = time.monotonic() + self.timeout
        self.socket.send(json.dumps({"id": identifier, "method": method, "params": params or {}}))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP {method} exceeded its deadline")
            self.socket.settimeout(remaining)
            response = json.loads(self.socket.recv())
            if response.get("id") == identifier:
                if "error" in response:
                    raise RuntimeError(f"CDP {method}: {response['error']}")
                return response.get("result", {})

    def close(self):
        self.socket.shutdown()


class Tab:
    """Keep a page's target identity distinct from its debugger connection."""

    def __init__(self, chrome, identifier, address):
        self.chrome = chrome
        self.identifier = identifier
        self.cdp = CDP(address)
        self.cdp.send("Page.enable")
        self.cdp.send("Runtime.enable")
        self.cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": ERROR_TRAP})

    def evaluate(self, expression):
        response = self.cdp.send("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True})
        if response.get("exceptionDetails"):
            raise RuntimeError(f"Browser evaluation: {response['exceptionDetails']}")
        return response.get("result", {}).get("value")

    def wait(self, expression, description, timeout=30):
        from .install_verification import wait_until
        last_error = []
        def probe():
            try:
                return self.evaluate(expression)
            except RuntimeError as exc:
                last_error[:] = [str(exc)]
                return False
        try:
            return wait_until(probe, timeout, description)
        except TimeoutError as exc:
            raise TimeoutError(f"{description}; last browser evaluation error: {last_error}") from exc

    def navigate(self, url):
        response = self.cdp.send("Page.navigate", {"url": url})
        if response.get("errorText"):
            raise RuntimeError(response["errorText"])
        self.ready()

    def ready(self):
        self.wait("document.readyState === 'complete' && Boolean(window.specolaSocket?.connected) && "
                  "Boolean(document.querySelector('#tree li')) && typeof model !== 'undefined' && model.order.length > 0",
                  "Dashboard did not connect and render bindings")

    def screenshot(self, path):
        data = self.cdp.send("Page.captureScreenshot", {"format": "png"})["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self):
        try:
            self.chrome.close_target(self.identifier)
        finally:
            self.cdp.close()


class Chrome:
    """Launch a fresh loopback debugger with a private profile and log."""

    def __init__(self, executable, directory, env):
        from .install_verification import OwnedProcess, wait_until
        self.directory = directory
        profile = directory / "chrome-profile"
        profile.mkdir(mode=0o700)
        self.process = OwnedProcess([executable, "--headless=new", "--remote-debugging-port=0",
            "--user-data-dir=" + str(profile), "--no-first-run", "--no-default-browser-check",
            "--window-size=1600,1000", "about:blank"], directory, "chrome", env)
        self.control = None
        self.targets = set()
        try:
            port_file = profile / "DevToolsActivePort"

            def endpoint():
                if self.process.process.poll() is not None:
                    raise RuntimeError(f"Chrome exited; see {self.process.log_path}")
                if port_file.exists():
                    lines = port_file.read_text().splitlines()
                    if len(lines) >= 2 and lines[0].isdigit():
                        return int(lines[0])
                return None
            self.port = wait_until(endpoint, 30, "Chrome did not expose its owned debugger")
            version = json.loads(self.http("/json/version"))
            self.version = version
            self.control = CDP(version["webSocketDebuggerUrl"])
            (directory / "chrome-version.json").write_text(json.dumps(version, indent=2))
        except BaseException:
            self.process.stop()
            raise

    def http(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return response.read().decode()

    def open(self, url):
        from .install_verification import wait_until
        assert self.control is not None
        identifier = self.control.send("Target.createTarget", {"url": "about:blank"})["targetId"]
        self.targets.add(identifier)
        page = wait_until(lambda: next((item for item in json.loads(self.http("/json/list"))
            if item["id"] == identifier and item.get("webSocketDebuggerUrl")), None), 10, "No owned target debugger")
        tab = Tab(self, identifier, page["webSocketDebuggerUrl"])
        try:
            tab.navigate(url)
        except BaseException:
            tab.close()
            raise
        return tab

    def close_target(self, identifier):
        from .install_verification import wait_until
        assert self.control is not None
        self.control.send("Target.closeTarget", {"targetId": identifier})
        wait_until(lambda: not any(item["id"] == identifier for item in json.loads(self.http("/json/list"))),
                   10, "Closed browser target remains present")
        self.targets.discard(identifier)

    def close(self):
        try:
            for identifier in tuple(self.targets):
                self.close_target(identifier)
        finally:
            try:
                if self.control:
                    self.control.close()
            finally:
                self.process.stop()


def assert_no_errors(tab):
    errors = tab.evaluate("window.__specolaErrors || []")
    if errors:
        raise AssertionError(f"Browser errors: {errors}")
    return {"errors": errors}

PLOT_SERIES_PROOF = r"""
(() => {
  const draws = window.__verificationDraws || [];
  const id = window.__verificationPlotBinding;
  if (!id) return {series_drawn: draws.some(draw => draw.kind === 'stroke' &&
    draw.color !== INK.muted && draw.color !== INK.text && draw.path.length > 1 &&
    Math.max(...draw.path.map(p => p[0])) - Math.min(...draw.path.map(p => p[0])) > 1)};
  const canvas = document.getElementById('plot');
  const width = canvas.clientWidth || 300, height = canvas.clientHeight || 150;
  const px = Math.max(50, Math.round(width - 48)), ids = scopedBindingIds();
  const points = seriesFor(id, px, ids);
  if (!points || points.length < 2 || points.some(p => Math.abs(p[1] - 0.05) > 1e-8 ||
      Math.abs(p[2] - 0.05) > 1e-8)) return {series_drawn: false, id, points};
  const peers = ids.flatMap(key => seriesFor(key, px, ids) || []);
  let min = Math.min(0, ...peers.map(p => p[1])), max = Math.max(0, ...peers.map(p => p[2]));
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.05; min -= pad; max += pad;
  const expected = points.map(p => [
    48 + (p[0] - model.minStep) / Math.max(1, model.maxStep - model.minStep) * (width - 48),
    (height - 14) - ((0.5 * (p[1] + p[2]) - min) / (max - min)) * (height - 14)]);
  const stroke = draws.find(draw => draw.kind === 'stroke' && draw.color === bindingColor(id) &&
    draw.width >= 2.3 && draw.alpha === 1 && draw.path.length === expected.length &&
    draw.path.every((p, i) => Math.abs(p[0] - expected[i][0]) < 0.25 &&
                              Math.abs(p[1] - expected[i][1]) < 0.25));
  const domain = {start: model.minStep, end: model.maxStep, px};
  const span = domain.end - domain.start + 1;
  const ordered = points.every((p, i) => i === 0 || p[0] > points[i - 1][0]);
  // The deterministic mock binds at each step. Local histories retain every
  // step; decimated windows retain the first sample in every pixel column.
  const raw_complete = points.length === span &&
    points.every((p, i) => p[0] === domain.start + i);
  const decimated_complete = span > px && points.length === px &&
    points.every((p, i) => p[0] === domain.start + Math.ceil(i * span / px));
  const history_complete = ordered && (raw_complete || decimated_complete);
  return {series_drawn: Boolean(stroke) && history_complete, history_complete,
    domain, cursor: model.step, id, points, stroke: stroke || null,
    windows: [...model.seriesCache.values()].map(r => ({
      start: r.start, end: r.end, bucket: r.bucket, received: r.series !== null
    }))};
})()
"""

def assert_page(tab, *, mock=False):
    """Check rendered data, canvas ink, fonts, errors and measured tick boxes."""
    tab.wait("model.mode !== 'live' && model.maxStep > model.minStep",
             "No finalized multi-step robustness domain")
    if mock:
        from .install_verification import FIXTURE_NAME
        tab.evaluate("""(() => {
          const spec = """ + json.dumps(FIXTURE_NAME) + """;
          const id = bindingId(spec, spec + '::object:1::verification_twice_mass');
          const entities = model.bindings.get(id)?.meta.entities || [];
          if (!entities.some(entity => entity.class === 'object' && entity.id === 1))
            throw new Error('Known object-1 function binding metadata is absent');
          const node = [...document.querySelectorAll('#tree li[data-kind="binding"]')].find(node => node.dataset.key === id);
          if (!node) throw new Error('Known object-1 function binding is absent');
          window.__verificationPlotBinding = id;
          if (model.selection?.type !== 'binding' || model.selection.id !== id) node.click();
          return id;
        })()""")
    tab.wait("(" + PLOT_SERIES_PROOF + ").series_drawn", "The expected robustness series was not drawn")
    series = tab.evaluate(PLOT_SERIES_PROOF)
    result = tab.evaluate("""(() => {
      const canvas = document.getElementById('plot');
      const bytes = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      const ink = bytes.some((v, i) => i % 4 === 3 && v > 0);
      const labels = window.__verificationLabels || [];
      const overlaps = [];
      for (let i = 0; i < labels.length; i++) for (let j = i + 1; j < labels.length; j++) {
        const a = labels[i], b = labels[j];
        if (Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) > 0.5 &&
            Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > 0.5) overlaps.push([a, b]);
      }
      return {rows: document.querySelectorAll('#tree li').length, bindings: model.order.length,
        text: document.getElementById('tree').textContent, ink, labels, overlaps,
        fonts: [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family),
        errors: window.__specolaErrors || []};
    })()""")
    result["series"] = series
    if not series.get("series_drawn") or not result["ink"] or len(result["labels"]) < 3 or not result["fonts"] or result["errors"] or result["overlaps"]:
        raise AssertionError(f"Rendered dashboard assertions failed: {result}")
    if mock:
        from .install_verification import FIXTURE_NAME
        if result["bindings"] != 4 or FIXTURE_NAME not in result["text"] or "known_failure" not in result["text"]:
            raise AssertionError(f"Known rendered fixture inventory differs: {result}")
    return result


def assert_graphics(tab, recording, capabilities):
    """Compare saved JPEG decoding and displayed pixels for this known scene."""
    from .dashboard.recording import RecordingReader, read_geometry
    from PIL import Image, ImageStat
    import io
    import numpy as np
    reader = RecordingReader(recording)
    result = {}
    if "camera" in capabilities:
        cameras = reader.cameras()
        frames = [(camera, step) for camera in cameras for step in reader.frame_steps(camera)]
        if not frames:
            raise AssertionError("Selected native camera profile saved no frames")
        camera = cameras[0]
        camera_steps = reader.frame_steps(camera)
        if not camera_steps:
            raise AssertionError("The selected native camera saved no frames")
        step = camera_steps[-1]
        frame = reader.frame(camera, step)
        encoded = frame[0] if frame else None
        if not encoded:
            raise AssertionError("Recorded frame bytes are absent")
        image = Image.open(io.BytesIO(encoded)).convert("RGB")
        if min(image.size) < 32 or max(ImageStat.Stat(image).stddev) < 2:
            raise AssertionError("Known native scene frame is too small or visually uniform")
        # Seek to the exact persisted frame and compare decoded pixels in
        # the browser with that same JPEG, rather than nonempty image bytes.
        tab.evaluate(f"model.camera = {json.dumps(camera)}; true")
        scrub(tab, step, camera=True)
        tab.wait(f"model.frameStep === {int(step)} && document.getElementById('camera').width > 31",
                 "Browser did not display the saved native frame")
        shown = tab.evaluate("document.getElementById('camera').toDataURL('image/png')")
        actual = Image.open(io.BytesIO(base64.b64decode(shown.split(",", 1)[1]))).convert("RGB")
        if actual.size != image.size:
            raise AssertionError(f"Browser frame dimensions differ: {actual.size} versus {image.size}")
        difference = float(np.abs(np.asarray(actual, dtype=float) - np.asarray(image, dtype=float)).mean())
        if difference > 3:
            raise AssertionError(f"Displayed and saved scene frames differ by {difference} mean channel levels")
        result["camera"] = {"name": camera, "step": step, "size": image.size,
                            "stddev": ImageStat.Stat(image).stddev, "display_mean_absolute_difference": difference}
    if "scene" in capabilities:
        geometry = read_geometry(recording)
        if not geometry or geometry["json"].get("nbody", 0) < 2 or not geometry["json"].get("geoms"):
            raise AssertionError("Known native scene geometry is absent or empty")
        tab.wait("document.body.classList.contains('has-scene') && "
                 "Boolean(document.querySelector('#scene-canvas canvas')) && Boolean(window.SpecolaScene3D)",
                 "Browser did not construct the native 3D scene")
        # Read the compositor's screenshot rather than a WebGL drawing
        # buffer that may already have been discarded after presentation.
        bounds = tab.evaluate("(() => {const r=document.querySelector('#scene-canvas canvas').getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height,viewportWidth:innerWidth,viewportHeight:innerHeight};})()")
        screenshot = tab.chrome.directory / "native-scene-render.png"
        tab.screenshot(screenshot)
        rendered = Image.open(screenshot).convert("RGB")
        scale_x = rendered.width / bounds["viewportWidth"]
        scale_y = rendered.height / bounds["viewportHeight"]
        cropped = rendered.crop((bounds["x"] * scale_x, bounds["y"] * scale_y,
            (bounds["x"] + bounds["width"]) * scale_x, (bounds["y"] + bounds["height"]) * scale_y))
        scene_contrast = ImageStat.Stat(cropped).stddev
        if min(cropped.size) < 32 or max(scene_contrast) < 2:
            raise AssertionError("The native WebGL scene rendered a uniform or absent panel")
        result["scene"] = {"bodies": geometry["json"]["nbody"], "geometries": len(geometry["json"]["geoms"]),
                           "rendered_stddev": scene_contrast, "screenshot": str(screenshot),
                           "visual_review": "Known object identity and scene appearance require inspection of the retained screenshot"}
    return result


def released(child, url, timeout):
    from .install_verification import wait_until
    child.wait(timeout)
    port = urlparse(url).port

    def bind():
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                return False
            return True
    wait_until(bind, 10, "Dashboard port was not reusable after final tab closure")
    return port


def verify_browser(verification, simulator, arguments, specs, directory):
    """Exercise actual page controls, two viewers, Stop and recorded replay."""
    from .install_verification import PROFILES, FIXTURE_NAME, published_report_manifests, report_artifacts, server_url, wait_until
    required = ("websocket", "flask", "flask_socketio", "socketio", "PIL")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    executable = shutil.which(verification.browser) if verification.browser else next((shutil.which(name) for name in
        ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(name)), None)
    if missing or not executable:
        return f"Browser verification prerequisites unavailable: modules={missing}, executable={executable!r}"

    def exercise():
        browser_dir = directory / "browser"
        browser_dir.mkdir()
        chrome = Chrome(executable, browser_dir, verification.env)
        children = []
        tabs = []
        verification.versions["browser"] = chrome.version.get("Browser", "unknown")
        evidence = {"browser": chrome.version, "screenshots": [], "status": "incomplete"}
        failure = None
        try:
            complete_root = browser_dir / "complete"
            drive, complete_status = verification.drive(simulator)
            child = verification.spawn(["run", *arguments, "--specs", str(specs), *drive,
                "--seed", "0", "--ui", "plain", "--web", "0", "--web-video", "jpeg",
                "--output-dir", str(complete_root)], browser_dir, "complete-run")
            children.append(child)
            url = server_url(child, verification.timeout)
            tab = chrome.open(url)
            tabs.append(tab)
            tab.wait("model.mode === 'ended'", "Complete run did not enter replay", verification.timeout)
            wait_until(lambda: len(published_report_manifests(complete_root)) == 1,
                       verification.timeout, "Automatic recording report was not published")
            recording = next(complete_root.glob("*/recording_000"))
            evidence["complete_report"] = report_artifacts(complete_root, expected_status=complete_status, expected_mock=simulator == "mock")
            evidence["complete_before_scrub"] = assert_page(tab, mock=simulator == "mock")
            evidence["complete_scrub"] = scrub(tab, tab.evaluate("model.maxStep"), camera="camera" in PROFILES[simulator]["graphics"])
            evidence["rendered"] = assert_page(tab, mock=simulator == "mock")
            screenshot = browser_dir / "complete.png"
            tab.screenshot(screenshot)
            evidence["screenshots"].append(str(screenshot))
            evidence["graphics"] = assert_graphics(tab, recording, PROFILES[simulator]["graphics"])

            # Author through actual inputs and Validate/Keep/Save controls.
            import yaml
            document = yaml.safe_load(specs.read_text())
            entry = document["specs"][-1] if simulator == "robodojo" else document["specs"][0]
            name = "browser[1m]\\literal"
            tab.evaluate("document.getElementById('tab-discover').click(); true")
            tab.wait("document.querySelectorAll('#class-tree li').length > 0", "Discovery did not render")
            for identifier, value in (("spec-name", name), ("spec-formula", entry["formula"]),
                                      ("spec-constants", json.dumps(entry.get("constants", {})))):
                tab.evaluate(f"(() => {{const e=document.getElementById({json.dumps(identifier)}); e.value={json.dumps(value)}; e.dispatchEvent(new Event('input', {{bubbles:true}})); return true;}})()")
            tab.evaluate("document.getElementById('validate').click(); true")
            tab.wait("!document.getElementById('keep').disabled", "Browser-authored specification was not accepted")
            report = tab.evaluate("document.getElementById('validation').textContent")
            if name not in report or "\x1b" in report:
                raise AssertionError(f"Literal authored name did not survive validation: {report!r}")
            tab.evaluate("document.getElementById('keep').click(); document.getElementById('save-specs').click(); true")
            tab.wait("document.getElementById('save-status').classList.contains('ok')", "Browser-authored file was not saved")
            saved = next(recording.parent.glob("authored/specs.yaml"))
            authored = yaml.safe_load(saved.read_text())
            discovered = json.loads((directory / "discovery.json").read_text())
            declarations = discovered.get("authored_functions", {}).get("declarations", {})
            if authored["specs"][0]["name"] != name or authored.get("functions", {}) != declarations:
                raise AssertionError("Saved browser authoring lost literal identity or loaded trusted declarations")
            evidence["authoring_errors"] = assert_no_errors(tab)
            tab.screenshot(browser_dir / "authoring.png")
            evidence["screenshots"].append(str(browser_dir / "authoring.png"))
            tab.evaluate("document.getElementById('tab-monitor').click(); true")

            # A second actual target stays connected while the first closes.
            second = chrome.open(url)
            tabs.append(second)
            tab.close()
            tabs.remove(tab)
            deadline = time.monotonic() + 3.0  # exceeds the server's 2 s reconnect grace
            while time.monotonic() < deadline:
                if child.process.poll() is not None or not second.evaluate("window.specolaSocket.connected"):
                    raise AssertionError("Closing one of two viewers ended the dashboard")
                time.sleep(0.1)
            second.evaluate("window.__verificationPreviousDocument = true; true")
            second.cdp.send("Page.reload", {"ignoreCache": True})
            second.wait("!window.__verificationPreviousDocument && document.readyState === 'complete' && Boolean(window.specolaSocket?.connected)",
                        "Refresh did not load and connect a new document")
            second.ready()
            evidence["refreshed_scrub"] = scrub(second, second.evaluate("model.minStep"), camera="camera" in PROFILES[simulator]["graphics"])
            second.evaluate("document.getElementById('play-pause').click(); true")
            second.wait("model.step > model.minStep", "Recorded playback did not advance")
            second.evaluate("document.getElementById('play-pause').click(); true")
            evidence["refreshed_rendered"] = assert_page(second, mock=simulator == "mock")
            evidence["refreshed_errors"] = assert_no_errors(second)
            second.screenshot(browser_dir / "refreshed-replay.png")
            second.close()
            tabs.remove(second)
            evidence["complete_released_port"] = released(child, url, verification.timeout)

            # Native check/run can launch another Kit app. Release the
            # completed server's owning process before either command.
            verification.command(["check", *arguments, str(saved), "--no-pager"], browser_dir, "authored-check")
            verification.command(["run", *arguments, "--specs", str(saved), "--mode", PROFILES[simulator]["mode"],
                "--steps", "4", "--ui", "plain", "--output-dir", str(browser_dir / "authored-run")], browser_dir, "authored-run")
            evidence["authored_report"] = report_artifacts(browser_dir / "authored-run", expected_status="completed")
            if {row["spec"] for row in evidence["authored_report"]["episode_outcomes"]} != {name}:
                raise AssertionError("Browser-authored literal name did not survive result reporting")

            # Launch another live run. Stopping it uses the actual control,
            # then leaves a finalized, readable and scrubbable recording.
            stop_root = browser_dir / "stopped"
            child = verification.spawn(["run", *arguments, "--specs", str(specs), "--mode", "hold", "--steps", "0",
                "--seed", "0", "--ui", "plain", "--web", "0", "--web-video", "jpeg", "--output-dir", str(stop_root)],
                browser_dir, "stop-run")
            children.append(child)
            url = server_url(child, verification.timeout)
            tab = chrome.open(url)
            tabs.append(tab)
            tab.wait("model.mode === 'live' && model.maxStep > model.minStep && !document.getElementById('stop-run').hidden",
                     "Live output/Stop control did not render", verification.timeout)
            if simulator == "mock":
                # Retain an actual early live window, then grow within its bucket.
                evidence["live_window"] = tab.wait("""(() => {
                  const px = Math.max(50, Math.round(document.getElementById('plot').clientWidth - 48));
                  const window = fetchSeriesWindows(scopedBindingIds(), px);
                  return window.series && {start: window.start, end: window.end,
                    bucket: window.bucket, series: window.series};
                })()""", "Live series window was not acknowledged")
            before = tab.evaluate("model.maxStep")
            tab.wait(f"model.maxStep > {before + 2}", "Live output did not advance beyond the early window")
            tab.evaluate("window.confirm = () => true; document.getElementById('stop-run').click(); true")
            tab.wait("model.mode === 'ended' && document.getElementById('stop-run').hidden && "
                     "document.getElementById('run-state').textContent === 'stopped'", "Stop did not leave replay available", verification.timeout)
            wait_until(lambda: len(published_report_manifests(stop_root)) == 1,
                       verification.timeout, "Stopped run report was not published")
            evidence["stopped_report"] = report_artifacts(stop_root, expected_status="truncated", expected_mock=simulator == "mock")
            verification.pdf_checks(simulator, complete_root, "recorded-complete")
            verification.pdf_checks(simulator, stop_root, "recorded-stopped")
            evidence["stopped_before_scrub"] = assert_page(tab, mock=simulator == "mock")
            evidence["stopped_scrub"] = scrub(tab, tab.evaluate("model.minStep"), camera="camera" in PROFILES[simulator]["graphics"])
            evidence["stopped_rendered"] = assert_page(tab, mock=simulator == "mock")
            evidence["stopped_errors"] = assert_no_errors(tab)
            tab.screenshot(browser_dir / "stopped.png")
            tab.close()
            tabs.remove(tab)
            evidence["stopped_released_port"] = released(child, url, verification.timeout)

            # Exercise the installed view command independently of live serving.
            child = verification.spawn(["view", str(recording), "--port", "0"], browser_dir, "view")
            children.append(child)
            url = server_url(child, verification.timeout)
            tab = chrome.open(url)
            tabs.append(tab)
            tab.wait("model.mode === 'view'", "Installed view command did not enter recorded mode")
            evidence["view_scrub"] = scrub(tab, tab.evaluate("model.maxStep"), camera="camera" in PROFILES[simulator]["graphics"])
            evidence["view_rendered"] = assert_page(tab, mock=simulator == "mock")
            evidence["view_errors"] = assert_no_errors(tab)
            tab.screenshot(browser_dir / "installed-view.png")
            tab.close()
            tabs.remove(tab)
            evidence["view_released_port"] = released(child, url, verification.timeout)
        except Exception as exc:
            failure = exc
            evidence.update(status="fail", failure=f"{type(exc).__name__}: {exc}")
            evidence["diagnostics"] = []
            for tab in tabs:
                try:
                    state = tab.evaluate("({scrub: window.__verificationScrub, "
                        "received: window.__verificationLastBundle, "
                        "model: window.__verificationModelState?.()})")
                    evidence["diagnostics"].append({"target": tab.identifier, "state": state})
                except Exception as diagnostic_error:
                    evidence["diagnostics"].append({"target": tab.identifier,
                        "error": str(diagnostic_error)[:2000]})
            raise
        finally:
            for tab in tabs:
                try:
                    tab.screenshot(browser_dir / ("failure-" + tab.identifier + ".png"))
                    tab.close()
                except Exception:
                    pass
            cleanup_errors = []
            try:
                chrome.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
            for child in children:
                try:
                    child.stop()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            try:
                (browser_dir / "cleanup.json").write_text(json.dumps({"errors": cleanup_errors}, indent=2))
                if failure is not None or cleanup_errors:
                    evidence.update(status="fail", cleanup_errors=cleanup_errors)
                    (browser_dir / "browser-evidence.partial.json").write_text(json.dumps(evidence, indent=2, default=str))
            except Exception as evidence_error:
                if failure is None:
                    raise
                print(f"Could not persist browser failure evidence: {evidence_error}", file=sys.stderr)
            if cleanup_errors:
                if failure is None:
                    raise RuntimeError("Owned server cleanup failed: " + repr(cleanup_errors))
                print("Owned server cleanup failed: " + repr(cleanup_errors), file=sys.stderr)
        evidence["status"] = "pass"
        (browser_dir / "browser-evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
        return evidence

    verification.check(simulator, "browser", exercise)
    return None
