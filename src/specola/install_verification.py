"""Bounded current-environment installation checks with retained evidence.

The seven CLI workflows are a coverage checklist. Each simulator profile
names a small scene; a passing profile does not establish other task coverage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any
import uuid


FIXTURE_NAME = "ll[1m]ll\\literal"
CLI_WORKFLOWS = ("discover", "check", "run", "write-specs", "report", "view", "doctor")
FIXTURE_MODULE = "specola.data.verification_adapter"
# Profiles use the existing public factories and drive modes. Assets remain
# in the active environment; verification never searches sibling environments.
PROFILES = {
    "mock": {"sim": "verification-mock", "task": None, "mode": "hold", "graphics": ()},
    "dmcontrol": {"task": "walker:walk", "mode": "random", "graphics": ("camera", "scene")},
    "metaworld": {"task": "pick-place-v3", "mode": "random", "graphics": ("camera", "scene")},
    "mamujoco": {"task": "HalfCheetah:2x3", "mode": "random", "graphics": ("camera", "scene")},
    "pybullet": {"task": "plane.urdf@fixed,kuka_iiwa/model.urdf@fixed,cube_small.urdf",
                 "mode": "hold", "graphics": ("camera",)},
    "robodojo": {"task": "stack_bowls", "mode": "hold", "graphics": ("camera", "scene")},
    "libero": {"task": "libero_goal:0", "mode": "hold", "graphics": ("camera", "scene")},
    "robocasa": {"task": "PickPlaceCounterToCabinet", "mode": "random", "graphics": ("camera", "scene")},
}


@dataclass
class Outcome:
    simulator: str
    capability: str
    status: str
    reason: str
    remedy: str
    duration_seconds: float
    log: str
    details: dict[str, Any]
    required: bool = True


class OwnedProcess:
    """Own one new process group, a closed stdin and a complete disk log."""

    def __init__(self, command, directory: Path, name: str, env, *, stdin=None):
        self.command = list(map(str, command))
        self.log_path = directory / (name + ".log")
        self.command_path = directory / (name + "-command.json")
        self.command_path.write_text(json.dumps({"argv": self.command, "cwd": str(directory),
            "environment": {key: env.get(key) for key in (
                "PYTHONPATH", "PYTHONNOUSERSITE", "MUJOCO_GL", "MPLCONFIGDIR", "NUMBA_CACHE_DIR",
                "HF_HUB_OFFLINE", "LIBERO_CONFIG_PATH", "ROBODOJO_ROOT", "ROBODOJO_ENV_CFG", "ROBODOJO_SEED", "ROBODOJO_LAYOUT")}}, indent=2))
        self._log = self.log_path.open("wb")
        self._stdin = None
        if stdin is not None:
            input_path = directory / (name + "-stdin.txt")
            input_path.write_text(stdin, encoding="utf-8")
            self._stdin = input_path.open("rb")
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen(self.command, cwd=directory, env=env,
                stdin=self._stdin or subprocess.DEVNULL, stdout=self._log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self._log.close()
            if self._stdin:
                self._stdin.close()
            raise

    def wait(self, timeout, expected_code=0):
        code = self.process.wait(timeout=timeout)
        if code != expected_code:
            raise RuntimeError(f"Child exited {code}, expected {expected_code}; see {self.log_path}")
        # A clean exit code is not a clean exit: interpreter-teardown
        # errors (a render context freed after its display) print a
        # traceback after the command's own output and still exit 0.
        self._log.flush()
        if b"Traceback (most recent call last)" in self.log_path.read_bytes():
            raise RuntimeError(f"Child exited {code} with a traceback in its output; see {self.log_path}")
        return code

    def stop(self):
        cleanup_started = time.monotonic()
        def present():
            self.process.poll()
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                return False
            return True
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not present():
                    break
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + 5
                while present() and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.process.wait(timeout=5)
            if present():
                raise RuntimeError(f"Owned process group {self.process.pid} survived cleanup")
        finally:
            (self.log_path.with_suffix(".process.json")).write_text(json.dumps({
                "argv": self.command, "pid": self.process.pid, "returncode": self.process.returncode,
                "subprocess_elapsed_seconds": time.monotonic() - self.started,
                "cleanup_seconds": time.monotonic() - cleanup_started,
            }, indent=2))
            self._log.close()
            if self._stdin:
                self._stdin.close()


def wait_until(probe, timeout, description):
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.05)


def server_url(child, timeout):
    def probe():
        content = child.log_path.read_text(errors="replace")
        matches = re.findall(r"(?:Web dashboard:|Serving .*? at)\s*(http://127\.0\.0\.1:\d+)", content)
        if matches:
            return matches[-1]
        if child.process.poll() is not None:
            raise RuntimeError(f"Server exited before announcing its URL; see {child.log_path}")
        return None
    return wait_until(probe, timeout, "Dashboard URL was not announced")


def published_report_manifests(directory: Path):
    """Find published manifests outside the exporter's private staging area."""
    # export_report replaces artifacts individually, then the manifest last.
    return [path for path in directory.rglob("report-manifest.json")
            if not any(part.startswith(".report-export-")
                       for part in path.relative_to(directory).parts)]


def report_artifacts(directory: Path, *, expected_status=None, expected_mock=False):
    """Validate published hashes, actual CSV outcomes and the PDF envelope."""
    manifests = published_report_manifests(directory)
    if len(manifests) != 1:
        raise AssertionError(f"Expected one report manifest under {directory}, found {len(manifests)}")
    path = manifests[0]
    document = json.loads(path.read_text())
    for filename, digest in document["files"].items():
        artifact = path.parent / filename
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Published report hash differs: {artifact}")
    pdf = path.parent / "report-run.pdf"
    if not pdf.read_bytes().startswith(b"%PDF-") or not document["pages"]:
        raise AssertionError("No published PDF pages")
    with (path.parent / "report-episode-summary.csv").open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if not episodes:
        raise AssertionError("Report contains no bound episode/spec outcomes")
    if expected_status and {row["status"] for row in episodes} != {expected_status}:
        raise AssertionError(f"Unexpected episode status: {episodes}")
    if expected_mock:
        actual = {row["spec"]: row["verdict"] for row in episodes}
        if actual != {FIXTURE_NAME: "pass", "known_failure": "fail"}:
            raise AssertionError(f"Known fixture outcomes differ: {actual}")
        with (path.parent / "report-plot-data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        counts = {name: {row["binding_key"] for row in rows if row["authored_spec"] == name}
                  for name in actual}
        if any(len(keys) != 2 for keys in counts.values()):
            raise AssertionError(f"Expected two object bindings per fixture spec: {counts}")
        values = {round(float(row["rho"]), 9) for row in rows
                  if row["authored_spec"] == FIXTURE_NAME and row["rho"] not in ("", "nan")}
        if values != {0.05, 0.35}:
            raise AssertionError(f"Known function robustness differs: {values}")
    return {"manifest": str(path), "pages": len(document["pages"]),
            "episode_outcomes": episodes, "scene_images": document["scene_images"]}


def native_prerequisites(simulator, env, *, working_directory=None):
    """Check the inspected profile's immediate files without native imports."""
    import importlib.util
    import yaml
    details: dict[str, Any] = {"scope": "Immediate profile files; recursive native asset resolution remains a runtime check"}
    missing = []
    base = Path(working_directory) if working_directory is not None else Path.cwd()
    def configured(value):
        path = Path(value)
        return path if path.is_absolute() else base / path
    if simulator == "libero":
        try:
            installed = metadata.version("hf-libero")
        except metadata.PackageNotFoundError:
            installed = "absent"
        if installed != "0.1.4":
            return False, "The offline asset preflight supports hf-libero 0.1.4", details
        config_dir = Path(env.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))).absolute()
        env["LIBERO_CONFIG_PATH"] = str(config_dir)
        env["HF_HUB_OFFLINE"] = "1"
        config = config_dir / "config.yaml"
        if not config.is_file():
            return False, f"Existing LIBERO config is required before import: {config}", details
        try:
            paths = yaml.safe_load(config.read_text())
        except (OSError, yaml.YAMLError) as exc:
            return False, f"LIBERO config cannot be read: {exc}", details
        if not isinstance(paths, dict):
            return False, "LIBERO config must be a path mapping", details
        for key in ("assets", "bddl_files", "init_states"):
            path = paths.get(key)
            if not isinstance(path, str) or not configured(path).is_dir():
                missing.append(f"configured {key}: {path!r}")
        module = importlib.util.find_spec("libero")
        local = Path(module.origin).parent / "libero" / "assets" if module and module.origin else None
        root = local if local is not None and local.exists() else Path.home() / ".cache/libero/assets"
        for relative in (
            "scenes/libero_tabletop_base_style.xml", "articulated_objects/wooden_cabinet.xml",
            "articulated_objects/flat_stove.xml", "turbosquid_objects/wine_rack/wine_rack.xml",
            "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
            "stable_hope_objects/cream_cheese/cream_cheese.xml",
            "turbosquid_objects/wine_bottle/wine_bottle.xml", "stable_scanned_objects/plate/plate.xml",
        ):
            if not (root / relative).is_file():
                missing.append(str(root / relative))
        task = "libero_goal/open_the_middle_drawer_of_the_cabinet"
        for key, suffix in (("bddl_files", ".bddl"), ("init_states", ".pruned_init")):
            if isinstance(paths.get(key), str) and not (configured(paths[key]) / (task + suffix)).is_file():
                missing.append(str(configured(paths[key]) / (task + suffix)))
        details.update(config=str(config), actual_asset_root=str(root), hub_offline=True)
    elif simulator == "robodojo":
        root = Path(env.get("ROBODOJO_ROOT", os.getcwd())).absolute()
        env["ROBODOJO_ROOT"] = str(root)
        overrides = {key: env[key] for key, expected in (("ROBODOJO_ENV_CFG", "arx_x5"),
            ("ROBODOJO_SEED", "0"), ("ROBODOJO_LAYOUT", "0")) if key in env and env[key] != expected}
        if overrides:
            return False, "The declared stack_bowls profile requires default arx_x5/family 0/layout ordinal 0: " + repr(overrides), details
        for relative in (
            "env/global_configs.py", "env_cfg/arx_x5.yml", "env_cfg/sim/sim_config.yml",
            "env_cfg/scene/default.yml", "env_cfg/robot/dual_x5.yml", "env_cfg/camera/camera_config.yml",
            "task/RoboDojo/tasks/stack_bowls.py", "task/RoboDojo/config/stack_bowls.yml",
            "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json",
            "Assets/Object/RoboDojo/Rigid/bowl/00010/object.usdz",
            "Assets/Object/RoboDojo/Geometry/camera_stand/00000/object.usd",
            "Assets/Background/brown_photostudio_02_4k.hdr",
        ):
            if not (root / relative).is_file():
                missing.append(str(root / relative))
        details.update(root=str(root), task="stack_bowls", config="arx_x5", family=0, layout_ordinal=0)
    if missing:
        details["missing"] = missing
        return False, "Required existing native profile files are absent", details
    return True, "Immediate profile prerequisites exist", details

def native_run_assertions(directory, specs, *, simulator, mode):
    """Require the selected specifications to bind and observations to advance."""
    import yaml
    document = yaml.safe_load(specs.read_text())
    expected = {entry["name"] for entry in document["specs"]}
    paths = list(directory.rglob("report-plot-data.csv"))
    if len(paths) != 1:
        raise AssertionError("Native run did not publish one plot-data CSV")
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    finite = [row for row in rows if row["binding_key"] and row["rho"] and math.isfinite(float(row["rho"]))]
    if {row["authored_spec"] for row in finite} != expected:
        raise AssertionError("Native finite bound-spec inventory differs from the selected fixture")
    times = {float(row["time"]) for row in finite if row["time"] and math.isfinite(float(row["time"]))}
    steps = {int(row["step"]) for row in finite}
    if len(times) < 2 or len(steps) < 2:
        raise AssertionError("Native observations did not advance across multiple times and steps")
    if simulator == "robodojo":
        distances = [float(row["rho"]) + 0.005 for row in finite if row["authored_spec"] == "bodies_stay_apart"]
        if not distances or any(value < -1e-7 or value > 0.0500001 for value in distances):
            raise AssertionError("Authored sampled mesh outputs are absent or outside their declared cutoff")
        if not any("bowl0" in row["binding"] for row in finite if row["authored_spec"] == "bodies_stay_apart"):
            raise AssertionError("The expected bowl0 function binding is absent")
    details = {"bound_specifications": sorted(expected), "observed_steps": len(steps),
               "observed_times": len(times), "mode": mode}
    if mode in ("random", "replay"):
        import numpy as np
        states = list(directory.rglob("*_states.npz"))
        if len(states) != 1:
            raise AssertionError("The declared moving profile did not save its simulator state trace")
        from .state_recording import StateRecording
        recording = StateRecording.load(states[0])
        changed = recording.states.shape[0] > 1 and np.any(recording.states[1:, 1:] != recording.states[:-1, 1:])
        if not changed:
            raise AssertionError("The native drive left every saved non-time simulator-state value unchanged")
        details["non_time_simulator_state_changed"] = True
    else:
        details["advance_scope"] = "Multiple observation times and steps through the source-inspected supported hold advancer; no physical-motion claim"
    return details

class Verification:
    """Run selected profiles and publish an outcome after each capability."""

    def __init__(self, output: Path, *, timeout=180.0, browser=None, demo=None, selected_profiles=()):
        self.directory = output.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.timeout = timeout
        self.browser = browser
        self.demo = demo
        self.outcomes = []
        self.completed = False
        self.selected_profiles = list(selected_profiles)
        self.env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        for variable, leaf in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
            cache = self.directory / leaf
            cache.mkdir()
            self.env[variable] = str(cache)
        self.env.setdefault("MUJOCO_GL", "egl")
        self.versions = {"python": sys.version, "executable": sys.executable}
        for package in ("specola", "numpy", "pandas", "matplotlib", "flask", "flask-socketio",
                        "python-socketio", "websocket-client", "Pillow", "mujoco", "hf-libero",
                        "robosuite", "robocasa", "dm-control", "metaworld", "gymnasium-robotics", "pybullet"):
            try:
                self.versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                self.versions[package] = "not installed"
        self.fixtures = self.directory / "fixtures"
        self.fixtures.mkdir()
        source = Path(__file__).parent / "data"
        for name in ("verification-specs.yaml", "verification_functions.py", "verification-robodojo.yaml", "verification_robodojo_functions.py", "verification-libero-demo.hdf5", "verification-libero-specs.yaml",
                     "verification-libero-origin.md", "verification-libero-LICENSE.txt", "verification-provenance.json"):
            shutil.copyfile(source / name, self.fixtures / name)
        self.publish()

    def publish(self):
        result = {"schema": 1, "scope": "seven CLI workflows and selected bounded current-environment profiles",
                  "cli_workflows": list(CLI_WORKFLOWS), "versions": self.versions,
                  "evidence_directory": str(self.directory),
                  "implementation": {name: {"path": str(Path(__file__).parent / name),
                      "sha256": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()}
                      for name in ("install_verification.py", "verification_browser.py", "doctor.py", "report.py", "adapters/robodojo.py")},
                  "packaged_fixtures": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.fixtures.iterdir()},
                  "whole_verification_elapsed_seconds": time.monotonic() - self.started,
                  "outcomes": [asdict(outcome) for outcome in self.outcomes],
                  "completed": self.completed, "selected_profiles": self.selected_profiles,
                  "ready": self.completed and bool(self.outcomes) and all(not outcome.required or outcome.status == "pass" for outcome in self.outcomes)}
        (self.directory / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def finish(self):
        """Publish terminal readiness only after orchestration has returned."""
        self.completed = True
        return self.publish()

    def outcome(self, simulator, capability, status, reason, *, started=None, log="", details=None, remedy="", required=True):
        value = Outcome(simulator, capability, status, reason, remedy,
                        0.0 if started is None else time.monotonic() - started, str(log), details or {}, required)
        self.outcomes.append(value)
        self.publish()
        return value

    def check(self, simulator, capability, operation):
        started = time.monotonic()
        log = self.directory / (simulator + "-" + capability.replace("/", "-") + ".log")
        try:
            details = operation() or {}
        except Exception as exc:
            log.write_text(traceback.format_exc())
            return self.outcome(simulator, capability, "fail", f"{type(exc).__name__}: {exc}",
                started=started, log=log, remedy="Inspect the retained capability and child logs, correct the failure, and rerun verification.")
        log.write_text(json.dumps(details, indent=2, default=str))
        return self.outcome(simulator, capability, "pass", "Expected assertions passed", started=started, log=log, details=details)

    def spawn(self, arguments, directory, name, *, stdin=None):
        return OwnedProcess([sys.executable, "-m", "specola.cli", *arguments], directory, name, self.env, stdin=stdin)

    def command(self, arguments, directory, name, *, stdin=None, expected_code=0):
        child = self.spawn(arguments, directory, name, stdin=stdin)
        try:
            child.wait(self.timeout, expected_code)
            return child.log_path.read_text(errors="replace")
        finally:
            child.stop()

    def arguments(self, simulator):
        profile = PROFILES[simulator]
        arguments = ["--sim", profile.get("sim", simulator)]
        if simulator == "mock":
            arguments += ["--adapter-module", FIXTURE_MODULE]
        if profile["task"]:
            arguments += ["--task", profile["task"]]
        return arguments

    def pdf_checks(self, simulator, directory, label):
        missing = [program for program in ("pdfinfo", "pdffonts", "pdftotext", "pdftoppm") if not shutil.which(program)]
        if missing:
            return self.outcome(simulator, "pdf-rendering/" + label, "not-run",
                "PDF inspection tools unavailable: " + ", ".join(missing),
                remedy="Install Poppler's pdfinfo, pdffonts, pdftotext and pdftoppm, then rerun verification.")
        def render():
            manifest_path = published_report_manifests(directory)[0]
            document = json.loads(manifest_path.read_text())
            pdf = manifest_path.parent / "report-run.pdf"
            destination = self.directory / "pdf-evidence" / (simulator + "-" + label)
            destination.mkdir(parents=True)
            def execute(argv, name):
                child = OwnedProcess(argv, destination, name, self.env)
                try:
                    child.wait(self.timeout)
                    return child.log_path.read_text(errors="replace")
                finally:
                    child.stop()
            info = execute(["pdfinfo", str(pdf)], "pdfinfo")
            count = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            if not count or int(count.group(1)) != len(document["pages"]):
                raise AssertionError("Parsed PDF page count differs from its published inventory")
            fonts = execute(["pdffonts", str(pdf)], "pdffonts")
            if "Type 3" in fonts or "TrueType" not in fonts:
                raise AssertionError("Expected embedded TrueType PDF fonts")
            text_path = destination / "report-text.txt"
            execute(["pdftotext", str(pdf), str(text_path)], "pdftotext")
            execute(["pdftoppm", "-scale-to", "1700", "-png", str(pdf), str(destination / "page")], "pdftoppm")
            pages = sorted(destination.glob("page-*.png"))
            if len(pages) != len(document["pages"]):
                raise AssertionError("Rendered PDF page count differs from its published inventory")
            inventory = [{"page": declared["page"], "title": declared["title"], "png": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "visual_reviewed": False}
                         for path, declared in zip(pages, document["pages"])]
            (destination / "rendered-page-inventory.json").write_text(json.dumps(inventory, indent=2))
            return {"pages": inventory, "automated_scope": "PDF parse, page count, font type and full-page rendering",
                    "layout_scope": "Exporter layout bounds and declared minimum font; human readability inspection remains pending"}
        return self.check(simulator, "pdf-rendering/" + label, render)

    def drive(self, simulator, *, stopped=False):
        if simulator == "libero" and self.demo and not stopped:
            path = self.fixtures / "verification-libero-demo.hdf5" if self.demo == "packaged" else Path(self.demo).resolve()
            if not path.is_file():
                raise ValueError(f"LIBERO replay demonstration is absent: {path}")
            return ["--mode", "replay", "--demo", str(path), "--steps", "0"], "completed"
        return ["--mode", "hold" if stopped else PROFILES[simulator]["mode"], "--steps", "0" if stopped else "12"], "truncated" if stopped else "completed"

    def profile(self, simulator):
        import yaml
        directory = self.directory / simulator
        directory.mkdir()
        arguments = self.arguments(simulator)
        for capability in ("camera", "scene"):
            if capability not in PROFILES[simulator]["graphics"]:
                self.outcome(simulator, capability, "unsupported", "The declared profile has no " + capability + " hook", required=False)
        fixture = "verification-specs.yaml" if simulator == "mock" else "verification-robodojo.yaml"
        if simulator == "libero" and self.demo:
            fixture = "verification-libero-specs.yaml"
        specs = self.fixtures / fixture

        def discover():
            options = ["--functions", str(specs)] if simulator in ("mock", "robodojo") else []
            output = directory / "discovery.json"
            self.command(["discover", *arguments, *options, "--format", "json", "--output", str(output)], directory, "discover")
            document = json.loads(output.read_text())
            if not document["entities"]:
                raise AssertionError("Discovery returned no entities")
            if simulator == "mock":
                if {entity["id"] for entity in document["entities"]} != set(range(5)):
                    raise AssertionError("Fixture discovery entity inventory differs")
                if document["authored_functions"]["active_functions"] != ["twice_mass"]:
                    raise AssertionError("Trusted fixture function was not prepared")
            elif simulator != "robodojo" and not (simulator == "libero" and self.demo):
                # This is an explicitly labelled finite-signal smoke predicate,
                # not a transcription of the benchmark's task success.
                candidates = [(entity, prop) for entity in document["entities"] for prop in entity["properties"]
                              if isinstance(prop.get("value"), (int, float)) and math.isfinite(prop["value"])]
                if not candidates:
                    raise AssertionError("No finite discovered scalar for the smoke specification")
                entity, prop = candidates[0]
                specs.write_text(yaml.safe_dump({"specs": [{"name": "finite_signal_smoke",
                    "formula": f"abs({entity['class']}_{entity['id']}[{prop['key']}]) >= 0"}]}))
            return {"entities": len(document["entities"]), "path": str(output), "task": PROFILES[simulator]["task"]}

        # Each profile owns its spec path; native discovery cannot overwrite
        # another profile's trusted packaged fixture.
        if simulator not in ("mock", "robodojo") and not (simulator == "libero" and self.demo):
            specs = directory / "smoke-specs.yaml"
        found = self.check(simulator, "discover", discover)
        if found.status != "pass":
            for capability in ("check", "run-and-reports", "browser"):
                self.outcome(simulator, capability, "not-run", "Discovery prerequisite failed", remedy="Fix discovery and rerun.")
            return

        checked = self.check(simulator, "check", lambda: {"output": self.command(
            ["check", *arguments, str(specs), "--no-pager"], directory, "check")})
        if checked.status != "pass":
            self.outcome(simulator, "run-and-reports", "not-run", "Specification check failed", remedy="Fix the selected specification and rerun.")
            self.outcome(simulator, "browser", "not-run", "Specification check failed", remedy="Fix the selected specification and rerun.")
            return

        def run():
            drive, status = self.drive(simulator)
            self.command(["run", *arguments, "--specs", str(specs), *drive,
                "--seed", "0", "--ui", "plain", "--output-dir", str(directory / "complete")], directory, "run")
            details = report_artifacts(directory / "complete", expected_status=status, expected_mock=simulator == "mock")
            if simulator != "mock":
                details["native"] = native_run_assertions(directory / "complete", specs, simulator=simulator, mode=drive[1])
            return details
        ran = self.check(simulator, "run-and-reports", run)
        if ran.status == "pass":
            self.pdf_checks(simulator, directory / "complete", "complete")
        if simulator == "mock" and ran.status == "pass":
            def report():
                self.command(["report", str(directory / "complete"), "--output-dir", str(directory / "regenerated")], directory, "report")
                return report_artifacts(directory / "regenerated", expected_status="completed", expected_mock=True)
            self.check(simulator, "report", report)

            def write():
                written = directory / "written.yaml"
                self.command(["write-specs", *arguments, "--functions", str(specs), "--ui", "plain", "--output", str(written)],
                    directory, "write-specs", stdin=f"written[1]\\literal\nobject_*[verification_twice_mass] >= 0.45\n\ny\n\n\n")
                saved = yaml.safe_load(written.read_text())
                if saved["specs"][0]["name"] != "written[1]\\literal" or "twice_mass" not in saved["functions"]:
                    raise AssertionError("The writer did not preserve literal names and trusted declarations")
                self.command(["check", *arguments, str(written), "--no-pager"], directory, "written-check")
                return {"saved": str(written)}
            self.check(simulator, "write-specs", write)
        if ran.status == "pass":
            from .verification_browser import verify_browser
            reason = verify_browser(self, simulator, arguments, specs, directory)
            if reason:
                self.outcome(simulator, "browser", "not-run", reason, remedy="Install specola[verify] and a Chrome/Chromium executable in this environment, then rerun.")
        else:
            self.outcome(simulator, "browser", "not-run", "Run prerequisite failed", remedy="Fix the retained run failure and rerun.")


def verify_installation(output=None, *, simulators=None, timeout=180.0, browser=None, demo=None):
    """Verify the active interpreter without installing or crossing environments."""
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("Verification timeout must be finite and positive")
    if os.name != "posix":
        raise RuntimeError("The owned-process verifier currently supports POSIX environments")
    from .registry import adapter_status
    statuses = {status.name: status for status in adapter_status()}
    selected = list(dict.fromkeys(["mock", *(simulators if simulators else (
        name for name, status in statuses.items() if name not in ("mock", "dataset") and status.ready))]))
    if demo and "libero" not in selected:
        raise ValueError("--demo requires selecting an available LIBERO profile or --sim libero")
    output = Path(output) if output is not None else Path.cwd() / (
        "specola-verification-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    verification = Verification(output, timeout=timeout, browser=browser, demo=demo, selected_profiles=selected)
    verification.check("installation", "doctor", lambda: {"output": verification.command(
        ["doctor"], verification.directory, "doctor-inventory")})
    def preflight():
        child = OwnedProcess([sys.executable, "-c",
            "from specola.report_export import report_preflight; import sys; report_preflight(sys.argv[1])",
            str(verification.directory / "report-preflight")], verification.directory, "report-preflight", verification.env)
        try:
            child.wait(timeout)
        finally:
            child.stop()
        return {"shared_preflight": "specola.report_export.report_preflight", "log": str(child.log_path)}
    reports = verification.check("installation", "report-dependencies", preflight)
    if reports.status != "pass":
        for simulator in selected:
            verification.outcome(simulator, "profile", "not-run", "Shared report preflight failed", remedy="Install or repair baseline reporting dependencies, then rerun.")
        return verification.finish()
    for simulator in selected:
        if simulator not in PROFILES:
            verification.outcome(simulator, "profile", "unsupported", "No bounded installation profile is declared",
                remedy="Select a declared profile with --sim; validate this adapter separately before use.")
            continue
        if simulator != "mock" and (simulator not in statuses or not statuses[simulator].ready):
            status = statuses.get(simulator)
            verification.outcome(simulator, "profile", "not-run", "Simulator prerequisites unavailable in the active environment",
                details={} if status is None else {"missing": list(status.missing), "load_error": status.load_error},
                remedy="Use the simulator's supported environment and assets, then rerun with --sim " + simulator)
            continue
        if simulator != "mock":
            ok, reason, details = native_prerequisites(simulator, verification.env, working_directory=verification.directory / simulator)
            verification.outcome(simulator, "assets", "pass" if ok else "not-run", reason, details=details,
                remedy="" if ok else "Configure the existing supported task assets before rerunning; verification does not download them.")
            if not ok:
                continue
        verification.profile(simulator)
    return verification.finish()
