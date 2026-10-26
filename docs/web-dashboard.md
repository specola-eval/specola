# Web dashboard

The web dashboard shows camera frames, an orbitable 3D scene, bindings,
robustness plots, and transitions while an episode runs. The same
interface can replay a saved recording without the simulator.

A second tab, **Discover**, browses the scene the episode was compiled
against: entity classes, their signals, and every way of selecting one.
It is available whenever the run carries a discovery snapshot, which
every `--web` run writes.

## Open the dashboard

```bash
pip install -e ".[web]"

# Live run
specola run --sim robocasa --specs my_specs.yaml --web

# Saved run
specola view results/<run_id>/recording_000

# A scene, with no episode: browse and author against a simulator
specola discover --sim robocasa --web
```

`discover --web` announces its URL on **stderr**, so `--format json`
keeps stdout to itself and the two compose: with both flags the document
is written first, then the browser opens. It serves until Ctrl-C.

Without an explicit `--ui`, the terminal shows a one-line notice plus the
final summary while `--web` serves the dashboard. The browser is the
viewport, so the Textual worker-thread overhead is not paid for a
terminal nobody is watching. Pass `--ui textual` or `--ui rich` to also
keep a live terminal UI beside the dashboard; the dashboard works
identically with all three `--ui` modes.

## Flags

- `--web [PORT]`: enable the dashboard and choose a port. The default is
  8090; `0` picks a free port. After the run ends the process keeps
  serving until Ctrl-C, so batch jobs should not pass `--web`.
- `--web-cameras NAME[,NAME...]|all`: choose cameras. The default is the
  adapter's default camera; `all` records every named camera.
- `--web-size WxH`: set the capture size. The default is 1280×720.
- `--web-stride N`: record every Nth frame. Monitoring data is recorded
  every step regardless; only frames thin out.
- `--web-video {auto,nvenc,jpeg}`: how camera frames are stored and
  shipped. `auto` (the default) probes the hardware h264 encoder once
  and uses it when present; otherwise, and whenever the pipeline
  cannot be started, the per-step JPEG store runs exactly as before;
  the fallback is the full feature, not a reduced one. A pipeline
  failure mid-run stops frames while monitoring data continues.

The server binds to `127.0.0.1`, has no authentication, and answers only
requests whose `Host` header names a loopback address. That last check is
what survives DNS rebinding: a page on any hostname that resolves to
127.0.0.1 otherwise presents a self-consistent `Host` and `Origin` pair
and passes a same-origin check. The port is not compared, so forwarding
to a different local port works. Do not expose the server directly to a
network. To view it from another machine, forward the
port over SSH:

```bash
ssh -L 8090:127.0.0.1:8090 <host>
```

## Recordings

The run writes its recording as it progresses:

```
results/<run_id>/recording_000/
  manifest.json          # run identity, cameras, capture size, final status
  snapshots.jsonl[.gz]   # one line per step; gzipped when the episode ends
  discovery.json.gz      # the scene this episode compiled against
  frames/<camera>/       # one JPEG per sampled step (jpeg store)
  video/<camera>/        # MP4 segments + index.json (video store)
  geometry.json          # 3D scene description, written once per episode
  meshes.bin             # mesh vertex/face buffer for geometry.json
  poses.bin              # one fixed-size body-pose record per step
```

`discovery.json.gz` is the scene the episode compiled against. The full
entity and signal catalogue plus the adapter's declared capabilities,
gzipped. It is written once, from the reset `Monitor.watch()` performs,
which is the reset the specifications are compiled against; a simulator
that randomizes per reset (RoboCasa draws a fresh kitchen each time)
would otherwise be described by a scene the run never monitored. Measured
at 1.66 MiB on a RoboCasa kitchen, against the ~14 MB of geometry the
same recording already writes.

Recordings written before this existed, and any whose capture failed,
replay exactly as before; only the Discover tab reports itself
unavailable.

The manifest's `frame_store` key says which frame store a recording
uses. Video recordings hold all-intra h264 in ~4-second MP4 segments, 
independently playable in any player, with an index mapping each
sampled step to its exact frame, so scrubbing behaves exactly like the
JPEG store. Old JPEG recordings replay unchanged, forever.

The three scene files exist only for adapters that expose a rigid-body
scene: the MuJoCo family, and Isaac Lab, whose adapter extracts the
environment's USD meshes. Without them the dashboard simply has no 3D
panel; everything else is unchanged.

Live viewing, scrub-back, and replay all use these files. Closing the
browser does not stop recording. An interrupted run remains readable,
although its manifest has no final status.

### Browser support for video recordings

Chrome and Safari decode the h264 stream natively (WebCodecs). Firefox
currently ships the API but rejects this stream. The camera pane then
shows a banner and the run can be repeated with `--web-video jpeg`.
Everything except the camera pixels (tree, plots, transitions, the 3D
panel, scrubbing) works identically in every browser either way.

### Environment requirements and common warnings

The environment that runs the monitor needs the `web` extra
(`pip install -e ".[web]"`: Flask, Flask-SocketIO, python-socketio,
Pillow). A policy-server process on the other side of a websocket never
runs the monitor and needs none of this.

Camera frames are JPEG-encoded by probing `cv2` first, then Pillow.
`cv2 unavailable, using PIL JPEG encoder` is that fallback announcing
itself once. Monitoring and the video feed proceed normally on Pillow;
install `opencv-python` only if the encode step itself shows up in
profiles. If both encoders are missing the dashboard runs without camera
frames and says so; nothing else degrades. A missing `flask_socketio`
(the extra was never installed in this environment) is the one failure
that disables `--web` entirely. The run continues, monitoring intact,
with a `Web warning: ... continuing without dashboard` line.

## Use the dashboard

- The scrub bar detaches you from the live head; the LIVE button
  re-attaches. In replay, play/pause and a rate selector appear.
- Click a specification to plot only its bindings. Click a binding to open
  its inspector and emphasize its series. Click it again or press Escape
  to return to the full view.
- Click a transition to seek to its step. Search and the state chips
  filter the tree; specifications with no matches are hidden.
- With more than one recorded camera, the ▦ cameras button toggles a
  tile grid; clicking a tile promotes that camera.
- The 3D panel sits beside the camera pane on runs with a scene:
  left-drag orbits, wheel zooms, right-drag pans. Checkboxes toggle
  geom groups; the recording says which start visible. The
  simulator's own render convention, so robosuite-family runs hide
  their collision hulls (group 0) exactly as their cameras do, with
  the plain-MuJoCo 0–2 default as the fallback. Isaac Lab exports
  follow the same split: collision hulls in hidden group 0, visual
  meshes in group 1. The 3D
  button in the status strip collapses the panel. Scrubbing moves the
  scene to the sought step along with everything else.
- Scrubbing to a step with no sampled frame shows the nearest earlier
  frame, captioned with its true step.
- The Stop button ends the live run; the run and its recording are
  marked `truncated`.

## Performance

Capturing the first camera costs about 17 ms per step at 640×480 and
21 ms at the 1280×720 default on a RoboCasa-scale scene (measured
2026-08-02; the cost is per-render, not per-pixel, so smaller frames buy
little). The 3D panel's pose broadcast adds no measurable step cost: an
interleaved 12-run A/B on the same scene (2026-08-03) measured
91.7 ms/step with poses against 91.9 without, a delta far inside the
run-to-run spread. Poses are data the extraction layer already
touches, so only two small array copies land on the simulator thread.
The one-time geometry export at the first step costs a scene build
(~14 MB of meshes on a RoboCasa kitchen) once per episode. JPEG
encoding runs off the simulator thread but contends with it for
roughly another 15 ms per step; the hardware video store removes that
contention. An interleaved 12-run A/B (2026-08-03, two 720p cameras,
RoboCasa scene) measured 102.8 ms/step under `--web-video jpeg` against
85.4 under the hardware encoder, a 1.20x step-rate gain
(9.7 to 11.7 steps/s) with the two arms' run distributions fully
separated. Additional cameras appear to cost far
less than the first. A live one-versus-six comparison moved the step
rate by only a few percent, consistent with the render context being
paid once, though the marginal cost has not been isolated on a pinned
scene. Overload thins frames, never monitoring data, and a slow or
hidden browser drops its own frames without slowing the run. Measured
localhost capture-to-browser latency is about 8 ms median for one
camera. `--web-stride 2` recovers most of the capture cost when the
per-step budget is tight. The terminal is summary-only by default under
`--web` (see above), so that terminal-UI cost is only paid when a live terminal UI
is explicitly requested too. In that case, on a laptop-class machine
`--ui rich` runs measurably faster than `--ui textual` beside a live
dashboard.

### Choosing cameras

Cameras feed the video panes and nothing else: verdicts, robustness, and
transitions come from extracted state, so a run monitors identically with
one camera, six, or none. That makes the camera flags a pure
looks-versus-speed dial:

- Watching a run day to day: keep the default single camera.
- Comparing viewpoints (occlusion, hand-offs): name the two or three you
  want, e.g. `--web-cameras robot0_agentview_center,robot0_eye_in_hand`.
- Recording every angle: `--web-cameras all`, and add `--web-stride 3` to
  pay a third of the capture cost: frames thin, monitoring data does not.
- Batch or sweep runs nobody watches: omit `--web` entirely; results are
  identical and the run sheds all capture and recording overhead.

## Command palette

`Ctrl-K` (`Cmd-K` on macOS) opens a palette that spans both views,
because what you want to reach is often not in the one you are looking
at: a spec name while browsing the scene, a class while watching the
monitor. It lists views, specs, bindings and entity classes, jumps to
whichever you pick, and reads its entries from the rendered trees at open
time rather than from a parallel index, so there is nothing that can go
stale against the run.

## Fonts

IBM Plex Sans and JetBrains Mono are vendored under
`static/vendor/fonts` and served by the dashboard itself, not fetched
from a font CDN. The server binds to loopback and is used on machines
that are often offline mid-experiment, and a third-party font request
would also tell that third party when somebody is looking at a run.
Licences and provenance: `static/vendor/NOTICE.txt`.

## Checking the dashboard

Two harnesses under `benchmarks/`, both runnable against any recording
that carries a discovery sidecar:

    python benchmarks/workbench_browser_check.py --recording <recording>

drives headless Chrome over the DevTools protocol and asserts eight
things: the monitor renders, the canvases read the same colour tokens
the stylesheet does, the vendored fonts load, Discover indexes the
scene, the palette refuses what the compiler refuses with nothing left
unverified, every offered form ships a starter or says why it withheld
one, the command palette opens and matches, and the page logged no
errors of its own. It exits non-zero on failure.

    python benchmarks/monitor_path_ab.py --recording <recording>

times connect-to-`run_meta`, a seek and a transitions page through the
socket, so a change to the dashboard can be compared against another
checkout rather than argued about.
