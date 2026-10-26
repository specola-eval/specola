/* specola dashboard client.
 *
 * Patterns from HRC monitor/dashboard/static/dashboard.js:
 * createImageBitmap decode with a decode token, rAF-batched dirty
 * rendering, DOM-diffed lists, paint-gated acks. Envelope guard is
 * per-connection: the server bumps our epoch on every accepted view
 * change; a higher epoch always wins, stale seq within an epoch is
 * rejected.
 */

"use strict";

const el = (id) => document.getElementById(id);

/* Canvas colours come from the same :root tokens the stylesheet uses.
 *
 * They used to be hardcoded here, and when the palette was retuned the
 * plot's axes and labels kept drawing in the old greys while every
 * styled element moved. Two palettes that happen to agree is a thing
 * that stops being true silently, and a canvas cannot inherit. */
const INK = (() => {
  const root = getComputedStyle(document.documentElement);
  const read = (name, fallback) =>
    root.getPropertyValue(name).trim() || fallback;
  return {
    text: read("--text", "#d3dae1"),
    muted: read("--muted", "#79838f"),
    border: read("--border", "#1e242b"),
    accent: read("--accent", "#5fb3a1"),
    bad: read("--bad", "#e2857a"),
  };
})();

const model = {
  meta: null,
  mode: "live",            // live | ended | view
  epoch: 0,
  latestSeq: -1,
  decodeToken: 0,
  step: 0,
  minStep: 0,              // tracker timesteps need not start at 0
  maxStep: 0,
  simTime: 0,
  follow: true,
  playing: false,
  rate: 1,
  cameras: [],
  camera: null,            // promoted camera
  gridOpen: false,
  bitmap: null,
  frameStep: null,         // actual step of the decoded frame
  tileBitmaps: new Map(),  // camera -> {bitmap, step, token}
  specs: new Map(),        // name -> {authored_formula}
  bindings: new Map(),     // id = spec + "NUL" + key -> entry
  order: [],               // binding ids in first-seen order (color index)
  collapsed: new Set(),    // spec names
  transitions: [],         // step-sorted; deduped via transitionKeys
  transitionKeys: new Set(),
  transitionsOffset: null, // server offset for "load earlier"
  selection: null,         // null | {type: "spec"|"binding", id}
  filterText: "",
  filterStates: new Set(),
  drops: { frames: 0, delivery: 0 },
  seriesCache: new Map(),  // scopeKey:px -> [[spec, key, points], ...]
  dirty: { status: true, camera: false, tree: false, plot: false,
           transitions: false, inspector: false, timeline: true },
  renderScheduled: false,
};

const bindingId = (spec, key) => spec + "\u0000" + key;
const idSpec = (id) => id.split("\u0000")[0];
const idKey = (id) => id.split("\u0000")[1];

/* ---- render scheduling ------------------------------------------------ */

function markDirty(...parts) {
  for (const part of parts) model.dirty[part] = true;
  scheduleRender();
}

function scheduleRender() {
  if (model.renderScheduled) return;
  model.renderScheduled = true;
  requestAnimationFrame(commitRender);
}

function commitRender() {
  model.renderScheduled = false;
  if (model.dirty.status) renderStatus();
  if (model.dirty.camera) renderCamera();
  if (model.dirty.tree) renderTree();
  if (model.dirty.plot) renderPlot();
  if (model.dirty.transitions) renderTransitions();
  if (model.dirty.inspector) renderInspector();
  if (model.dirty.timeline) renderTimeline();
}

/* ---- ingest ------------------------------------------------------------ */

const transitionKey = (t) => `${t.step}|${t.spec}|${t.key}|${t.to}`;

function addTransitions(list) {
  // Point-read bundles re-deliver historical deltas and the paged log
  // overlaps the live tail: dedup by key, keep step-sorted.
  let appended = false;
  let outOfOrder = false;
  for (const t of list) {
    const key = transitionKey(t);
    if (model.transitionKeys.has(key)) continue;
    model.transitionKeys.add(key);
    const last = model.transitions[model.transitions.length - 1];
    if (last && t.step < last.step) outOfOrder = true;
    model.transitions.push(t);
    appended = true;
  }
  if (outOfOrder) model.transitions.sort((a, b) => a.step - b.step);
  return appended;
}

function registerBinding(metaEntry) {
  const id = bindingId(metaEntry.spec, metaEntry.key);
  if (model.bindings.has(id)) return;
  model.bindings.set(id, {
    meta: metaEntry, state: "pending", active: true, rho: null,
    worst: null, final: null, first_violation: null, error: null,
    series: { steps: [], values: [] },
  });
  model.order.push(id);
}

function ingestMeta(meta) {
  // Re-emitted on new bindings during live runs: must never disturb an
  // existing connection's position, follow state, or camera choice.
  const first = model.meta === null;
  // A different run means different streams: drop the decoders (cheap,
  // and every unit is a keyframe) and let the new run re-probe support.
  // An epoch bump does not — seeks and camera changes keep decoding the
  // same all-intra stream, and rebuilding a hardware decoder per scrub
  // step would be the expensive way to change nothing.
  if (!first && meta.run_id !== model.meta.run_id) resetVideoDecoders();
  model.meta = meta;
  model.mode = meta.mode;
  model.minStep = meta.min_step ?? 0;
  model.maxStep = Math.max(model.maxStep, meta.max_step);
  for (const name of meta.manifest.cameras || []) {
    if (!model.cameras.includes(name)) model.cameras.push(name);
  }
  model.camera = model.camera || model.cameras[0] || null;
  for (const s of meta.specs || []) model.specs.set(s.name, s);
  for (const b of meta.bindings || []) registerBinding(b);
  el("view-command").textContent = meta.view_command || "";
  if (first && meta.mode !== "live") {
    enterReplayMode();
    if (meta.max_step >= model.minStep) {
      seekTo(model.minStep);             // first recorded step, not 0:
    }                                    // tracker steps may start at 1
  }
  markDirty("status", "camera", "tree", "plot", "transitions", "timeline");
}

async function ingestBundle(bundle, acknowledge) {
  if (bundle.epoch < model.epoch ||
      (bundle.epoch === model.epoch && bundle.seq <= model.latestSeq)) {
    acknowledge?.({ ignored: true });
    return;
  }
  model.epoch = bundle.epoch;
  model.latestSeq = bundle.seq;
  model.step = bundle.step;
  model.simTime = bundle.sim_time;
  model.maxStep = Math.max(model.maxStep, bundle.max_step);
  model.drops = bundle.drops || model.drops;
  // Ahead of the frame decode below: the 3D scene has no decode step and
  // should not wait on one. Every bundle path carries poses, so seek and
  // replay move the scene with no extra code.
  if (bundle.poses?.bytes) scene3d("setPoses", bundle.poses.bytes);
  for (const s of bundle.new_specs || []) model.specs.set(s.name, s);
  for (const b of bundle.new_bindings || []) registerBinding(b);
  for (const spec of bundle.specs || []) {
    for (const b of spec.bindings) {
      const entry = model.bindings.get(bindingId(spec.name, b.key));
      if (!entry) continue;
      Object.assign(entry, {
        state: b.state, active: b.active, rho: b.rho, worst: b.worst,
        final: b.final, first_violation: b.first_violation, error: b.error,
      });
      // Local tails stay step-monotonic: point-read bundles arrive out
      // of order while scrubbing; gaps are filled by series_window.
      const steps = entry.series.steps;
      if (steps.length === 0 || bundle.step > steps[steps.length - 1]) {
        steps.push(bundle.step);
        entry.series.values.push(b.rho);
      }
    }
  }
  // Bundles carry the step's complete live binding population. A
  // binding absent from the bundle does not exist at this step, so any
  // state it carries is a ghost from another step viewed earlier
  // (backward scrub, or a playback pass wrapping around) — dynamic
  // contact-pair bindings otherwise stay "violated" over the whole
  // prefix and desync the tree from the camera. Gated on detachment
  // rather than mode: following the live head keeps a pruned binding's
  // last state visible (retention), while any time-travel — replay or
  // a live-run scrub — shows the truth of the step on screen.
  if (!model.follow) {
    const present = new Set();
    for (const spec of bundle.specs || []) {
      for (const b of spec.bindings) present.add(bindingId(spec.name, b.key));
    }
    for (const [id, entry] of model.bindings) {
      if (!present.has(id) && (entry.active || entry.state !== "pending")) {
        Object.assign(entry, {
          state: "pending", active: false, rho: null, worst: null,
          final: null, first_violation: null, error: null,
        });
      }
    }
  }
  addTransitions(bundle.transitions || []);
  for (const name of Object.keys(bundle.frames || {})) {
    if (!model.cameras.includes(name)) model.cameras.push(name);
    if (!model.camera) model.camera = name;
  }

  let painted = false;
  const frames = bundle.frames || {};
  if (model.gridOpen) {
    painted = await decodeTiles(frames);
  } else {
    const frame = model.camera && frames[model.camera];
    if (frame && frame.bytes) {
      const token = ++model.decodeToken;
      try {
        const image = await decodeFrame(model.camera, frame);
        if (!image) {
          /* codec this browser cannot decode: keep the last image */
        } else if (token === model.decodeToken) {
          model.bitmap?.close?.();
          model.bitmap = image;
          model.frameStep = frame.step;
          painted = true;
        } else {
          image.close();
        }
      } catch (_) { /* stale/corrupt frame: keep the last bitmap */ }
    }
  }
  markDirty("status", "camera", "tree", "plot", "transitions", "timeline");
  if (model.selection?.type === "binding") markDirty("inspector");
  // Paint-gated ack (HRC pattern): ack after the frame is committed.
  requestAnimationFrame(() =>
    acknowledge?.({ epoch: bundle.epoch, seq: bundle.seq, painted }));
}

async function decodeTiles(frames) {
  let any = false;
  for (const [name, frame] of Object.entries(frames)) {
    if (!frame || !frame.bytes) continue;
    const slot = model.tileBitmaps.get(name) || { token: 0 };
    const token = ++slot.token;
    model.tileBitmaps.set(name, slot);
    try {
      const image = await decodeFrame(name, frame);
      if (!image) {
        /* codec this browser cannot decode: keep the last image */
      } else if (token === slot.token) {
        slot.bitmap?.close?.();
        slot.bitmap = image;
        slot.step = frame.step;
        any = true;
      } else {
        image.close();
      }
    } catch (_) { /* keep last */ }
  }
  return any;
}

/* ---- frame decode -------------------------------------------------------
 * Frame entries name their codec. JPEG entries take the createImageBitmap
 * path they always did; h264 entries go through one VideoDecoder per
 * camera. Both decode sites (the single pane above and the grid) call
 * decodeFrame, and both get back something drawImage accepts and close()
 * releases, so the paint-gated ack and the token guards are unchanged.
 *
 * Every access unit is an IDR — the recorder encodes all-intra — so units
 * are independent: a decoder can be rebuilt at any point without losing
 * sync, and out-of-order scrubbing needs no reference frames.
 */

const videoDecoders = new Map();     // camera -> {codec, decoder, waiters}
const videoUnsupported = new Set();  // cameras whose decode failed once
let videoBannerDismissed = false;

const isVideoCodec = (codec) =>
  typeof codec === "string" && codec.startsWith("avc");

// An ImageBitmap reports width/height, a decoded VideoFrame reports its
// display size instead; both are drawable.
const imageWidth = (image) => image.displayWidth ?? image.width;
const imageHeight = (image) => image.displayHeight ?? image.height;

function decodeFrame(camera, frame) {
  if (!isVideoCodec(frame.codec)) {
    return createImageBitmap(new Blob([frame.bytes], { type: "image/jpeg" }));
  }
  return decodeAccessUnit(camera, frame);
}

function videoDecoderFor(camera, codec) {
  const existing = videoDecoders.get(camera);
  if (existing && existing.codec === codec) return existing;
  if (existing) closeVideoDecoder(camera);
  const state = { codec, decoder: null, waiters: new Map() };
  // Throws where WebCodecs is missing entirely; the caller treats that
  // like any other decode failure.
  state.decoder = new VideoDecoder({
    output: (videoFrame) => {
      const waiter = state.waiters.get(videoFrame.timestamp);
      if (!waiter) {
        videoFrame.close();          // stale output: nobody is waiting
        return;
      }
      state.waiters.delete(videoFrame.timestamp);
      waiter.resolve(videoFrame);
    },
    error: (error) => failVideoDecode(camera, error),
  });
  // No `description`: the units are Annex-B with in-band SPS/PPS, which
  // is the mode WebCodecs assumes when the description is absent.
  state.decoder.configure({ codec });
  videoDecoders.set(camera, state);
  return state;
}

// Resolves to a VideoFrame, or to null when this camera's decoder is
// unusable. Never rejects and never hangs: the caller awaits this before
// setting `painted`, so an unsettled promise would stall the ack.
async function decodeAccessUnit(camera, frame) {
  if (videoUnsupported.has(camera)) return null;
  let state;
  try {
    state = videoDecoderFor(camera, frame.codec);
  } catch (error) {
    failVideoDecode(camera, error);
    return null;
  }
  const timestamp = frame.step;
  // The waiter object is its own token: only the decode that installed it
  // may settle it, so a superseded decode of the same step can neither
  // steal the frame nor mark the newer decode painted.
  const waiter = { resolve: null, promise: null };
  waiter.promise = new Promise((resolve) => { waiter.resolve = resolve; });
  const previous = state.waiters.get(timestamp);
  state.waiters.set(timestamp, waiter);
  if (previous) previous.resolve(null);
  try {
    state.decoder.decode(new EncodedVideoChunk(
      { type: "key", timestamp, data: frame.bytes }));
    // flush() bounds the queue at one unit and forces the output out:
    // WebCodecs is otherwise free to hold decoded frames back, which
    // would stall the paint-gated ack behind them.
    await state.decoder.flush();
  } catch (error) {
    // The output may already have delivered a frame into this waiter
    // before flush() rejected; nobody will await it now, so close it.
    waiter.promise.then((decoded) => decoded?.close?.());
    failVideoDecode(camera, error);
    return null;
  }
  if (state.waiters.get(timestamp) === waiter) {
    state.waiters.delete(timestamp);
    waiter.resolve(null);            // flushed with no output for this unit
  }
  return waiter.promise;
}

function closeVideoDecoder(camera) {
  const state = videoDecoders.get(camera);
  if (!state) return;
  videoDecoders.delete(camera);
  for (const waiter of state.waiters.values()) waiter.resolve(null);
  state.waiters.clear();
  try { state.decoder.close(); } catch (_) { /* already closed */ }
}

// Firefox ships the API, reports h264 supported from isConfigSupported,
// and then fails the decode — so the probe is the first real access unit,
// and its failure retires the codec for this camera rather than retrying
// per frame.
function failVideoDecode(camera, error) {
  videoUnsupported.add(camera);
  closeVideoDecoder(camera);
  console.warn("h264 decode failed for camera", camera, error);
  showVideoBanner();
}

function showVideoBanner() {
  if (videoBannerDismissed || el("video-banner")) return;
  const banner = document.createElement("div");
  banner.id = "video-banner";
  banner.textContent =
    "this browser cannot decode h264 video — rerun with --web-video jpeg";
  const dismiss = document.createElement("button");
  dismiss.textContent = "×";
  dismiss.title = "dismiss";
  dismiss.addEventListener("click", () => {
    videoBannerDismissed = true;
    banner.remove();
  });
  banner.appendChild(dismiss);
  el("camera-pane").appendChild(banner);
}

function resetVideoDecoders() {
  for (const camera of Array.from(videoDecoders.keys())) {
    closeVideoDecoder(camera);
  }
  videoUnsupported.clear();
}

function ingestStatus(payload) {
  model.maxStep = Math.max(model.maxStep, payload.max_step);
  model.drops = payload.drops || model.drops;
  markDirty("status", "timeline");
}

function ingestEpisodeEnd(payload) {
  model.mode = "ended";
  enterReplayMode();
  model.maxStep = Math.max(model.maxStep, payload.max_step ?? 0);
  el("ended-banner").hidden = false;
  el("ended-status").textContent = payload.status;
  el("view-command").textContent = payload.view_command || "";
  markDirty("status", "timeline");
}

/* ---- render: status / camera / timeline -------------------------------- */

function setText(element, value) {
  if (element.textContent !== value) element.textContent = value;
}

function renderStatus() {
  model.dirty.status = false;
  if (!model.meta) return;
  const manifest = model.meta.manifest || {};
  setText(el("run-id"), model.meta.run_id || "");
  // Recordings written before the manifest stopped storing the
  // "(default)" placeholder still carry it; filter both spellings.
  const runLabel = [manifest.simulator, manifest.task]
    .filter((part) => part && part !== "(default)")
    .join(" · ");
  setText(el("run-task"), runLabel);
  setText(el("run-state"), model.mode);
  setText(el("drop-counts"),
          `drops ${model.drops.frames || 0}/${model.drops.delivery || 0}`);
  el("stop-run").hidden = model.mode !== "live";
}

function renderCamera() {
  model.dirty.camera = false;
  const pane = el("camera-pane");
  const toggle = el("camera-grid-toggle");
  toggle.hidden = model.cameras.length < 2;
  toggle.classList.toggle("open", model.gridOpen);
  el("camera-grid").hidden = !model.gridOpen;
  el("camera").hidden = model.gridOpen;
  if (model.gridOpen) {
    renderCameraGrid();
    pane.classList.toggle("has-frames", model.tileBitmaps.size > 0);
    return;
  }
  if (!model.bitmap) return;
  pane.classList.add("has-frames");
  const canvas = el("camera");
  canvas.width = imageWidth(model.bitmap);
  canvas.height = imageHeight(model.bitmap);
  canvas.getContext("2d").drawImage(model.bitmap, 0, 0);
  const caption = el("frame-caption");
  const stale = model.frameStep !== null && model.frameStep !== model.step;
  caption.hidden = !stale;
  if (stale) setText(caption, `frame from step ${model.frameStep}`);
}

function renderTimeline() {
  model.dirty.timeline = false;
  const scrub = el("scrub");
  scrub.min = model.minStep;
  scrub.max = model.maxStep;
  if (!scrubbing) scrub.value = model.step;
  setText(el("step-label"),
          `step ${model.step}/${model.maxStep} · t=${
            (model.simTime ?? 0).toFixed(2)}s`);
  const live = el("follow-live");
  live.hidden = model.mode !== "live";
  live.classList.toggle("active", model.follow);
  drawTicks();
}

function drawTicks() {
  const canvas = el("ticks");
  canvas.width = canvas.clientWidth || 1;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = INK.bad;
  const span = Math.max(1, model.maxStep - model.minStep);
  for (const t of model.transitions) {
    const x = ((t.step - model.minStep) / span) * canvas.width;
    ctx.fillRect(x, 0, 1.5, canvas.height);
  }
}

function unlockReplayControls() {
  el("play-pause").hidden = false;
  el("rate").hidden = false;
}

// Shared transition for the two paths that leave live-follow for good:
// a non-live first meta (view mode) and episode end.
function enterReplayMode() {
  model.follow = false;
  unlockReplayControls();
  loadEarlierTransitions();              // replay: pull the log history
}

/* ---- controls ----------------------------------------------------------- */

let scrubbing = false;
const socket = io();
// Shared with discover.js, which needs the same connection rather than
// a second one: two sockets would be two client states on the server.
window.specolaSocket = socket;

socket.on("connect", () => {
  /* A reconnect gets a fresh server-side client whose epoch restarts at
   * zero, while `model.epoch` is page-lifetime state that only ever
   * moves up. One view change before a reconnect -- a camera toggle, a
   * scrub, pressing play -- leaves the page permanently ahead of the
   * server, and every live bundle after that is dropped as stale with
   * nothing on screen to say so. Measured: 10 of 10 bundles dropped
   * after one camera change and a reconnect.
   *
   * The server documents that reconnects happen (a saturated step phase
   * can stall packet handling past the ping timeout, and the client
   * reconnects), so this is the ordinary path, not an edge. */
  model.epoch = 0;
  model.latestSeq = -1;
});

function sendView(patch) {
  socket.emit("view", patch, (ack) => {
    if (ack?.ok && typeof ack.epoch === "number") {
      model.epoch = ack.epoch;
      model.latestSeq = -1;
    }
  });
}

function currentCameraSelection() {
  if (model.gridOpen) return model.cameras.slice();
  return model.camera ? [model.camera] : undefined;
}

function setPlaying(value) {
  // Single authority for the play flag AND its glyph: a seek pauses
  // playback server-side (the epoch bump kills the stream), and the
  // LIVE button clears the flag — both previously left the glyph
  // showing pause, so resuming took two clicks.
  model.playing = value;
  el("play-pause").textContent = value ? "⏸" : "▶";
}

function seekTo(step) {
  model.follow = false;
  model.step = step;
  const view = { seek: step, cameras: currentCameraSelection() };
  if (model.playing) {
    setPlaying(false);
    view.play = false;
  }
  sendView(view);
  markDirty("timeline", "status");
}

el("scrub").addEventListener("input", (event) => {
  scrubbing = true;
  seekTo(Number(event.target.value));
});
["change", "pointerup"].forEach((kind) =>
  el("scrub").addEventListener(kind, () => { scrubbing = false; }));

el("follow-live").addEventListener("click", () => {
  if (model.mode !== "live") return;
  model.follow = true;
  setPlaying(false);
  sendView({ follow: true, cameras: currentCameraSelection() });
  markDirty("timeline", "status");
});

el("stop-run").addEventListener("click", () => {
  if (window.confirm("Stop the running episode?")) {
    socket.emit("stop_run", {}, () => {});
  }
});

el("play-pause").addEventListener("click", () => {
  setPlaying(!model.playing);
  sendView({ play: model.playing, rate: model.rate,
             cameras: currentCameraSelection() });
});

el("rate").addEventListener("change", (event) => {
  model.rate = Number(event.target.value);
  sendView({ rate: model.rate, play: model.playing });
});

/* ---- socket -------------------------------------------------------------- */

socket.on("run_meta", ingestMeta);
socket.on("step_bundle", (bundle, ack) => { ingestBundle(bundle, ack); });
socket.on("run_status", ingestStatus);
socket.on("episode_end", ingestEpisodeEnd);
socket.on("geometry", ingestGeometry);
// Relayed as an event rather than handled here: the Discover view owns
// its own state, and the monitor must not gain a reason to re-render
// when the scene arrives.
socket.on("discovery_ready", (payload) => {
  window.dispatchEvent(
    new CustomEvent("specola-discovery", { detail: payload }));
});
window.addEventListener("resize", () => markDirty("plot", "timeline"));

/* ---- selection, colors, scope --------------------------------------------
 * Line/swatch color identifies the binding (CVD palette); pill color
 * carries the state. The two encodings are never mixed.
 */

// Categorical, not semantic: these identify which binding a line
// belongs to, so they stay four distinguishable hues rather than
// collapsing onto the accent. Retuned to sit on the new ground while
// keeping the CVD-safe separation the original set had.
const SERIES_PALETTE = ["#5aa0d6", "#c98bb0", "#c2a15a", "#5fb3a1"];
const STATE_ORDER = ["pending", "satisfied", "violated", "suspended",
                     "rewarming", "missing_data", "error"];
// Worst binding state wins the spec row's color (run_dashboard.py
// _SPEC_STYLE_PRECEDENCE).
const STATE_PRECEDENCE = ["violated", "error", "missing_data", "pending",
                          "rewarming", "suspended", "satisfied"];

function seriesColor(index) {
  if (index < SERIES_PALETTE.length) return SERIES_PALETTE[index];
  const hue = (210 + index * 137.508) % 360;   // golden-angle extension
  return `hsl(${hue} 55% 55%)`;
}

const bindingColor = (id) => seriesColor(model.order.indexOf(id));

// Binding whose state ranks worst in STATE_PRECEDENCE; null when no
// id carries a ranked state.
function worstBinding(ids) {
  let best = null;
  let bestRank = STATE_PRECEDENCE.length;
  for (const id of ids) {
    const rank = STATE_PRECEDENCE.indexOf(model.bindings.get(id).state);
    if (rank !== -1 && rank < bestRank) {
      bestRank = rank;
      best = id;
    }
  }
  return best;
}

function worstSpec() {
  const worst = worstBinding(model.order);
  if (worst !== null) return idSpec(worst);
  return model.order.length ? idSpec(model.order[0]) : null;
}

function scopeSpec() {
  if (model.selection?.type === "binding") {
    return idSpec(model.selection.id);
  }
  return model.selection?.type === "spec" ? model.selection.id : worstSpec();
}

function scopedBindingIds() {
  // The main plot always shows the whole scoped spec; a selected
  // binding is emphasized there and gets its solo series in the
  // inspector sparkline (no duplicated single-series plot).
  const spec = scopeSpec();
  return model.order.filter((id) => idSpec(id) === spec);
}

function select(kind, id) {
  const same = model.selection &&
    model.selection.type === kind && model.selection.id === id;
  if (same && kind === "binding") {
    model.selection = { type: "spec", id: idSpec(id) };   // widen to spec
  } else if (same) {
    model.selection = null;                               // default scope
  } else {
    model.selection = { type: kind, id };
  }
  el("inspector").hidden = model.selection?.type !== "binding";
  markDirty("tree", "plot", "inspector", "transitions");
}

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !model.selection) return;
  // Re-selecting the current selection widens binding → spec → clear,
  // which is exactly Escape's contract.
  select(model.selection.type, model.selection.id);
});

/* ---- tree ----------------------------------------------------------------- */

const filteredSpecState = new Map();  // spec -> pill state under filter

function specWorstState(specName) {
  const override = filteredSpecState.get(specName);
  if (override) return override;
  const worst = worstBinding(
    model.order.filter((id) => idSpec(id) === specName));
  return worst !== null ? model.bindings.get(worst).state : "pending";
}

function bindingVisible(id, entry) {
  if (model.filterStates.size && !model.filterStates.has(entry.state)) {
    return false;
  }
  if (!model.filterText) return true;
  const needle = model.filterText.toLowerCase();
  return (entry.meta.label || "").toLowerCase().includes(needle) ||
         id.toLowerCase().includes(needle);
}

function formatRho(value) {
  return value === null || value === undefined ? "—" : value.toFixed(3);
}

function buildChips() {
  const host = el("state-chips");
  if (host.childElementCount) return;
  for (const state of STATE_ORDER) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.dataset.state = state;
    chip.textContent = state;
    chip.addEventListener("click", () => {
      if (model.filterStates.has(state)) model.filterStates.delete(state);
      else model.filterStates.add(state);
      chip.classList.toggle("on", model.filterStates.has(state));
      markDirty("tree");
    });
    host.appendChild(chip);
  }
}

el("search").addEventListener("input", (event) => {
  model.filterText = event.target.value.trim();
  markDirty("tree");
});

function treeRowFor(kind, id) {
  const li = document.createElement("li");
  li.dataset.kind = kind;
  li.dataset.key = id;
  if (kind === "spec") {
    li.innerHTML = '<span class="twist"></span><span class="label"></span>' +
                   '<span class="pill state"></span>';
    li.addEventListener("click", (event) => {
      if (event.target.classList.contains("twist")) {
        if (model.collapsed.has(id)) model.collapsed.delete(id);
        else model.collapsed.add(id);
        markDirty("tree");
      } else {
        select("spec", id);
      }
    });
  } else {
    li.innerHTML = '<span class="swatch"></span><span class="label"></span>' +
                   '<span class="rho"></span><span class="pill state"></span>';
    li.addEventListener("click", () => select("binding", id));
  }
  return li;
}

function renderTree() {
  model.dirty.tree = false;
  buildChips();
  const host = el("tree");
  const filtering = model.filterText || model.filterStates.size;
  filteredSpecState.clear();
  const wanted = [];                       // [kind, id] in display order
  for (const specName of model.specs.keys()) {
    const visible = model.order.filter(
      (id) => idSpec(id) === specName &&
              bindingVisible(id, model.bindings.get(id)));
    // A filter hides whole specs with no matching bindings — otherwise
    // every header survives and the filter looks like it did nothing.
    if (filtering && !visible.length) continue;
    if (filtering) {
      // The surviving header's pill must describe its VISIBLE children,
      // not bindings the filter hid.
      const worst = worstBinding(visible);
      filteredSpecState.set(
        specName,
        worst !== null ? model.bindings.get(worst).state : "pending");
    }
    wanted.push(["spec", specName]);
    if (model.collapsed.has(specName)) continue;
    for (const id of visible) wanted.push(["binding", id]);
  }
  // DOM diff keyed on data-key (HRC alert-list pattern).
  const existing = new Map(
    Array.from(host.children).map((node) => [
      node.dataset.kind + "\u0000" + node.dataset.key, node]));
  const seen = new Set();
  let cursor = host.firstChild;
  for (const [kind, id] of wanted) {
    const mapKey = kind + "\u0000" + id;
    seen.add(mapKey);
    let node = existing.get(mapKey);
    if (!node) node = treeRowFor(kind, id);
    if (node !== cursor) host.insertBefore(node, cursor);
    else cursor = cursor.nextSibling;
    updateTreeRow(node, kind, id);
  }
  for (const [mapKey, node] of existing) {
    if (!seen.has(mapKey)) node.remove();
  }
}

function updateTreeRow(node, kind, id) {
  const selected = model.selection &&
    model.selection.type === kind && model.selection.id === id;
  node.classList.toggle("selected", Boolean(selected));
  const pill = node.querySelector(".pill");
  if (kind === "spec") {
    setText(node.querySelector(".twist"),
            model.collapsed.has(id) ? "▸" : "▾");
    setText(node.querySelector(".label"), id);
    const state = specWorstState(id);
    setText(pill, state);
    pill.className = "pill state state-" + state;
  } else {
    const entry = model.bindings.get(id);
    node.querySelector(".swatch").style.background = bindingColor(id);
    setText(node.querySelector(".label"), entry.meta.label || idKey(id));
    setText(node.querySelector(".rho"), formatRho(entry.rho));
    setText(pill, entry.state);
    pill.className = "pill state state-" + entry.state;
  }
}

/* ---- plot ------------------------------------------------------------------ */

function fetchSeriesWindows(ids, px) {
  // The end bucket is part of the key: a growing live run must refetch
  // as maxStep advances or the cached history freezes at first fetch.
  const bucket = Math.ceil((model.maxStep + 1) / 500);
  const scopeKey = ids.join("|") + ":" + px + ":" + bucket;
  if (model.seriesCache.has(scopeKey)) {
    return model.seriesCache.get(scopeKey);
  }
  if (model.seriesCache.size > 32) model.seriesCache.clear();
  model.seriesCache.set(scopeKey, null);     // request in flight
  socket.emit(
    "series_window",
    { bindings: ids.map((id) => [idSpec(id), idKey(id)]),
      start: 0, end: Math.max(model.maxStep, 0), px },
    (ack) => {
      if (ack?.ok) {
        model.seriesCache.set(scopeKey, ack.series);
        markDirty("plot", "inspector");
      } else {
        model.seriesCache.delete(scopeKey);
      }
    });
  return null;
}

function seriesFor(id, px, fetchIds) {
  // Dense local tail first; ask the server for decimated history when
  // the tail cannot cover the domain.
  const entry = model.bindings.get(id);
  const local = entry.series;
  const domainSpan = model.maxStep + 1;
  const tailEnd = local.steps.length
    ? local.steps[local.steps.length - 1] : -1;
  const covered = local.steps.length &&
    local.steps[0] <= Math.max(0, domainSpan * 0.02) &&
    tailEnd >= model.maxStep - Math.max(2, domainSpan * 0.02) &&
    local.steps.length >= Math.min(domainSpan, px) * 0.5;
  if (covered) {
    const points = [];
    for (let i = 0; i < local.steps.length; i += 1) {
      if (local.values[i] === null) continue;
      points.push([local.steps[i], local.values[i], local.values[i]]);
    }
    return points;
  }
  const remote = fetchSeriesWindows(fetchIds || scopedBindingIds(), px);
  if (!remote) return null;
  const match = remote.find(
    (row) => row[0] === idSpec(id) && row[1] === idKey(id));
  return match ? match[2] : [];
}

function sizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 300;
  const height = canvas.clientHeight || 150;
  if (canvas.width !== Math.round(width * ratio) ||
      canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function formatAxisValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function scopeUnit(ids) {
  // One unit label only when every scoped binding agrees on it.
  const units = new Set(
    ids.map((id) => model.bindings.get(id).meta.unit || null));
  if (units.size === 1) {
    const [unit] = units;
    return unit || "";
  }
  return "";
}

function renderPlot() {
  model.dirty.plot = false;
  const ids = scopedBindingIds();
  const spec = scopeSpec();
  const unit = scopeUnit(ids);
  const unitSuffix = unit ? ` (${unit})` : "";
  setText(el("plot-title"), spec ? `ρ · ${spec}${unitSuffix}` : "ρ");
  const { ctx, width, height } = sizeCanvas(el("plot"));
  ctx.clearRect(0, 0, width, height);
  const LEFT = 48;                          // y-axis value gutter
  const BOTTOM = 14;                        // x-axis step labels
  const plotW = width - LEFT;
  const plotH = height - BOTTOM;
  ctx.font = "10px ui-monospace, Menlo, monospace";
  if (!ids.length) {
    // ρ=0 reference line is ALWAYS drawn, even with nothing scoped.
    ctx.strokeStyle = INK.muted;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(LEFT, plotH / 2);
    ctx.lineTo(width, plotH / 2);
    ctx.stroke();
    ctx.setLineDash([]);
    return;
  }
  const px = Math.max(50, Math.round(plotW));
  const perBinding = new Map();
  let min = 0;
  let max = 0;
  for (const id of ids) {
    const points = seriesFor(id, px, ids);
    if (!points) continue;                  // request in flight
    perBinding.set(id, points);
    for (const [, lo, hi] of points) {
      if (lo < min) min = lo;
      if (hi > max) max = hi;
    }
  }
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.05;
  min -= pad;
  max += pad;
  const minStep = model.minStep;
  const domain = Math.max(1, model.maxStep - minStep);
  const sx = (step) => LEFT + ((step - minStep) / domain) * plotW;
  const sy = (value) => plotH - ((value - min) / (max - min)) * plotH;

  // Axis values: y at max / 0 / min with the scope's unit, x in steps.
  ctx.fillStyle = INK.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  const axisSuffix = unit ? " " + unit : "";
  ctx.fillText(formatAxisValue(max) + axisSuffix, LEFT - 5, 1);
  ctx.textBaseline = "bottom";
  ctx.fillText(formatAxisValue(min) + axisSuffix, LEFT - 5, plotH - 1);
  if (min < 0 && max > 0) {
    ctx.textBaseline = "middle";
    ctx.fillText("0", LEFT - 5, sy(0));
  }
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillText(`step ${minStep}`, LEFT + 2, height - 3);
  ctx.textAlign = "right";
  ctx.fillText(String(model.maxStep), width - 2, height - 3);
  ctx.textAlign = "left";

  // ρ=0 reference line: always drawn.
  ctx.strokeStyle = INK.muted;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(LEFT, sy(0));
  ctx.lineTo(width, sy(0));
  ctx.stroke();
  ctx.setLineDash([]);

  const selectedId =
    model.selection?.type === "binding" ? model.selection.id : null;
  for (const [id, points] of perBinding) {
    if (!points.length) continue;
    const color = bindingColor(id);
    const emphasized = selectedId === null || selectedId === id;
    // min/max band
    ctx.beginPath();
    for (let i = 0; i < points.length; i += 1) {
      const [step, lo] = points[i];
      if (i === 0) ctx.moveTo(sx(step), sy(lo));
      else ctx.lineTo(sx(step), sy(lo));
    }
    for (let i = points.length - 1; i >= 0; i -= 1) {
      ctx.lineTo(sx(points[i][0]), sy(points[i][2]));
    }
    ctx.closePath();
    ctx.globalAlpha = emphasized ? 0.18 : 0.08;
    ctx.fillStyle = color;
    ctx.fill();
    // midline; the selected binding is emphasized, siblings dimmed
    ctx.globalAlpha = emphasized ? 1 : 0.45;
    ctx.lineWidth = selectedId === id ? 2.4 : 1.4;
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let i = 0; i < points.length; i += 1) {
      const [step, lo, hi] = points[i];
      const mid = (lo + hi) / 2;
      if (i === 0) ctx.moveTo(sx(step), sy(mid));
      else ctx.lineTo(sx(step), sy(mid));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
  }

  // current-step cursor
  ctx.strokeStyle = INK.text;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.moveTo(sx(model.step), 0);
  ctx.lineTo(sx(model.step), plotH);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/* ---- inspector --------------------------------------------------------------- */

function formatAt(rec) {
  if (!rec) return "—";
  const rho = rec.rho !== undefined ? formatRho(rec.rho) + " " : "";
  const time = rec.time === null || rec.time === undefined
    ? "" : ` (t=${rec.time.toFixed(2)}s)`;
  return `${rho}@ step ${rec.step}${time}`;
}

function renderInspector() {
  model.dirty.inspector = false;
  if (model.selection?.type !== "binding") return;
  const id = model.selection.id;
  const entry = model.bindings.get(id);
  if (!entry) return;
  setText(el("inspector-label"), entry.meta.label || id);
  setText(el("inspector-authored"), entry.meta.authored_formula || "—");
  setText(el("inspector-resolved"), entry.meta.resolved_formula || "—");
  setText(el("inspector-backend"), entry.meta.backend || "—");
  setText(el("inspector-window"),
          entry.meta.window_steps === null ? "—"
            : String(entry.meta.window_steps));
  setText(el("inspector-current"), formatRho(entry.rho));
  setText(el("inspector-worst"), formatAt(entry.worst));
  setText(el("inspector-final"), formatAt(entry.final));
  setText(el("inspector-violation"), formatAt(entry.first_violation));
  const { ctx, width, height } = sizeCanvas(el("inspector-sparkline"));
  ctx.clearRect(0, 0, width, height);
  // Solo fetch: the sparkline needs one binding, not the whole spec.
  const points = seriesFor(id, Math.round(width), [id]) || [];
  if (!points.length) return;
  let min = 0;
  let max = 0;
  for (const [, lo, hi] of points) {
    if (lo < min) min = lo;
    if (hi > max) max = hi;
  }
  if (min === max) { min -= 1; max += 1; }
  const minStep = model.minStep;
  const domain = Math.max(1, model.maxStep - minStep);
  const sy = (v) => height - ((v - min) / (max - min)) * height;
  ctx.strokeStyle = INK.muted;
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(0, sy(0));
  ctx.lineTo(width, sy(0));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = bindingColor(id);
  ctx.beginPath();
  points.forEach(([step, lo, hi], index) => {
    const x = ((step - minStep) / domain) * width;
    const y = sy((lo + hi) / 2);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // min/max values with the binding's unit
  const unit = model.bindings.get(id).meta.unit;
  const suffix = unit ? " " + unit : "";
  ctx.font = "9px ui-monospace, Menlo, monospace";
  ctx.fillStyle = INK.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(formatAxisValue(max) + suffix, width - 2, 1);
  ctx.textBaseline = "bottom";
  ctx.fillText(formatAxisValue(min) + suffix, width - 2, height - 1);
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
}

/* ---- transitions ---------------------------------------------------------------- */

const TRANSITIONS_SHOWN = 200;

function transitionVisible(t) {
  if (model.selection?.type === "binding") {
    return bindingId(t.spec, t.key) === model.selection.id;
  }
  const spec = scopeSpec();
  return !spec || t.spec === spec;
}

function renderTransitions() {
  model.dirty.transitions = false;
  const body = el("transitions").tBodies[0];
  const rows = [];
  for (let i = model.transitions.length - 1;
       i >= 0 && rows.length < TRANSITIONS_SHOWN; i -= 1) {
    const t = model.transitions[i];
    if (transitionVisible(t)) rows.push(t);
  }
  const keyOf = transitionKey;
  const existing = new Map(
    Array.from(body.children).map((node) => [node.dataset.tkey, node]));
  const seen = new Set();
  let cursor = body.firstChild;
  for (const t of rows) {
    const mapKey = keyOf(t);
    seen.add(mapKey);
    let node = existing.get(mapKey);
    if (!node) {
      node = document.createElement("tr");
      node.dataset.tkey = mapKey;
      node.innerHTML = "<td></td><td></td><td></td><td></td>";
      const step = t.step;
      node.addEventListener("click", () => seekTo(step));
      const cells = node.children;
      setText(cells[0], String(t.step));
      setText(cells[1],
              t.time === null || t.time === undefined
                ? "—" : t.time.toFixed(2));
      const entry = model.bindings.get(bindingId(t.spec, t.key));
      setText(cells[2], entry?.meta.label || t.key);
      cells[3].innerHTML =
        `<span class="state-${t.from || "pending"}">${t.from ?? "·"}</span>` +
        ` → <span class="state-${t.to}">${t.to}</span>`;
    }
    if (node !== cursor) body.insertBefore(node, cursor);
    else cursor = cursor.nextSibling;
  }
  for (const [mapKey, node] of existing) {
    if (!seen.has(mapKey)) node.remove();
  }
  // ponytail: local tail + one paging call; virtualized log if runs
  // outgrow this.
  el("transitions-earlier").hidden = model.transitionsOffset === 0;
}

function loadEarlierTransitions() {
  const limit = 200;
  const offset = model.transitionsOffset === null
    ? 0                                  // probe: learn the total first
    : Math.max(0, model.transitionsOffset - limit);
  socket.emit("transitions_page", { offset, limit }, (ack) => {
    if (!ack?.ok) return;
    if (model.transitionsOffset === null) {
      // First request: the server total tells us where our live tail
      // sits; fetch the page directly before it.
      model.transitionsOffset = Math.max(
        0, ack.total - model.transitions.length);
      if (model.transitionsOffset > 0) loadEarlierTransitions();
      else markDirty("transitions");
      return;
    }
    addTransitions(ack.entries);
    model.transitionsOffset = ack.offset;
    markDirty("transitions", "timeline");
  });
}

el("transitions-earlier").addEventListener("click", loadEarlierTransitions);

/* ---- camera grid ------------------------------------------------------------------- */

function renderCameraGrid() {
  const host = el("camera-grid");
  const existing = new Map(
    Array.from(host.children).map((node) => [node.dataset.camera, node]));
  for (const name of model.cameras) {
    let tile = existing.get(name);
    if (!tile) {
      tile = document.createElement("div");
      tile.className = "tile";
      tile.dataset.camera = name;
      tile.innerHTML = `<canvas></canvas><span class="name"></span>`;
      setText(tile.querySelector(".name"), name);
      tile.addEventListener("click", () => promoteCamera(name));
      host.appendChild(tile);
    }
    const slot = model.tileBitmaps.get(name);
    if (slot?.bitmap) {
      const canvas = tile.querySelector("canvas");
      canvas.width = imageWidth(slot.bitmap);
      canvas.height = imageHeight(slot.bitmap);
      canvas.getContext("2d").drawImage(slot.bitmap, 0, 0);
    }
  }
  for (const [name, node] of existing) {
    if (!model.cameras.includes(name)) node.remove();
  }
}

function sendCameraChange(cameras) {
  // While not following there is no bundle stream: re-request the
  // current step so the new camera selection paints immediately.
  if (model.follow) {
    sendView({ cameras });
  } else {
    sendView({ seek: model.step, cameras });
  }
}

function promoteCamera(name) {
  model.camera = name;
  model.gridOpen = false;
  for (const slot of model.tileBitmaps.values()) slot.bitmap?.close?.();
  model.tileBitmaps.clear();
  sendCameraChange([name]);
  markDirty("camera");
}

el("camera-grid-toggle").addEventListener("click", () => {
  if (model.gridOpen) {
    promoteCamera(model.camera || model.cameras[0]);
    return;
  }
  model.gridOpen = true;
  sendCameraChange(model.cameras.slice());
  markDirty("camera");
});

/* ---- 3D scene panel -------------------------------------------------------
 * scene3d.js is a module script, so it evaluates after this classic one:
 * calls made before it registers window.SpecolaScene3D are queued and
 * flushed on its ready event. Adapters without a 3D scene never send a
 * geometry event, and the panel then stays out of the layout entirely.
 */

const scene3dQueue = [];

function scene3d(method, ...args) {
  if (window.SpecolaScene3D) window.SpecolaScene3D[method](...args);
  else {
    // If the module never evaluates (import-map failure), pose bundles
    // would grow this queue for the whole run; keep only the newest.
    if (scene3dQueue.length > 32) scene3dQueue.shift();
    scene3dQueue.push([method, args]);
  }
}

window.addEventListener("specola-scene3d-ready", () => {
  for (const [method, args] of scene3dQueue.splice(0)) {
    window.SpecolaScene3D[method](...args);
  }
});

function ingestGeometry(payload) {
  if (!payload?.json) return;
  document.body.classList.add("has-scene");
  el("scene-toggle").hidden = false;
  scene3d("init", el("scene-canvas"));
  scene3d("setGeometry", payload.json, payload.meshes);
  buildGroupToggles(payload.json);
}

function buildGroupToggles(geometry) {
  // One checkbox per geom group actually present in this scene. The
  // default-visible set comes from the export — the simulator's own
  // cameras decide (robosuite hides group 0, its collision hulls) —
  // with the plain-MuJoCo 0-2 as the fallback for older recordings.
  // scene3d.js applies the same list.
  const defaults = new Set(geometry.visible_groups ?? [0, 1, 2]);
  const host = el("scene-groups");
  host.textContent = "";
  const groups = Array.from(
    new Set((geometry.geoms || []).map((geom) => geom.group)))
    .sort((a, b) => a - b);
  for (const group of groups) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = defaults.has(group);
    box.addEventListener(
      "change", () => scene3d("setGroupVisible", group, box.checked));
    label.append(box, String(group));
    host.appendChild(label);
  }
  // Hide the row only when there is nothing to choose AND the lone
  // group is visible by default; a single hidden-by-default group
  // needs its checkbox or the panel would be empty with no recovery.
  host.hidden = groups.length < 2
    && groups.every((group) => defaults.has(group));
}

el("scene-toggle").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("scene-collapsed");
  el("scene-toggle").classList.toggle("on", !collapsed);
});

/* ---- command palette ---------------------------------------------------
 * Spans both views, because the thing a person wants to reach is often
 * not in the one they are looking at: a spec name while in Discover, a
 * class while watching the monitor. Ctrl/Cmd-K opens it.
 *
 * Sources are read live at open time rather than kept in a parallel
 * index, so nothing here can go stale against the model or the tree.
 */
(() => {
  const overlay = () => el("palette-overlay");
  const input = () => el("command-input");
  const list = () => el("command-results");
  let entries = [];
  let cursor = 0;

  function collect() {
    const found = [];
    found.push(
      { kind: "view", label: "Monitor", sub: "watch this run",
        run: () => el("tab-monitor").click() },
      { kind: "view", label: "Discover", sub: "browse the scene",
        run: () => { const t = el("tab-discover");
          if (!t.disabled) t.click(); } },
    );
    // Specs and bindings, from the tree the monitor is already rendering.
    for (const row of document.querySelectorAll("#tree li")) {
      const label = row.querySelector(".label")?.textContent?.trim();
      if (!label) continue;
      found.push({
        // treeRowFor stamps dataset.kind; there is no class to read.
        kind: row.dataset.kind || "binding",
        label,
        sub: row.querySelector(".pill.state")?.textContent?.trim() || "",
        run: () => { el("tab-monitor").click(); row.click();
          row.scrollIntoView({ block: "center" }); },
      });
    }
    // Classes, from Discover's own tree when it has been opened.
    for (const row of document.querySelectorAll("#class-tree li.class-row")) {
      const label = row.querySelector(".label")?.textContent?.trim();
      if (!label) continue;
      found.push({
        kind: "class", label,
        sub: row.querySelector(".count")?.textContent?.trim() || "",
        run: () => { const t = el("tab-discover");
          if (!t.disabled) t.click(); row.click();
          row.scrollIntoView({ block: "center" }); },
      });
    }
    return found;
  }

  function score(entry, needle) {
    const label = entry.label.toLowerCase();
    if (!needle) return 0;
    const at = label.indexOf(needle);
    if (at === -1) return null;
    // Prefix beats interior, shorter beats longer: `object` should not
    // rank under `distr_cab_main_object` for the query "object".
    return at * 10 + label.length;
  }

  function render() {
    const needle = input().value.trim().toLowerCase();
    const ranked = entries
      .map((entry) => ({ entry, rank: score(entry, needle) }))
      .filter((row) => row.rank !== null)
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 60);
    cursor = Math.min(cursor, Math.max(0, ranked.length - 1));
    const host = list();
    host.textContent = "";
    if (!ranked.length) {
      const empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = needle ? "nothing matches" : "type to search";
      host.appendChild(empty);
      // Cleared here, not merely left behind: `visible` was assigned at
      // the end of this function, so an empty result kept the previous
      // query's list and Enter navigated to whatever had been first.
      visible = [];
      return;
    }
    ranked.forEach((row, index) => {
      const item = document.createElement("li");
      item.classList.toggle("on", index === cursor);
      item.innerHTML =
        `<span class="kind"></span><span class="what"></span>` +
        `<span class="sub"></span>`;
      item.querySelector(".kind").textContent = row.entry.kind;
      item.querySelector(".what").textContent = row.entry.label;
      item.querySelector(".sub").textContent = row.entry.sub;
      item.addEventListener("click", () => choose(row.entry));
      host.appendChild(item);
    });
    host.children[cursor]?.scrollIntoView({ block: "nearest" });
    list().dataset.count = String(ranked.length);
    visible = ranked.map((row) => row.entry);
  }

  let visible = [];

  function choose(entry) { close(); entry.run(); }

  function open() {
    entries = collect();
    overlay().hidden = false;
    input().value = "";
    cursor = 0;
    render();
    input().focus();
  }

  function close() { overlay().hidden = true; }

  input().addEventListener("input", () => { cursor = 0; render(); });

  document.addEventListener("keydown", (event) => {
    const open_ = !overlay().hidden;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      open_ ? close() : open();
      return;
    }
    if (!open_) return;
    if (event.key === "Escape") { close(); event.preventDefault(); }
    else if (event.key === "ArrowDown") {
      cursor = Math.min(cursor + 1, visible.length - 1); render();
      event.preventDefault();
    } else if (event.key === "ArrowUp") {
      cursor = Math.max(cursor - 1, 0); render(); event.preventDefault();
    } else if (event.key === "Enter") {
      if (visible[cursor]) choose(visible[cursor]);
      event.preventDefault();
    }
  }, true);

  overlay().addEventListener("click", (event) => {
    if (event.target === overlay()) close();
  });
})();
