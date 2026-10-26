/* specola Discover: browse the episode's scene, and the ways to select it.
 *
 * Kept out of dashboard.js on purpose. That file drives the monitor —
 * decode tokens, rAF-batched dirty rendering, paint-gated acks — and none
 * of that is wanted here, while a mistake in here must not be able to
 * stop a live run from rendering.
 *
 * Two things shape this file. The scene can be enormous: the raw
 * document of results/20260816_004207_robocasa_notask holds 2,053
 * entities and 71,912 properties, and RoboCasa randomizes per reset so
 * other episodes are larger. DOM nodes are created for the expanded
 * class and the selected scope only, never for the whole document. And selectors are never computed here — the server
 * resolves a selection against the immutable Python model and returns the
 * same forms the terminal shows, because two implementations of the
 * selector language would eventually disagree and the disagreement would
 * be a specification that monitors the wrong thing.
 */

"use strict";

(() => {
  const byId = (id) => document.getElementById(id);

  const state = {
    available: false,
    variant: "semantic",
    browseMode: "class",       // "class" | "role", the terminal's C key
    documents: {},          // variant -> parsed document
    entitiesByClass: new Map(),
    selection: null,        // {entityClass, entityId}
    signals: [],
    signal: null,
    signalIndex: null,
    forms: [],
    form: null,
    expanded: new Set(),
    search: "",
    requestId: 0,
    pending: null,
    signalPending: null,
    searchTimer: null,
    // Bumped whenever the variant changes. Every in-flight fetch, signal
    // request and palette request carries the generation it was issued
    // under and is dropped if it comes back into a different one: a slow
    // Raw fetch finishing after a switch to Semantic would otherwise
    // render Raw rows under a Semantic toggle, and the class a person
    // then clicked would be resolved against the other document.
    generation: 0,
  };

  /* ---- view switching -------------------------------------------------- */

  function showView(view) {
    const discovering = view === "discover";
    byId("discover").hidden = !discovering;
    byId("layout").hidden = discovering;
    byId("timeline").hidden = discovering;
    byId("status-strip").hidden = discovering;
    for (const tab of document.querySelectorAll("#view-tabs .tab")) {
      tab.classList.toggle("on", tab.dataset.view === view);
    }
    if (discovering && !state.documents[state.variant]) loadDocument();
  }

  for (const tab of document.querySelectorAll("#view-tabs .tab")) {
    tab.addEventListener("click", () => {
      if (!tab.disabled) showView(tab.dataset.view);
    });
  }

  /* ---- the document ----------------------------------------------------- */

  async function loadDocument() {
    const variant = state.variant;
    const generation = state.generation;
    if (state.documents[variant]) return state.documents[variant];
    byId("discover-summary").textContent = "loading the scene…";
    try {
      // Content-Encoding: gzip, so the browser decompresses this in C
      // rather than the page decoding a megabyte-plus of JSON by hand.
      const response = await fetch(
        `/api/discovery?variant=${encodeURIComponent(variant)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const document_ = await response.json();
      // Cached under its own variant either way -- the bytes are good,
      // they just are not what is on screen any more.
      state.documents[variant] = document_;
      if (generation !== state.generation) return document_;
      indexDocument(document_);
      renderSummary(document_);
      renderClasses();
      return document_;
    } catch (error) {
      if (generation !== state.generation) return null;
      byId("discover-summary").textContent =
        `scene unavailable: ${error.message}`;
      return null;
    }
  }

  function indexDocument(document_) {
    state.entitiesByClass = new Map();
    for (const entity of document_.entities) {
      let bucket = state.entitiesByClass.get(entity.class);
      if (!bucket) {
        bucket = [];
        state.entitiesByClass.set(entity.class, bucket);
      }
      bucket.push(entity);
    }
  }

  function renderSummary(document_) {
    const counts = document_.classes
      .filter((record) => state.variant === "raw" || !record.raw)
      .map((record) => `${record.live_count} ${record.class}`)
      .join(" · ");
    const properties = document_.counts.after.properties;
    byId("discover-summary").textContent =
      `${counts} · ${properties} signals · dt ${document_.timing.effective_dt} s`;
  }

  /* ---- browse ----------------------------------------------------------- */

  function matchesSearch(text) {
    return !state.search || text.toLowerCase().includes(state.search);
  }

  /* The terminal's tree, node for node.
   *
   * Class mode nests class -> role -> entity, with roleless entities
   * hanging directly off their class; role mode nests role -> class ->
   * entity and carries a "(no declared role)" branch. A declared class
   * with no live entity is listed too, as "0 live · declared" -- a
   * runtime-populated class such as `contact` is empty at reset and
   * still selectable, and omitting it would hide the fact that the
   * simulator declares it at all.
   */

  const ROLE_ABSENT = "\u0000roleless";

  function declaredEmptyClasses() {
    const document_ = state.documents[state.variant];
    if (!document_) return [];
    return (document_.declared_classes || [])
      .filter((record) => !state.entitiesByClass.has(record.class))
      .map((record) => record.class);
  }

  function scopeKey(scope) {
    return `${scope.entityClass || ""}|${scope.role || ""}|` +
      `${scope.entityId === null || scope.entityId === undefined ? "" : scope.entityId}`;
  }

  function isSelected(scope) {
    return Boolean(state.selection) &&
      scopeKey(state.selection) === scopeKey(scope);
  }

  function branchRow(host, scope, label, count, depth) {
    const open = state.expanded.has(scopeKey(scope));
    const row = document.createElement("li");
    row.className = "class-row";
    row.style.paddingLeft = `${6 + depth * 12}px`;
    row.classList.toggle("selected", isSelected(scope));
    row.innerHTML =
      `<span class="twist"></span><span class="label"></span>` +
      `<span class="count"></span>`;
    row.querySelector(".twist").textContent = open ? "▾" : "▸";
    row.querySelector(".label").textContent = label;
    row.querySelector(".count").textContent = count;
    row.tabIndex = -1;
    row.addEventListener("click", (event) => {
      event.stopPropagation();
      const key = scopeKey(scope);
      if (open) state.expanded.delete(key); else state.expanded.add(key);
      state.focusList = "class-tree";
      select(scope);
      renderClasses();
    });
    host.appendChild(row);
    return open;
  }

  function leafRow(host, entityClass, entity, depth) {
    const scope = { entityClass, role: null, entityId: entity.id };
    const row = document.createElement("li");
    row.className = "entity-row";
    row.style.paddingLeft = `${6 + depth * 12}px`;
    row.classList.toggle("selected", isSelected(scope));
    row.innerHTML = `<span class="label"></span><span class="sub"></span>`;
    row.querySelector(".label").textContent =
      `${entity.class}_${entity.id}`;
    row.querySelector(".sub").textContent =
      entity.label || entity.declared_name || "";
    row.tabIndex = -1;
    row.addEventListener("click", (event) => {
      event.stopPropagation();
      state.focusList = "class-tree";
      select(scope);
      renderClasses();
    });
    host.appendChild(row);
  }

  function renderClasses() {
    const host = byId("class-tree");
    host.textContent = "";
    if (!state.documents[state.variant]) return;
    if (state.browseMode === "class") renderClassMode(host);
    else renderRoleMode(host);
  }

  function renderClassMode(host) {
    const names = [
      ...state.entitiesByClass.keys(), ...declaredEmptyClasses(),
    ].sort();
    for (const entityClass of names) {
      const entities = state.entitiesByClass.get(entityClass) || [];
      const matching = entities.filter(
        (entity) => matchesSearch(haystack(entity)));
      if (!matching.length && !matchesSearch(entityClass)) continue;

      const scope = { entityClass, role: null, entityId: null };
      const count = entities.length
        ? String(entities.length) : "0 live · declared";
      if (!branchRow(host, scope, entityClass, count, 0)) continue;

      const shown = matching.length ? matching : entities;
      const roles = [...new Set(
        shown.map((entity) => entity.role).filter(Boolean))].sort();
      const placed = new Set();
      for (const role of roles) {
        const members = shown.filter((entity) => entity.role === role);
        const roleScope = { entityClass, role, entityId: null };
        const open = branchRow(
          host, roleScope, role, String(members.length), 1);
        for (const entity of members) {
          placed.add(entity.id);
          if (open) leafRow(host, entityClass, entity, 2);
        }
      }
      for (const entity of shown) {
        if (!placed.has(entity.id)) leafRow(host, entityClass, entity, 1);
      }
    }
  }

  function renderRoleMode(host) {
    const document_ = state.documents[state.variant];
    const entities = document_.entities.filter(
      (entity) => state.variant === "raw" || !entity.raw);
    const roles = [...new Set(
      entities.map((entity) => entity.role).filter(Boolean))].sort();
    if (entities.some((entity) => !entity.role)) roles.push(ROLE_ABSENT);

    for (const role of roles) {
      const members = entities.filter(
        (entity) => (role === ROLE_ABSENT ? !entity.role
          : entity.role === role));
      const matching = members.filter(
        (entity) => matchesSearch(haystack(entity)));
      const label = role === ROLE_ABSENT ? "(no declared role)" : role;
      if (!matching.length && !matchesSearch(label)) continue;

      // A role branch spans classes, so it carries no class of its own;
      // the server reads `role` alone, exactly as DiscoveryScope does.
      const roleScope = { entityClass: null, role, entityId: null };
      if (!branchRow(host, roleScope, label, String(members.length), 0)) {
        continue;
      }
      const shown = matching.length ? matching : members;
      for (const entityClass of [...new Set(
        shown.map((entity) => entity.class))].sort()) {
        const classMembers = shown.filter(
          (entity) => entity.class === entityClass);
        const scope = { entityClass, role, entityId: null };
        const open = branchRow(
          host, scope, entityClass, String(classMembers.length), 1);
        if (!open) continue;
        for (const entity of classMembers) {
          leafRow(host, entityClass, entity, 2);
        }
      }
    }
  }

  /* The same haystack `entity_search_blob` builds, over the same fields,
   * built once per entity and cached on it.
   *
   * This is a second implementation of a Python rule, which the palette
   * deliberately refuses to have — but the two cases are not alike. A
   * condensation that drifts names a *different signal* and yields a
   * specification that monitors the wrong thing. A search haystack that
   * drifts shows fewer rows: the failure is a missing hit, never a wrong
   * selector. Serializing the blob instead was measured at +0.184 MiB
   * gzipped on RoboCasa's semantic document (0.44 → 0.62 MiB), which is
   * a 42% transfer increase to avoid a miss.
   * `test_the_search_haystack_matches_pythons_field_for_field` reads
   * this function and `entity_search_blob` and compares the two field
   * sets directly, so the list below cannot silently go stale. */
  function haystack(entity) {
    if (entity._blob !== undefined) return entity._blob;
    const parts = [
      entity.class, String(entity.id), entity.label, entity.role,
      entity.role_basis, entity.role_provenance, entity.declared_name,
      entity.declared_category, entity.declaration_source,
      ...(entity.fixture_names || []), ...(entity.fixture_refs || []),
      entity.fixture_type, entity.fixture_provenance,
      ...(entity.region_fixture_names || []),
      ...(entity.region_fixture_refs || []), entity.region_key,
    ];
    for (const property of entity.properties || []) {
      parts.push(property.key, property.family, property.group,
                 property.unit, property.component, property.frame,
                 property.note, ...(property.tags || []));
    }
    entity._blob = parts.filter(Boolean).join(" ").toLowerCase();
    return entity._blob;
  }


  /* ---- signals ----------------------------------------------------------- */

  function select(scope) {
    state.selection = {
      entityClass: scope.entityClass ?? null,
      role: scope.role ?? null,
      entityId: scope.entityId ?? null,
    };
    state.signal = null;
    state.signalIndex = null;
    state.signals = [];
    showSignal(null);
    requestSignals();
    requestPalette();
  }

  const SIGNAL_PAGE = 300;

  /* Signals are aggregated by Python and only by Python.
   *
   * This file used to group and condense them itself. The two rules
   * drifted: the JavaScript kept a shared prefix, while
   * ``_glob_for_keys`` keeps a shared prefix *and* a shared suffix, so a
   * family Python calls ``xpos_body_*_0`` was named ``xpos_body_*``
   * here. The pattern sent back then matched nothing, and the server
   * answered with the palette for the whole scope. Asking the server for
   * the rows removes the second implementation rather than repairing it.
   */

  function requestSignals() {
    const socket = window.specolaSocket;
    if (!state.selection || !socket) return;
    const requestId = ++state.requestId;
    const generation = state.generation;
    state.signalPending = requestId;
    byId("signal-list").textContent = "";
    byId("signal-scope").textContent = scopeLabel();
    socket.emit("signal_rows", {
      request_id: requestId,
      entity_class: state.selection.entityClass,
      entity_id: state.selection.entityId,
      role: state.selection.role,
      include_raw: state.variant === "raw",
    }, (reply) => {
      if (!reply || reply.request_id !== state.signalPending) return;
      if (generation !== state.generation) return;
      state.signals = reply.signals || [];
      renderSignals(reply.reason);
    });
  }

  function scopeLabel() {
    const { entityClass, role, entityId } = state.selection;
    if (entityId !== null) return `${entityClass}_${entityId}`;
    const base = entityClass || "(all classes)";
    return role ? `${base} · ${role === ROLE_ABSENT ? "no role" : role}`
      : base;
  }

  function signalHaystack(signal) {
    return [signal.pattern, signal.quantity, signal.component,
            signal.unit, signal.frame, signal.example_key]
      .filter(Boolean).join(" ");
  }

  function renderSignals(reason) {
    const host = byId("signal-list");
    host.textContent = "";
    if (!state.selection) return;
    byId("signal-scope").textContent = scopeLabel();
    const signals = state.signals.filter(
      (signal) => matchesSearch(signalHaystack(signal)));
    if (!signals.length) {
      const empty = document.createElement("li");
      empty.className = "signal-more";
      empty.textContent = reason || "no signal matches this search";
      host.appendChild(empty);
      return;
    }
    // A bounded first page: one RoboCasa fixture scope can aggregate
    // tens of thousands of signals, and a list that long is neither
    // renderable nor readable.
    const shown = signals.slice(0, SIGNAL_PAGE);
    for (const signal of shown) {
      const row = document.createElement("li");
      row.className = "signal-row";
      row.classList.toggle("selected", state.signal === signal.pattern);
      row.innerHTML =
        `<span class="label"></span><span class="unit"></span>`;
      row.querySelector(".label").textContent = signal.pattern;
      row.querySelector(".unit").textContent =
        [signal.unit, signal.frame].filter(Boolean).join(" / ") || "—";
      row.tabIndex = -1;
      row.addEventListener("click", () => {
        state.signal = signal.pattern;
        // Position in the server's own list, because two families in one
        // scope can render the same text -- see ``_pick_signal``.
        state.signalIndex = state.signals.indexOf(signal);
        state.focusList = "signal-list";
        showSignal(signal);
        // Re-rendering replaces every row, so the node clicked is
        // detached by the time focus would land on it; the freshly
        // rendered selected row is the one to focus.
        renderSignals();
        byId("signal-list").querySelector(".selected")?.focus();
        requestPalette();
      });
      host.appendChild(row);
    }
    if (signals.length > shown.length) {
      const more = document.createElement("li");
      more.className = "signal-more";
      more.textContent =
        `${signals.length - shown.length} more — narrow with search`;
      host.appendChild(more);
    }
  }

  function fillDetails(host, pairs) {
    /* Both detail panels are definition lists of label/value pairs, and
     * both were building their own dt/dd. One builder, so a change to
     * how a detail row looks is one change. */
    host.textContent = "";
    for (const [label, value] of pairs) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = String(value);
      host.append(term, description);
    }
  }

  /* The terminal's "Selected signal" panel: unit, frame, and what the
   * signal currently reads. The row stands for a family of keys, so the
   * value is one entity's -- an example of what the family reads, not a
   * summary of all of them -- and the key it came from is named beside
   * it so it is clear which. */
  function showSignal(signal) {
    const host = byId("signal-detail");
    host.textContent = "";
    if (!signal) { host.hidden = true; return; }
    host.hidden = false;
    fillDetails(host, [
      ["Unit", signal.unit || "—"],
      ["Frame", signal.frame || "—"],
      ["Value", formatValue(signal.example_value)],
      ["Read from", signal.example_key || "—"],
    ]);
  }

  function formatValue(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number") return String(value);
    if (!Number.isFinite(value)) return String(value);
    // Enough digits to tell two entities apart, few enough to keep the
    // column narrow.
    return Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.001)
      ? value.toExponential(2)
      : value.toFixed(3).replace(/\.?0+$/, "");
  }

  /* ---- palette ------------------------------------------------------------ */

  function requestPalette() {
    if (!state.selection || !window.specolaSocket) return;
    const requestId = ++state.requestId;
    const generation = state.generation;
    state.pending = requestId;
    byId("palette-list").textContent = "";
    window.specolaSocket.emit("selector_palette", {
      request_id: requestId,
      entity_class: state.selection.entityClass,
      entity_id: state.selection.entityId,
      role: state.selection.role,
      signal: state.signal,
      signal_index: state.signalIndex,
      include_raw: state.variant === "raw",
    }, (reply) => {
      // A reply that arrived after the selection moved on describes a
      // scope nobody is looking at.
      if (!reply || reply.request_id !== state.pending) return;
      if (generation !== state.generation) return;
      state.forms = reply.forms || [];
      renderPalette(reply.reason);
    });
  }

  function renderPalette(reason) {
    const host = byId("palette-list");
    host.textContent = "";
    if (!state.forms.length) {
      const empty = document.createElement("li");
      empty.className = "palette-empty";
      empty.textContent = reason || "no selector form for this selection";
      host.appendChild(empty);
      showForm(null);
      return;
    }
    for (const form of state.forms) {
      const row = document.createElement("li");
      row.className = "palette-row";
      row.classList.toggle("unavailable", !form.available);
      // A row nobody checked must not look like one that passed.
      row.classList.toggle("unchecked", form.available && !form.verified);
      // Forms only. Coverage is reported by the signals pane above; a
      // count on this row is the thing this panel deliberately dropped.
      row.textContent = form.partner_reference
        ? `${form.reference}  ↔  ${form.partner_reference}`
        : form.reference;
      row.tabIndex = -1;
      row.addEventListener("click", () => {
        state.focusList = "palette-list";
        row.focus();
        showForm(form);
        for (const other of host.children) {
          other.classList.remove("selected");
        }
        row.classList.add("selected");
      });
      host.appendChild(row);
    }
    // The same choice `default_form` makes, on the flag the server now
    // sends: a role selection has to land on the row that reproduces it,
    // or the page silently offers a class-wide reference that monitors
    // entities the selection excluded. Reimplementing it as "the first
    // available one" is how three surfaces came to answer one question
    // three ways.
    const first = state.forms.find((form) => form.available &&
                                    form.reproduces_role) ||
      state.forms.find((form) => form.available) ||
      state.forms[0];
    showForm(first);
    const index = state.forms.indexOf(first);
    if (index >= 0) host.children[index].classList.add("selected");
  }

  function showForm(form) {
    state.form = form;
    const detail = byId("form-detail");
    const yaml = byId("starter-yaml");
    const copy = byId("copy-starter");
    if (!form) {
      detail.textContent = "";
      yaml.textContent = "";
      copy.hidden = true;
      byId("insert-selector").hidden = true;
      byId("copy-selector").hidden = true;
      return;
    }
    const lines = [
      ["Reference", form.reference],
      ["Shape", `${form.binding_kind} binding · ${form.component_kind} component`],
    ];
    if (form.partner_reference) {
      lines.push(["Second slot", form.partner_reference]);
    }
    if (form.available && form.note) lines.push(["Means", form.note]);
    if (form.expansions != null) {
      lines.push(["Expands to", form.expansions === 1
        ? "1 formula" : `${form.expansions.toLocaleString()} formulas`]);
    }
    if (form.available && !form.verified) {
      lines.push(["Checked", "not against this scene"]);
    }
    if (!form.available) {
      lines.push(["Unavailable", form.unavailable_reason]);
    }
    fillDetails(detail, lines);
    // The YAML is the server's, produced by the shared starter
    // authority. Assembling it here is how a difference selector's
    // backslash gets eaten.
    yaml.textContent = form.starter ? toYaml(form.starter) : "";
    copy.hidden = !form.starter;
    byId("insert-selector").hidden = !form.available;
    byId("copy-selector").hidden = !form.available;
  }

  function toYaml(starter) {
    /* Renders the server's already-built mapping. It never invents a
     * key and never quotes by hand: a value containing a backslash or a
     * colon is emitted single-quoted, which is what the Python writer
     * produces for the same entry. */
    const quote = (value) =>
      /[\\:#{}[\],&*?|>%@`"']|^\s|\s$/.test(value)
        ? `'${value.replace(/'/g, "''")}'`
        : value;
    const lines = ["specs:"];
    lines.push(`- name: ${quote(starter.name)}`);
    lines.push(`  formula: ${quote(starter.formula)}`);
    if (starter.constants) {
      lines.push("  constants:");
      for (const [key, value] of Object.entries(starter.constants)) {
        lines.push(`    ${key}: ${value}`);
      }
    }
    if (starter.backend) lines.push(`  backend: ${starter.backend}`);
    if (starter.symmetric) lines.push("  symmetric: true");
    return lines.join("\n");
  }

  byId("copy-starter").addEventListener("click", () => {
    const text = byId("starter-yaml").textContent;
    if (!text) return;
    navigator.clipboard?.writeText(text);
    const button = byId("copy-starter");
    button.textContent = "copied";
    setTimeout(() => { button.textContent = "Copy"; }, 1200);
  });

  /* ---- controls ------------------------------------------------------------ */

  byId("discover-search").addEventListener("input", (event) => {
    // Debounced: every keystroke otherwise re-filters and re-renders a
    // list that can hold tens of thousands of rows.
    const value = event.target.value.trim().toLowerCase();
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => {
      state.search = value;
      renderClasses();
      renderSignals();
    }, 200);
  });

  for (const [id, variant] of [
    ["variant-semantic", "semantic"], ["variant-raw", "raw"],
  ]) {
    byId(id).addEventListener("click", async () => {
      if (state.variant === variant) return;
      state.variant = variant;
      // Invalidates every reply still in flight, including the ones this
      // handler is about to clear the panes for.
      state.generation += 1;
      state.pending = null;
      state.signalPending = null;
      state.signals = [];
      byId("variant-semantic").classList.toggle(
        "on", variant === "semantic");
      byId("variant-raw").classList.toggle("on", variant === "raw");
      state.expanded.clear();
      state.selection = null;
      state.signal = null;
      byId("signal-list").textContent = "";
      byId("palette-list").textContent = "";
        showForm(null);
      const document_ = state.documents[variant] || await loadDocument();
      if (document_) {
        indexDocument(document_);
        renderSummary(document_);
        renderClasses();
      }
    });
  }

  /* ---- authoring ----------------------------------------------------------- */

  /* The terminal writer's workflow, unchanged: insert a selector, name
   * the draft, validate it against this episode's own scene, keep it,
   * accumulate, save. The three properties that make that workflow
   * trustworthy are kept here rather than reinvented:
   *
   * A validation belongs to the revision it validated. Editing the draft
   * bumps the revision and disables Keep, so a reply that arrives after
   * a keystroke cannot authorize keeping something nobody checked.
   *
   * The report is the checker's own text, not a summary. It says which
   * keys bound, in what units, under which groups and tags -- which is
   * the part an author reads to learn *why* a formula binds what it
   * binds.
   *
   * Nothing is compiled here. `spec_check` runs the same `_check_entry`
   * the terminal runs, against the same scene, so the two surfaces
   * cannot accept different drafts.
   */

  const draft = {
    specs: [],
    savedPath: null,
    keptDirty: false,
    revision: 0,
    validatedRevision: null,
    validationOk: false,
    validatedEntry: null,
    lastReport: "Not validated.",
    lastVerdict: null,       // "accepted" | "rejected" | null
    programmatic: false,
    checkSeq: 0,
  };

  const nameField = () => byId("spec-name");
  const formulaField = () => byId("spec-formula");
  const constantsField = () => byId("spec-constants");

  function hasDraft() {
    const constants = constantsField().value.trim();
    return Boolean(
      nameField().value.trim() ||
      formulaField().value.trim() ||
      (constants !== "" && constants !== "{}"));
  }

  function setReport(text, verdict) {
    draft.lastReport = text;
    draft.lastVerdict = verdict || null;
    const host = byId("validation");
    host.textContent = text;
    host.classList.toggle("accepted", verdict === "accepted");
    host.classList.toggle("rejected", verdict === "rejected");
    byId("draft-state").classList.remove("stale");
    byId("draft-state").textContent =
      verdict === "accepted" ? "accepted"
        : verdict === "rejected" ? "rejected" : "not validated";
  }

  function draftEdited() {
    if (draft.programmatic) return;
    // The STALE banner needs one paint, on the first edit after a
    // validation; later keystrokes would rewrite identical content.
    const alreadyStale = draft.validatedRevision !== null &&
      draft.revision !== draft.validatedRevision;
    draft.revision += 1;
    draft.validationOk = false;
    byId("keep").disabled = true;
    if (draft.validatedRevision !== null && !alreadyStale) {
      byId("validation").textContent =
        `Validation — STALE\nDraft edited since the last validation.\n\n${draft.lastReport}`;
      const label = byId("draft-state");
      label.textContent = "stale";
      label.classList.add("stale");
    }
  }

  for (const field of ["spec-name", "spec-formula", "spec-constants"]) {
    byId(field).addEventListener("input", draftEdited);
  }

  function usableForm() {
    const form = state.form;
    if (!form) return null;
    if (!form.available) {
      setReport(
        `This form is unavailable here.\n${form.unavailable_reason || ""}`,
        "rejected");
      return null;
    }
    return form;
  }

  function insertAtCursor(text) {
    const area = formulaField();
    const start = area.selectionStart ?? area.value.length;
    const end = area.selectionEnd ?? area.value.length;
    area.value = area.value.slice(0, start) + text + area.value.slice(end);
    const caret = start + text.length;
    area.setSelectionRange(caret, caret);
    area.focus();
    draftEdited();
  }

  byId("insert-selector").addEventListener("click", () => {
    const form = usableForm();
    if (!form) return;
    if (form.partner_reference) {
      // Both slots, not one and a note. A lone `{A}` slot compiles and
      // is accepted, so inserting half of a two-slot form and pressing
      // Validate returns a green verdict for a comparison the row never
      // offered. The difference is what the starter is built around, so
      // it is what gets inserted.
      insertAtCursor(
        `abs(${form.reference} - ${form.partner_reference})`);
      setReport(
        "Two-slot form inserted as abs(a - b): it is order-independent, " +
        "which is what\nsetting symmetric: true asserts. Compare it " +
        "against a threshold.", null);
      return;
    }
    insertAtCursor(form.reference);
  });

  byId("copy-selector").addEventListener("click", () => {
    const form = usableForm();
    if (!form) return;
    navigator.clipboard?.writeText(form.reference);
    const button = byId("copy-selector");
    button.textContent = "copied";
    setTimeout(() => { button.textContent = "Copy selector"; }, 1200);
  });

  byId("validate").addEventListener("click", () => {
    const socket = window.specolaSocket;
    const name = nameField().value.trim().toLowerCase().replace(/ /g, "_");
    const formula = formulaField().value.trim();
    const constants = constantsField().value;
    const revision = draft.revision;

    // The cheap refusals happen here, in the same order the terminal
    // writer applies them, so an author gets the same message without a
    // round trip.
    if (!name) return failValidation("Specification name is required.");
    if (draft.specs.some((spec) => spec.name === name)) {
      return failValidation(`Spec name '${name}' is already used.`);
    }
    if (!formula) return failValidation("Formula is required.");
    if (!socket) return failValidation("Not connected to the server.");

    const seq = ++draft.checkSeq;
    setReport("Checking…", null);
    socket.emit("spec_check", {
      revision, name, formula, constants,
    }, (reply) => {
      // Two gates. The sequence gate drops a reply overtaken by a later
      // check; the revision gate drops a reply for text that has since
      // been edited. Either alone would let a stale verdict through.
      if (seq !== draft.checkSeq) return;
      if (!reply || !reply.ok) {
        return failValidation(
          (reply && reply.error) || "The server could not run the check.");
      }
      if (reply.revision !== draft.revision) return;
      const body = reply.report || "";
      if (reply.accepted) {
        setReport(`✓ Current draft accepted\n${body}`, "accepted");
        // The server's own normalized entry, kept verbatim: parsing the
        // constants a second time here is a second chance to send
        // something the checker never saw.
        draft.validatedEntry = reply.entry;
      } else {
        setReport(`✗ Current draft rejected\n${body}`, "rejected");
        draft.validatedEntry = null;
      }
      draft.validatedRevision = draft.revision;
      draft.validationOk = Boolean(reply.accepted);
      byId("keep").disabled = !reply.accepted;
    });
  });

  function failValidation(message) {
    draft.checkSeq += 1;            // cancels any reply still in flight
    setReport(`✗ Current draft rejected\n${message}`, "rejected");
    draft.validatedRevision = draft.revision;
    draft.validationOk = false;
    draft.validatedEntry = null;
    byId("keep").disabled = true;
  }

  byId("keep").addEventListener("click", () => {
    if (!draft.validationOk ||
        draft.validatedRevision !== draft.revision ||
        !draft.validatedEntry) return;
    draft.specs.push({ ...draft.validatedEntry });
    draft.keptDirty = true;
    byId("save-specs").disabled = false;
    refreshKept();
    clearDraft();
  });

  function refreshKept() {
    const lines = [`Kept specs: ${draft.specs.length}`];
    draft.specs.forEach((spec, index) => {
      lines.push(`${index + 1} ✓ ${spec.name}`);
    });
    byId("kept-specs").textContent = lines.join("\n");
  }

  function clearDraft() {
    draft.programmatic = true;
    try {
      nameField().value = "";
      formulaField().value = "";
      constantsField().value = "{}";
    } finally {
      draft.programmatic = false;
    }
    draft.revision += 1;
    draft.validatedRevision = null;
    draft.validationOk = false;
    draft.validatedEntry = null;
    draft.checkSeq += 1;
    setReport("Not validated.", null);
    byId("keep").disabled = true;
  }

  byId("clear-draft").addEventListener("click", clearDraft);

  /* ---- modal --------------------------------------------------------------- */

  function openModal(title, body, actions) {
    byId("modal-title").textContent = title;
    byId("modal-body").textContent = body;
    const host = byId("modal-actions");
    host.textContent = "";
    for (const action of actions) {
      const button = document.createElement("button");
      button.textContent = action.label;
      if (action.danger) button.className = "danger";
      button.addEventListener("click", () => {
        byId("modal").hidden = true;
        action.run();
      });
      host.appendChild(button);
    }
    byId("modal").hidden = false;
    host.firstChild?.focus();
  }

  /* ---- saving -------------------------------------------------------------- */

  /* The browser never names the file. The server derives the destination
   * from the recording it was started on, which is why there is no path
   * field here to traverse and no directory a page can reach. */

  function save(collision) {
    const socket = window.specolaSocket;
    if (!socket) {
      return setSaveStatus("Not connected to the server.", "error");
    }
    if (!draft.specs.length) {
      return setSaveStatus(
        "Keep at least one validated specification first.", "error");
    }
    socket.emit("spec_save", {
      entries: draft.specs, collision,
    }, (reply) => {
      if (!reply) {
        return setSaveStatus("The server did not answer.", "error");
      }
      if (reply.ok) {
        draft.savedPath = reply.path;
        draft.keptDirty = false;
        // The kept list is emptied, because it has been published. It
        // used to survive the save, so keeping one more specification
        // and saving again wrote every earlier entry a second time into
        // a second file -- and `--specs 'authored/*.yaml'` would then
        // load each of those names twice.
        draft.specs = [];
        refreshKept();
        byId("save-specs").disabled = true;
        return setSaveStatus(
          `Saved ${reply.path} — kept list cleared`, "ok");
      }
      if (reply.collision) {
        setSaveStatus(reply.error || "", "error");
        openModal(
          "File exists",
          `${reply.path} already exists.\n\n` +
          "Overwrite it, or save alongside it under the next free name?",
          [
            { label: "Cancel", run: () => {} },
            { label: "Save as new name", run: () => save("rename") },
            { label: "Overwrite", danger: true,
              run: () => save("overwrite") },
          ]);
        return;
      }
      // A rejected entry means the scene no longer accepts something
      // kept earlier. The checker's own text says which and why.
      setSaveStatus(reply.error || "Could not save.", "error");
      if (reply.report) setReport(`✗ ${reply.error}\n${reply.report}`, "rejected");
    });
  }

  function setSaveStatus(message, kind) {
    const host = byId("save-status");
    host.textContent = message;
    host.classList.toggle("error", kind === "error");
    host.classList.toggle("ok", kind === "ok");
  }

  byId("save-specs").addEventListener("click", () => save("ask"));

  window.addEventListener("beforeunload", (event) => {
    if (!draft.keptDirty && !hasDraft()) return;
    // Kept specifications live only in this page until they are saved.
    event.preventDefault();
    event.returnValue = "";
  });

  /* ---- browse mode, help, keyboard ----------------------------------------- */

  for (const [id, mode] of [
    ["browse-class", "class"], ["browse-role", "role"],
  ]) {
    byId(id).addEventListener("click", () => setBrowseMode(mode));
  }

  function setBrowseMode(mode) {
    if (state.browseMode === mode) return;
    state.browseMode = mode;
    byId("browse-class").classList.toggle("on", mode === "class");
    byId("browse-role").classList.toggle("on", mode === "role");
    // Same invalidation the variant toggle does. `renderSignals` bails
    // when nothing is selected, but `renderPalette` does not: a reply
    // still in flight would repaint the rows and un-hide Insert for a
    // scope that is no longer selected, and a cold `fixture` palette is
    // 2.19 s, so that window is seconds wide.
    state.generation += 1;
    state.pending = null;
    state.signalPending = null;
    // The two modes have different scope shapes, so a selection made in
    // one is not addressable in the other.
    state.expanded.clear();
    state.selection = null;
    state.signal = null;
    state.signalIndex = null;
    state.signals = [];
    showSignal(null);
    byId("signal-list").textContent = "";
    byId("palette-list").textContent = "";
    showForm(null);
    renderClasses();
  }

  /* The terminal's `?` overlay, same keys, same grammar reminder. */
  const HELP = [
    ["↑ / ↓", "move within a list"],
    ["Enter", "open the focused row"],
    ["Tab", "next pane"],
    ["C", "class / role browse mode"],
    ["/", "search"],
    ["R", "raw / semantic"],
    ["Y", "explain the selected form"],
    ["I", "insert the selected selector into the formula"],
    ["?", "this help"],
    ["Esc", "close"],
    ["", ""],
    ["*", "wildcard binding"],
    ["{a, b}", "set of components"],
    ["@", "group or tag"],
    ["\\", "difference"],
    ["any: / all:", "quantifier"],
  ];

  function showHelp() {
    openModal(
      "Discover — keys and selector grammar",
      HELP.map(([key, meaning]) =>
        key ? `${key.padEnd(12)}${meaning}` : "").join("\n"),
      [{ label: "Close", run: () => {} }]);
  }

  byId("discover-help").addEventListener("click", showHelp);

  const LISTS = ["class-tree", "signal-list", "palette-list"];

  function focusedList() {
    const active = document.activeElement;
    for (const id of LISTS) {
      if (byId(id).contains(active)) {
        state.focusList = id;
        return byId(id);
      }
    }
    return byId(state.focusList || "class-tree");
  }

  function focusList(id) {
    state.focusList = id;
    const list = byId(id);
    const row = list.querySelector(".selected") ||
      [...list.children].find((child) => child.tabIndex === -1);
    if (row) {
      row.focus();
      row.scrollIntoView({ block: "nearest" });
    }
    return Boolean(row);
  }

  function moveFocus(list, delta) {
    const rows = [...list.children].filter(
      (row) => row.tagName === "LI" && row.className !== "signal-more");
    if (!rows.length) return;
    const current = rows.findIndex((row) => row === document.activeElement);
    const next = rows[Math.min(Math.max(current + delta, 0), rows.length - 1)]
      || rows[0];
    next.tabIndex = -1;
    next.focus();
    next.scrollIntoView({ block: "nearest" });
  }

  document.addEventListener("keydown", (event) => {
    if (byId("discover").hidden) return;
    if (!byId("modal").hidden) {
      if (event.key === "Escape") byId("modal").hidden = true;
      return;
    }
    const target = event.target;
    const typing = target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement;
    if (typing) {
      // Escape leaves a field; every other key belongs to the field.
      if (event.key === "Escape") {
        target.blur();
        byId("class-tree").focus();
        event.preventDefault();
      }
      return;
    }
    switch (event.key) {
      case "/":
        byId("discover-search").focus();
        event.preventDefault();
        break;
      case "c": case "C":
        setBrowseMode(state.browseMode === "class" ? "role" : "class");
        break;
      case "r": case "R":
        byId(state.variant === "semantic" ? "variant-raw"
          : "variant-semantic").click();
        break;
      case "?":
        showHelp();
        break;
      case "i": case "I":
        byId("insert-selector").click();
        break;
      case "y": case "Y":
        if (state.form) showForm(state.form);
        break;
      case "ArrowDown":
        moveFocus(focusedList(), 1);
        event.preventDefault();
        break;
      case "ArrowUp":
        moveFocus(focusedList(), -1);
        event.preventDefault();
        break;
      case "Tab": {
        // The help advertises this, and native tab order skips the rows
        // entirely -- they are list items, not controls.
        const order = LISTS.filter((id) => byId(id).children.length);
        if (!order.length) break;
        const at = order.indexOf(state.focusList || order[0]);
        const step = event.shiftKey ? -1 : 1;
        focusList(order[(at + step + order.length) % order.length]);
        event.preventDefault();
        break;
      }
      case "Enter":
        if (target instanceof HTMLElement && target.tagName === "LI") {
          target.click();
          event.preventDefault();
        }
        break;
      default:
        break;
    }
  });

  /* ---- readiness ----------------------------------------------------------- */

  window.addEventListener("specola-discovery", (event) => {
    const detail = event.detail || {};
    state.available = Boolean(detail.available);
    const tab = byId("tab-discover");
    tab.disabled = !state.available;
    tab.title = state.available
      ? "browse this run's scene"
      : (detail.reason || "no discovery snapshot for this run");
  });
})();
