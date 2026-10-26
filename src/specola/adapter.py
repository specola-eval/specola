"""
adapter.py — Simulator-Neutral Adapter Interface
=================================================

Defines the boundary between simulator-specific state extraction and the
simulator-independent STL monitoring stack.

The adapter layer is responsible for:

- producing validated MonitorObservation snapshots;
- assigning stable entity IDs while an entity exists within an episode;
- exposing simulator-defined string entity classes;
- exposing arbitrary scalar properties;
- validating simulator-independent structural invariants;
- tracking which classes are currently active.

The adapter layer does not:

- define spatial component names such as px, py, or speed;
- interpret region properties or derive bounding-box components;
- validate STL formulas;
- require every entity in a class to expose identical properties;
- freeze the set of entities, classes, or properties for an episode;
- construct a fixed-dimensional optimization state space.

Dynamic entities and properties
-------------------------------
Entities may appear and disappear during an episode. New entity IDs, classes,
and property keys may also appear.

The only identity invariant enforced by the base adapter is that an entity ID
cannot silently change its entity class within the same episode. Entity IDs
may be reused after reset(), because reset begins a new identity namespace.

Template-method interface
-------------------------
Subclasses implement:

    _reset_impl()
    _get_observation_impl()

The public reset() and get_observation() methods perform validation
automatically, so simulator adapters cannot accidentally bypass the common
contract.
"""

from __future__ import annotations

import math

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Dict, FrozenSet, Mapping, NamedTuple, Optional, Set, Tuple

import numpy as np

from .entities import EntityState, SpecConfigError, TrackState
from .extraction_observation import ExtractionObservation, ExtractionObservationError
from .query_config import QueryConfiguration


_NO_METADATA_SCOPE_OVERRIDE = object()


class PropertyMeta(NamedTuple):
    """
    Optional metadata for one dynamic scalar property.

    group:
        Group label (e.g. "joint position", "joint angle"). Groups order the
        discovery listing, and a spec selector can filter on them
        (``robot_*[qacc_* @ joint angular acceleration]``), so a group
        label determines which keys a selector resolves to -- it is part
        of the authoring vocabulary, not only presentation.

        Two consequences for adapter authors. Keep a group's keys in one
        unit: a selector aggregates the whole group, so a group carrying
        both metres and radians makes a ``min``/``max`` over it compare
        raw numbers across units (``specola check`` warns, but the group
        should not provoke the warning). And treat a group label as a
        published name -- renaming one silently changes what existing
        specs resolve to.

    note:
        The key's unit (e.g. "m", "rad/s"), shown in the discovery
        listing's UNIT column. For components of one dimensionless
        quantity it carries the component label instead ("quat w").

    tags:
        Structural labels for the key, matched by the spec selector
        filter ``@ label`` alongside the group (e.g. ("arm", "right"),
        ("base",)). Empty for keys with no structural classification.

        Tags say *where* a key is; the group says *what kind of quantity*
        it is, which is why a tag may span units and a group should not.
        Both answer the same ``@ label`` filter, so the two vocabularies
        share one namespace: do not use a name as both a group and a tag,
        or the filter silently selects the union of the two axes.

    unit, family, component, frame:
        Optional discovery metadata for scalar presentation and conservative
        selector previews. ``family`` relates simulator-native keys that
        represent the same scalar quantity across entity instances;
        ``component`` identifies an explicitly declared scalar coordinate.
        Adapters should leave these unset when they do not have authoritative
        semantics. ``unit`` overrides the legacy ``note``-as-unit convention.
    """

    group: str
    note: Optional[str] = None
    tags: Tuple[str, ...] = ()
    unit: Optional[str] = None
    family: Optional[str] = None
    component: Optional[str] = None
    frame: Optional[str] = None


@dataclass
class MonitorObservation:
    """
    One monitor-facing simulator snapshot ω(t).

    Parameters
    ----------
    timestep:
        Discrete observation index supplied by the adapter.

        Timesteps must increase strictly within one monitor episode, but they
        are not required to begin at zero.

    time:
        Simulator or observation time in seconds.

        Time must be finite and nondecreasing within an episode.

    perceived_entities:
        Entities available to the monitor at this observation, indexed by
        entity ID.

        The population may change between observations.

    component_tags:
        Structural tag tuples per property key, indexed by entity ID —
        the observation-time counterpart of ``PropertyMeta.tags`` for
        entities that did not exist at reset (dynamic classes such as
        contact pairs). It must be a finite ``Mapping`` from present,
        nonnegative entity IDs to ``Mapping`` objects; no dict-like value
        is coerced. Binding eligibility reads it to evaluate compiled ``@``
        tag filters per step. Entities or keys absent from the mapping carry
        no tags; adapters only populate entries whose class declares
        ``component_tags_by_class``.

    extraction:
        Optional typed native quantities from the same instant. The envelope
        retains structured values and their scalar projection links. Its
        presence does not establish extraction completeness.
    """

    timestep: int
    time: float
    perceived_entities: Dict[int, EntityState] = field(default_factory=dict)
    component_tags: Mapping[int, Mapping[str, Tuple[str, ...]]] = field(
        default_factory=dict
    )
    extraction: Optional[ExtractionObservation] = None


class SimulatorAdapter(ABC):
    """
    Abstract interface for one simulator or perception backend.

    State, declaration, and criterion boundary (normative)
    ------------------------------------------------------
    An adapter is a faithful projection of simulator state and simulator
    declarations. Field inclusion is a semantic commitment: an included key
    asserts that its value, owner, unit, frame, and provenance mean what the
    adapter says they mean. Field omission is a semantic commitment too: an
    omitted simulator quantity has a categorical, documented reason for not
    being authorable. Both decisions need provenance and rationale.

    Every included field is author-visible through discovery and eagerly
    materialised in every observation where its owner is present. An
    approximation may be exposed only when it is identified as an
    approximation rather than silently substituted for the simulator
    quantity.

    Compiled specifications never decide what a field *means*. They may
    decide what is *read*: a monitor session calls
    ``set_extraction_scope()`` with the property keys its compiled specs
    require, and an adapter that honours the scope emits those keys and
    no others. The scope is a projection, never a redefinition — a
    scoped observation is the unscoped one with keys removed. Discovery,
    ``specola check``, and every other author-facing entrypoint leave the
    scope unset, so the surface an author sees is always the full one.
    A key that is not read is also not checked, so validation coverage
    of off-specification state is given up while a scope is in force;
    ``set_extraction_scope`` states that trade in full.

    Simulator declarations may identify entities, roles, ordering, and
    declared observation channels. They do not make benchmark criteria into
    state. Success and verdict APIs are forbidden as inputs to adapter
    extraction: an adapter must not call a benchmark success predicate,
    reward function, or task verdict to decide what state to emit.
    Reconciliation between a declared source and live simulator state fails
    loudly rather than guessing an identity or silently dropping the source.

    When a supported criterion depends on relations among declared entities,
    the author-visible vocabulary must expose sufficient generic operands and
    state to reconstruct those relations without exposing criterion truth.
    A relation population is independent of compiled specifications, goals,
    and criteria. If a false or absent relation is semantically meaningful,
    its design supplies an explicit state or binding rather than making the
    relation silently disappear. Every relation design documents whether its
    population is static or dynamic, ordered or unordered, how candidates are
    filtered, and how each result remains attributable to its operands. The
    adapter exposes the simulator inputs to the relation; it never emits the
    criterion's answer.

    Engine introspection can enumerate candidate fields, but it cannot infer
    ownership, units, groups, population, or physical meaning. Those require
    an engine capability declaration and, where applicable, explicit
    simulator policy. Structurally, dynamic contacts may appear and disappear
    like other dynamic entities or properties; assigning stable identity and
    useful semantics to them still requires an explicit adapter design.

    Subclass responsibilities
    -------------------------
    A subclass must:

    - call ``super().__init__()`` in its constructor;
    - implement ``dt``;
    - implement ``spatial_dims``;
    - implement ``_reset_impl()``;
    - implement ``_get_observation_impl()``;
    - emit stable entity IDs within one episode;
    - expose scalar simulator data through ``EntityState.properties``.

    Dynamic population
    ------------------
    The following may change during an episode:

    - the set of perceived entity IDs;
    - the set of active entity classes;
    - the properties exposed by an entity;
    - the number of valid monitor bindings.

    An entity ID may disappear and later reappear. When it reappears within
    the same episode, it must retain its original entity class.

    Optional class vocabulary
    -------------------------
    A subclass may override ``supported_classes`` when it knows the complete
    class vocabulary it can emit.

    Returning ``None`` means that the vocabulary is open and class validation
    must be deferred until runtime.
    """

    # Class-level defaults so partially constructed instances (tests build
    # adapters with ``__init__`` skipped) still resolve the metadata scope;
    # ``__init__`` shadows each with the same value per instance.
    _metadata_scope_override = _NO_METADATA_SCOPE_OVERRIDE
    _last_observation = None
    _extraction_scope = None
    _extraction_entity_scope = None
    _accepted_extraction_scope = None
    _extraction_provider = None
    _prepared_extraction = None

    def __init__(self) -> None:
        self._episode_started: bool = False
        self._episode_index: int = -1

        self._last_observation: Optional[MonitorObservation] = None
        self._query_configuration = QueryConfiguration.empty()
        self._query_configuration_epoch = 0
        self._accepted_query_configuration_epoch = None
        self._structured_metadata_by_quantity = {}

        # Identity history is retained for the duration of one episode,
        # including temporarily absent entities.
        self._entity_class_by_id: Dict[int, str] = {}
        # Entities whose structural validation already passed this
        # episode. The property-key schema is fixed once an episode
        # begins, so the full structural walk (types, shapes, name
        # strings) runs once per entity — on first appearance — instead
        # of every step (measured 2026-07-31: ~0.5s/step on RoboCasa's
        # ~2700 entities). Value-carrying checks stay elsewhere
        # (non-finite values raise in the MuJoCo emit path).
        self._validated_entity_ids: set = set()
        self._validated_property_names_by_entity: Dict[int, set] = {}

        # Classes currently visible in the most recent observation.
        self._active_classes: FrozenSet[str] = frozenset()

        # Union of classes observed since the most recent reset.
        self._seen_classes: Set[str] = set()

        # Property keys worth extracting, per entity class. None means
        # "everything", which is the only value any entrypoint other
        # than a monitor session ever sets.
        self._extraction_scope: Optional[Dict[str, FrozenSet[str]]] = None
        # Entities worth extracting, per entity class, as compiled slot
        # selectors (``MonitorEngine.extraction_entity_scope``). None means
        # every entity; set only together with a key scope.
        self._extraction_entity_scope: Optional[Dict[str, FrozenSet[tuple]]] = None
        # Metadata describes the last accepted observation, so a newly
        # installed scope becomes metadata-visible only after one observation
        # under that scope passes validation and is committed.
        self._accepted_extraction_scope: Optional[
            Dict[str, FrozenSet[str]]
        ] = None
        self._metadata_scope_override = _NO_METADATA_SCOPE_OVERRIDE

    def set_extraction_scope(
        self,
        scope: Optional[Mapping[str, FrozenSet[str]]],
    ) -> None:
        """Restrict extraction to the property keys the given classes need.

        ``scope`` maps an entity class to the property keys a compiled
        spec set requires from it. Every entity of a listed class is
        given those keys — bound or not — because binding eligibility is
        decided by scanning unbound entities for the components a slot
        requires. A class absent from the mapping contributes no keys.
        ``None`` (the default, and the value every non-monitor
        entrypoint leaves in place) restores full extraction.

        The scope survives ``reset()``: a session sets it once, after
        compiling its specs, and every subsequent episode of that
        session is scoped the same way.

        A scoped-out key is not read, so it is also not checked: the
        type and finiteness validation an adapter runs as it emits a
        value never sees it. A NaN in a property no specification reads
        therefore goes unreported while a scope is in force. That is the
        accepted trade — nothing consumes the value, and the modes whose
        job is to describe the simulator surface run unscoped and still
        check every property.

        The base class applies a fail-safe exact projection before common
        validation, so a new adapter cannot silently expose an incorrect
        scoped surface. Bundled live adapters must also use the scope while
        reading/transferring/materializing native fields; post-hoc projection
        preserves semantics but does not provide the intended monitoring
        optimization. ``MujocoAdapter`` already performs that early pruning.

        ``property_metadata_for`` applies the same projection after specs
        compile. ``Monitor.watch`` explicitly clears a prior episode's scope
        before reset and discovery. Standalone discovery and specification
        authoring construct fresh adapters and never install a scope, so they
        continue to see the complete surface. ``None`` also clears any
        entity scope (see ``set_extraction_entity_scope``).
        """
        self._extraction_scope = (
            None
            if scope is None
            else {
                str(entity_class): frozenset(keys)
                for entity_class, keys in scope.items()
            }
        )
        if scope is None:
            self._extraction_entity_scope = None

    def set_extraction_entity_scope(
        self,
        entities: Optional[Mapping[str, FrozenSet[tuple]]],
    ) -> None:
        """Narrow which entities of a class are read.

        ``entities`` holds the compiled slot selectors of
        ``MonitorEngine.extraction_entity_scope``, per class. Unlike the
        key scope it is not a base-class projection: an adapter honours it
        only for a class whose entities it builds per read and whose omitted
        members no compiled slot could bind. The MuJoCo family honours it
        for ``geom_pair`` (maintainer decision 2026-10-07, issue #136) and
        every other class keeps all of its entities. ``None`` restores every
        entity. A separate method, so an adapter that overrides
        ``set_extraction_scope(scope)`` keeps working unchanged.

        Installed before ``reset()``, an empty mapping (no compiled slot
        yet) lets the reset read none of those entities: ``Monitor.watch``
        and ``specola check`` reset this way (maintainer decision
        2026-10-07). The adapter's ``unread_entity_metadata()`` then
        describes what it left unread, and ``capabilities.reset_catalog``
        puts those entities in the reset catalog, so compilation still
        sees the whole scene.
        """
        self._extraction_entity_scope = (
            None
            if entities is None
            else {
                str(entity_class): frozenset(selectors)
                for entity_class, selectors in entities.items()
            }
        )

    def _requested_property_keys(
        self,
        entity_class: str,
    ) -> Optional[FrozenSet[str]]:
        """Return requested keys for one class, or ``None`` for full output."""

        if self._extraction_scope is None:
            return None
        return self._extraction_scope.get(entity_class, frozenset())

    def _property_is_requested(self, entity_class: str, key: str) -> bool:
        """Whether one valid property key belongs to the active scope."""

        requested = self._requested_property_keys(entity_class)
        return requested is None or key in requested

    def _metadata_extraction_scope(self):
        """Use prospective scope before acceptance, then accepted scope."""
        if self._metadata_scope_override is not _NO_METADATA_SCOPE_OVERRIDE:
            return self._metadata_scope_override
        if self._last_observation is None:
            return self._extraction_scope
        return self._accepted_extraction_scope

    def _project_property_metadata(
        self,
        entity_id: int,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Project metadata with the same class/key rule as observations."""

        if self._last_observation is None:
            return metadata
        entity = self._last_observation.perceived_entities.get(entity_id)
        if entity is None:
            return metadata
        scope = self._metadata_extraction_scope()
        if scope is None:
            return metadata
        requested = scope.get(entity.entity_class, frozenset())
        return {
            key: value
            for key, value in metadata.items()
            if not isinstance(key, str) or not key or key in requested
        }

    def _project_observation_to_scope(
        self,
        observation: MonitorObservation,
    ) -> MonitorObservation:
        """Apply the mandatory property projection without changing identity."""

        if self._extraction_scope is None or not isinstance(
            observation, MonitorObservation
        ):
            return observation
        if not isinstance(observation.perceived_entities, dict):
            return observation

        projected_entities: Dict[int, EntityState] = {}
        for entity_id, entity in observation.perceived_entities.items():
            if not isinstance(entity, EntityState) or not isinstance(
                entity.properties, Mapping
            ):
                projected_entities[entity_id] = entity
                continue
            requested = self._requested_property_keys(entity.entity_class)
            if requested is None:
                projected_entities[entity_id] = entity
                continue
            properties = {
                key: value
                for key, value in entity.properties.items()
                if not isinstance(key, str) or not key or key in requested
            }
            projected_entities[entity_id] = EntityState(
                entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                position=entity.position,
                velocity=entity.velocity,
                track_age=entity.track_age,
                track_state=entity.track_state,
                properties=properties,
            )

        component_tags = observation.component_tags
        if isinstance(component_tags, Mapping):
            projected_tags = {}
            for entity_id, tags in component_tags.items():
                entity = observation.perceived_entities.get(entity_id)
                if not isinstance(entity, EntityState) or not isinstance(tags, Mapping):
                    projected_tags[entity_id] = tags
                    continue
                requested = self._requested_property_keys(entity.entity_class)
                if requested is None:
                    projected_tags[entity_id] = tags
                    continue
                projected_tags[entity_id] = {
                    key: labels
                    for key, labels in tags.items()
                    if (
                        not isinstance(key, str)
                        or not key
                        or key not in entity.properties
                        or key in requested
                        or type(labels) is not tuple
                        or not labels
                        or any(
                            not isinstance(label, str) or not label
                            for label in labels
                        )
                    )
                }
            component_tags = projected_tags

        extraction = observation.extraction
        if type(extraction) is ExtractionObservation:
            extraction.validate_observation(observation.timestep, observation.time, observation.perceived_entities)
            extraction = extraction.project_scalar_links(projected_entities)
        return MonitorObservation(
            timestep=observation.timestep,
            time=observation.time,
            perceived_entities=projected_entities,
            component_tags=component_tags,
            extraction=extraction,
        )

    # ------------------------------------------------------------------
    # Simulator metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Nominal monitor update period in seconds.

        This is used when specifications express temporal bounds in seconds.
        """

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        """
        Length of every EntityState.position and EntityState.velocity vector.

        Common values are 2 and 3, but the base adapter only requires a
        positive integer. Interpretation of individual axes belongs to the
        parser and component-resolution layer.
        """

    @property
    def supported_classes(self) -> Optional[FrozenSet[str]]:
        """
        Complete class vocabulary supported by this adapter, when known.

        Returns
        -------
        Optional[FrozenSet[str]]
            A complete class set, or ``None`` when the adapter has an open or
            dynamically discovered vocabulary.

        Notes
        -----
        This is adapter capability metadata, not the set of classes currently
        visible in the scene.
        """
        return None

    @property
    def active_classes(self) -> FrozenSet[str]:
        """
        Entity classes present in the most recently accepted observation.

        This set may change at every timestep.
        """
        return self._active_classes

    @property
    def seen_classes(self) -> FrozenSet[str]:
        """
        Union of entity classes observed since the latest reset.
        """
        return frozenset(self._seen_classes)

    @property
    def episode_started(self) -> bool:
        """Whether reset() has successfully produced an initial observation."""
        return self._episode_started

    @property
    def episode_index(self) -> int:
        """Zero-based count of successfully started adapter episodes."""
        return self._episode_index

    @property
    def last_observation(self) -> Optional[MonitorObservation]:
        """Most recently validated observation, or None before reset()."""
        return self._last_observation

    # ------------------------------------------------------------------
    # Public template-method interface
    # ------------------------------------------------------------------

    @property
    def query_configuration(self) -> QueryConfiguration:
        """The immutable configured native subject, independent of scope."""
        return self._query_configuration

    @property
    def query_configuration_epoch(self) -> int:
        return self._query_configuration_epoch

    @property
    def extraction_episode_epoch(self) -> int:
        """The epoch being read, including the candidate reset observation."""
        return self._episode_index if self._episode_started else self._episode_index + 1

    def set_query_configuration(self, configuration: QueryConfiguration) -> None:
        """Install a detached subject; a changed subject requires a reset."""
        if type(configuration) is not QueryConfiguration:
            raise TypeError("query configuration must be a QueryConfiguration")
        if configuration != self._query_configuration:
            self._query_configuration = configuration
            self._query_configuration_epoch += 1

    def reset(self) -> MonitorObservation:
        """
        Begin a new monitor episode.

        The subclass resets its adapter-side episode state and discovers the
        simulator's current scene in ``_reset_impl()``. This interface does
        not imply ``env.reset()``: callers prepare or reset the simulator
        before starting the adapter episode. The base class then validates
        and records the resulting initial observation.

        Reset creates a new entity-ID namespace, so IDs from a previous
        episode may be reused.
        """
        if self._prepared_extraction is not None:
            raise RuntimeError("an extraction read is already active")
        self._episode_started = False
        self._last_observation = None
        self._accepted_extraction_scope = None
        self._accepted_query_configuration_epoch = None
        self._structured_metadata_by_quantity.clear()
        self._entity_class_by_id.clear()
        self._validated_entity_ids.clear()
        self._validated_property_names_by_entity.clear()
        self._active_classes = frozenset()
        self._seen_classes.clear()

        from .conformance.provider import adapter_extraction_status, guard_extraction_source

        provider = guard_extraction_source(self, reset=True)
        if provider is not None:
            self._bind_extraction_subject(reset=True)
        self._validate_adapter_metadata()

        observation = self._project_observation_to_scope(
            self._read_extraction_observation(reset=True, provider=provider)
        )
        self._validate_observation(observation, previous=None)

        # Commit state only after successful validation.
        self._record_observation(observation)
        self._episode_started = True
        self._episode_index += 1
        self._accepted_query_configuration_epoch = self._query_configuration_epoch

        return observation

    def get_observation(self) -> MonitorObservation:
        """
        Return the next validated monitor observation.

        Raises
        ------
        RuntimeError
            If reset() has not successfully started an episode.
        """
        if self._prepared_extraction is not None:
            raise RuntimeError("an extraction read is already active")
        if not self._episode_started:
            raise RuntimeError(
                "Adapter episode has not started. Call adapter.reset() before "
                "adapter.get_observation()."
            )

        if self._accepted_query_configuration_epoch != self._query_configuration_epoch:
            raise RuntimeError("Query configuration changed. Call adapter.reset() before observing the new subject.")

        from .conformance.provider import adapter_extraction_status, guard_extraction_source

        provider = guard_extraction_source(self, reset=False)
        if provider is not None:
            self._bind_extraction_subject(reset=False)
        self._validate_adapter_metadata()

        observation = self._project_observation_to_scope(
            self._read_extraction_observation(reset=False, provider=provider)
        )
        self._validate_observation(
            observation,
            previous=self._last_observation,
        )

        self._record_observation(observation)
        return observation

    def _bind_extraction_subject(self, *, reset: bool) -> None:
        """Refresh native subject handles before provider populations resolve."""

    def _read_extraction_observation(self, *, reset: bool, provider=None) -> MonitorObservation:
        """Validate a configured native population before applying scalar scope."""
        from .conformance.provider import ExtractionProvider, PreparedExtraction

        if self._prepared_extraction is not None:
            raise RuntimeError("an extraction read is already active")
        read = self._reset_impl if reset else self._get_observation_impl
        if provider is None:
            return read()
        if not isinstance(provider, ExtractionProvider):
            raise TypeError("adapter extraction provider has the wrong type")
        prepared = provider.prepare(self, reset=reset)
        if type(prepared) is not PreparedExtraction:
            raise TypeError("extraction provider returned an invalid prepared subject")
        prepared.validate_before_read(self)
        # The scalar scope selects output from this fixed native subject. It
        # cannot suppress a required reader or alter a query population. A
        # provider whose scalar population is rule-backed may keep the scope
        # active for those legacy scalars during the read; typed projections
        # are always read in full and projected after the join.
        scope = self._extraction_scope
        if not self._extraction_scope_during_read(prepared):
            self._extraction_scope = None
        self._prepared_extraction = prepared
        try:
            self._validate_prepared_extraction(prepared)
            observation = read()
            prepared.validate_after_read(observation, self)
        finally:
            self._prepared_extraction = None
            self._extraction_scope = scope
        return observation

    def _validate_prepared_extraction(self, prepared) -> None:
        """Reconcile producer-specific runtime layouts before reading values."""

    def _extraction_scope_during_read(self, prepared) -> bool:
        """Whether the scalar scope may restrict legacy scalars while reading.

        The default keeps the accepted full-subject read. An adapter whose
        prepared population declares a legacy scalar rule may return True:
        the join then requires only the typed projections, which that
        adapter must emit regardless of scope.
        """
        return False

    def render(self) -> None:
        """
        Refresh an on-screen view of the simulator, if one is supported.

        The base implementation is a no-op, so calling ``render()`` once
        per environment step is always safe regardless of the adapter or
        platform. Adapters with a live view (see ``MujocoAdapter``)
        override this; embedding code that wants a live view simply calls
        ``adapter.render()`` in its own step loop — the monitoring engine
        itself never renders.
        """

    def close_render(self) -> None:
        """
        Release any on-screen view opened by ``render()``.

        No-op in the base implementation and safe to call at any time,
        including without a prior ``render()``. Callers that enabled a
        live view should call this before process exit — leaving a
        viewer open can crash interpreter teardown (GL context
        destruction races the viewer thread).
        """

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        Return one HxWx3 uint8 RGB frame of the current simulator state,
        or ``None`` when the adapter has no frame source.

        Presentation-only capability for visualization sinks (see
        ``dashboard.recorder``): implementations render from simulator
        state and must not enable benchmark observation channels or
        change the policy's inputs. ``width``/``height`` are a
        best-effort hint — simulators whose cameras have a fixed
        resolution may return frames of their native size. ``camera``
        selects a named source where the adapter has one; ``None``
        means the adapter default. The base implementation returns
        ``None``, so calling it on any adapter is always safe.
        """
        return None

    # -- state tier (replay) ------------------------------------------------

    def state_signature(self) -> Optional[dict]:
        """Facts that decide whether a captured state means the same thing.

        A JSON-serialisable dict that must carry ``length``, the positive
        integer length of the array ``capture_state`` returns; the state
        recording writer refuses a signature without it. The other keys
        name the engine, its version and a digest of the loaded model.
        None when the adapter has no state tier; then ``--mode replay``
        of a Specola state recording is refused with that reason.
        """
        return None

    def capture_state(self) -> Optional[np.ndarray]:
        """The simulator state the current observation describes, flat float64, or None."""
        return None

    def restore_state(self, state: np.ndarray) -> None:
        """Put the simulator into ``state`` and bring derived quantities up to date.

        No physics step: the next ``get_observation`` reads this state.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot restore a recorded state"
        )

    def camera_names(self) -> Tuple[str, ...]:
        """
        Named frame sources ``capture_frame`` accepts, in adapter order;
        empty when the adapter has no named cameras. The first name,
        when present, is the adapter's default camera.
        """
        return ()

    def scene_geometry(self) -> Optional[dict]:
        """
        Static 3D scene description for the dashboard, or ``None`` when
        the adapter has no poseable rigid-body scene.

        Presentation-only capability, like ``capture_frame``: adapters
        that have one return ``{"json": dict, "meshes": bytes}``, the
        geometry pair of the dashboard recording format — a body/geom
        table plus one packed vertex-and-index buffer the geom entries
        index into. Called at most once per episode, after reset, so it
        may cost as much as a full model walk. The base implementation
        returns ``None``, so calling it on any adapter is safe.
        """
        return None

    def scene_poses(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Per-step body poses as ``(xpos (nbody, 3), xquat (nbody, 4))``
        float32 copies, or ``None`` when the adapter has no scene.

        Rows are the bodies of ``scene_geometry``'s table, in the same
        order; quaternions are ``(w, x, y, z)``. Runs on the simulator
        thread once per step, so implementations do copies only — no
        rendering, no simulator queries beyond the pose arrays. The
        arrays must own their data: the caller keeps them past the next
        step, while the simulator's own buffers are overwritten in
        place. The return must not alternate between ``None`` and arrays
        within an episode — the recording joins pose records to data
        lines by ordinal, so a skipped step would silently shift every
        later record. The base implementation returns ``None``.
        """
        return None

    @abstractmethod
    def _reset_impl(self) -> MonitorObservation:
        """
        Perform simulator-specific episode reset and entity discovery.

        The returned observation is validated by reset().
        """

    @abstractmethod
    def _get_observation_impl(self) -> MonitorObservation:
        """
        Extract the next simulator-specific observation.

        The returned observation is validated by get_observation().
        """

    # ------------------------------------------------------------------
    # Dynamic introspection helpers
    # ------------------------------------------------------------------

    def active_properties_by_entity(
        self,
    ) -> Dict[int, FrozenSet[str]]:
        """
        Property keys currently exposed by each perceived entity.

        No assumption is made that entities in one class expose identical
        properties.
        """
        if self._last_observation is None:
            return {}

        return {
            entity_id: frozenset(entity.properties.keys())
            for entity_id, entity
            in self._last_observation.perceived_entities.items()
        }

    def active_properties_by_class(
        self,
    ) -> Dict[str, FrozenSet[str]]:
        """
        Union of currently exposed property keys for each active class.

        This is discovery information only. It is not a guarantee that every
        entity in the class provides every returned property.
        """
        properties: Dict[str, Set[str]] = {}

        if self._last_observation is None:
            return {}

        for entity in self._last_observation.perceived_entities.values():
            properties.setdefault(entity.entity_class, set()).update(
                entity.properties.keys()
            )

        return {
            entity_class: frozenset(keys)
            for entity_class, keys in properties.items()
        }

    def entities_supporting_property(
        self,
        entity_class: str,
        property_name: str,
    ) -> FrozenSet[int]:
        """
        Return active entity IDs of a class that currently expose a property.

        This helper is useful for discovery and diagnostics. Formula-specific
        binding eligibility remains the responsibility of the monitor layer.
        """
        if self._last_observation is None:
            return frozenset()

        return frozenset(
            entity.entity_id
            for entity in self._last_observation.perceived_entities.values()
            if entity.entity_class == entity_class
            and property_name in entity.properties
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_adapter_metadata(self) -> None:
        """Validate metadata shared by every adapter implementation."""
        dt = self.dt

        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
            raise TypeError(
                f"Adapter dt must be a real scalar, got "
                f"{type(dt).__name__}"
            )

        dt_value = float(dt)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError(
                f"Adapter dt must be finite and positive, got {dt!r}"
            )

        dims = self.spatial_dims

        if (
            isinstance(dims, (bool, np.bool_))
            or not isinstance(dims, Integral)
        ):
            raise TypeError(
                f"Adapter spatial_dims must be an integer, got "
                f"{type(dims).__name__}"
            )

        if int(dims) <= 0:
            raise ValueError(
                f"Adapter spatial_dims must be positive, got {dims!r}"
            )

        supported = self.supported_classes

        if supported is not None:
            if not isinstance(supported, frozenset):
                raise TypeError(
                    "supported_classes must return a frozenset[str] or None"
                )

            for entity_class in supported:
                self._validate_entity_class(entity_class)

    def _validate_observation(
        self,
        observation: MonitorObservation,
        previous: Optional[MonitorObservation],
    ) -> None:
        """Validate one observation without mutating adapter state."""
        if not isinstance(observation, MonitorObservation):
            raise TypeError(
                f"Adapter must return MonitorObservation, got "
                f"{type(observation).__name__}"
            )

        self._validate_timestep(observation.timestep)
        self._validate_time(observation.time)

        if previous is not None:
            if observation.timestep <= previous.timestep:
                raise ValueError(
                    f"Observation timestep must increase strictly within an "
                    f"episode: previous={previous.timestep}, "
                    f"current={observation.timestep}"
                )

            if float(observation.time) < float(previous.time):
                raise ValueError(
                    f"Observation time cannot move backwards within an "
                    f"episode: previous={previous.time}, "
                    f"current={observation.time}"
                )

        entities = observation.perceived_entities

        if not isinstance(entities, dict):
            raise TypeError(
                "MonitorObservation.perceived_entities must be a dict "
                "mapping entity IDs to EntityState objects"
            )

        newly_validated = []
        newly_validated_property_names = {}
        for dictionary_id, entity in entities.items():
            self._validate_entity_dictionary_id(dictionary_id)
            if dictionary_id in self._validated_entity_ids:
                # VALUE checks run every step — values change every
                # step: identity/class mutation, track state, vector
                # shape and finiteness, property value types and
                # finiteness. Initial identity registration and supported-
                # class membership are first-appearance-only. Property
                # names remain dynamic, so each name is checked the first
                # time it appears.
                if not isinstance(entity, EntityState):
                    raise TypeError(
                        f"Observation entry {dictionary_id} must be "
                        f"EntityState, got {type(entity).__name__}"
                    )
                self._validate_entity_id(entity.entity_id)
                self._validate_entity_class(entity.entity_class)
                if int(dictionary_id) != entity.entity_id:
                    raise ValueError(
                        f"Observation dictionary key {dictionary_id} does "
                        f"not match entity.entity_id={entity.entity_id}"
                    )
                self._validate_entity_identity(entity)
                self._validate_track_state(entity)
                self._validate_spatial_vector(
                    entity.position,
                    entity_id=entity.entity_id,
                    vector_name="position",
                )
                self._validate_spatial_vector(
                    entity.velocity,
                    entity_id=entity.entity_id,
                    vector_name="velocity",
                )
                new_property_names = self._validate_property_values(
                    entity,
                    self._validated_property_names_by_entity[dictionary_id],
                )
                if new_property_names:
                    newly_validated_property_names[
                        dictionary_id
                    ] = new_property_names
                continue
            self._validate_entity_entry(dictionary_id, entity)
            newly_validated.append(dictionary_id)
            newly_validated_property_names[dictionary_id] = set(
                entity.properties
            )
        self._validate_component_tags(observation)

        from .conformance.verification import current_verification
        verification = current_verification(self)
        if verification is not None and observation.extraction is None:
            raise ExtractionObservationError("candidate evidence requires a structured observation")

        if observation.extraction is not None:
            if type(observation.extraction) is not ExtractionObservation:
                raise ExtractionObservationError("MonitorObservation.extraction must be a typed extraction observation")
            if observation.extraction.verification_only != (verification is not None):
                raise ExtractionObservationError("structured observation has the wrong verification mode")
            observation.extraction.validate_observation(observation.timestep, observation.time, entities)
            if (observation.extraction.queries != self.query_configuration
                    or observation.extraction.configuration_epoch != self.query_configuration_epoch
                    or observation.extraction.episode_epoch != self.extraction_episode_epoch):
                raise ExtractionObservationError("structured envelope does not match the installed query or episode identity")
            for metadata in observation.extraction.metadata:
                accepted = self._structured_metadata_by_quantity.get(metadata.quantity_id)
                if accepted is not None and metadata != accepted:
                    raise ExtractionObservationError("structured quantity meaning changed within an episode: " + metadata.quantity_id)
        if previous is not None:
            if (previous.extraction is None) != (observation.extraction is None):
                raise ExtractionObservationError("structured envelope presence changed within an episode")
            if previous.extraction is not None and observation.extraction is not None:
                for identity_field in ("episode_epoch", "configuration_epoch", "producer_surface_sha256", "manifest_sha256", "queries", "verification_only"):
                    if getattr(previous.extraction, identity_field) != getattr(observation.extraction, identity_field):
                        raise ExtractionObservationError("structured subject changed within an episode: " + identity_field)

        # Committed only after the whole observation passes: a rejected
        # observation must not leave earlier entities marked validated
        # (this method's contract is "validate without mutating").
        self._validated_entity_ids.update(newly_validated)
        if observation.extraction is not None:
            self._structured_metadata_by_quantity.update((item.quantity_id, item) for item in observation.extraction.metadata)
        for entity_id, property_names in newly_validated_property_names.items():
            self._validated_property_names_by_entity.setdefault(
                entity_id, set()
            ).update(property_names)

    @staticmethod
    def _validate_timestep(timestep: int) -> None:
        if (
            isinstance(timestep, (bool, np.bool_))
            or not isinstance(timestep, Integral)
        ):
            raise TypeError(
                f"Observation timestep must be an integer, got "
                f"{type(timestep).__name__}"
            )

        if int(timestep) < 0:
            raise ValueError(
                f"Observation timestep must be nonnegative, got {timestep}"
            )

    @staticmethod
    def _validate_time(time: float) -> None:
        if isinstance(time, (bool, np.bool_)) or not isinstance(time, Real):
            raise TypeError(
                f"Observation time must be a real scalar, got "
                f"{type(time).__name__}"
            )

        if not np.isfinite(float(time)):
            raise ValueError(
                f"Observation time must be finite, got {time!r}"
            )

    def _validate_entity_entry(
        self,
        dictionary_id: int,
        entity: EntityState,
    ) -> None:
        self._validate_entity_dictionary_id(dictionary_id)

        if not isinstance(entity, EntityState):
            raise TypeError(
                f"Observation entry {dictionary_id} must be EntityState, got "
                f"{type(entity).__name__}"
            )

        if int(dictionary_id) != entity.entity_id:
            raise ValueError(
                f"Observation dictionary key {dictionary_id} does not match "
                f"entity.entity_id={entity.entity_id}"
            )

        self._validate_entity_id(entity.entity_id)
        self._validate_entity_class(entity.entity_class)
        self._validate_entity_identity(entity)
        self._validate_track_state(entity)
        self._validate_spatial_vector(
            entity.position,
            entity_id=entity.entity_id,
            vector_name="position",
        )
        self._validate_spatial_vector(
            entity.velocity,
            entity_id=entity.entity_id,
            vector_name="velocity",
        )
        self._validate_properties(entity)

        supported = self.supported_classes
        if (
            supported is not None
            and entity.entity_class not in supported
        ):
            raise SpecConfigError(
                f"Adapter emitted unsupported entity class "
                f"{entity.entity_class!r} for entity {entity.entity_id}. "
                f"Declared classes: {sorted(supported)}"
            )

    @staticmethod
    def _validate_entity_dictionary_id(dictionary_id: int) -> None:
        if (
            isinstance(dictionary_id, (bool, np.bool_))
            or not isinstance(dictionary_id, Integral)
        ):
            raise TypeError(
                f"Entity dictionary keys must be integers, got "
                f"{type(dictionary_id).__name__}"
            )

    @staticmethod
    def _validate_entity_id(entity_id: int) -> None:
        if (
            isinstance(entity_id, (bool, np.bool_))
            or not isinstance(entity_id, Integral)
        ):
            raise TypeError(
                f"entity_id must be an integer, got "
                f"{type(entity_id).__name__}"
            )

        if int(entity_id) < 0:
            raise ValueError(
                f"entity_id must be nonnegative, got {entity_id}"
            )

    @staticmethod
    def _validate_entity_class(entity_class: str) -> None:
        """
        Validate only the simulator-independent class contract.

        Parser-specific naming restrictions do not belong here.
        """
        if not isinstance(entity_class, str):
            raise TypeError(
                f"entity_class must be str, got "
                f"{type(entity_class).__name__}"
            )

        if not entity_class.strip():
            raise ValueError("entity_class cannot be empty or whitespace")

    def _validate_entity_identity(self, entity: EntityState) -> None:
        """
        Ensure that an entity ID retains one class within an episode.

        New IDs and new classes are allowed at any observation.
        """
        previous_class = self._entity_class_by_id.get(entity.entity_id)

        if (
            previous_class is not None
            and previous_class != entity.entity_class
        ):
            raise SpecConfigError(
                f"Entity ID {entity.entity_id} changed class within one "
                f"episode: {previous_class!r} -> "
                f"{entity.entity_class!r}. Allocate a new entity ID when "
                f"identity or class changes."
            )

    @staticmethod
    def _validate_track_state(entity: EntityState) -> None:
        if not isinstance(entity.track_state, TrackState):
            raise TypeError(
                f"Entity {entity.entity_id} track_state must be TrackState, "
                f"got {type(entity.track_state).__name__}"
            )

        if (
            isinstance(entity.track_age, (bool, np.bool_))
            or not isinstance(entity.track_age, Integral)
        ):
            raise TypeError(
                f"Entity {entity.entity_id} track_age must be an integer, "
                f"got {type(entity.track_age).__name__}"
            )

        if int(entity.track_age) < 0:
            raise ValueError(
                f"Entity {entity.entity_id} track_age must be nonnegative, "
                f"got {entity.track_age}"
            )

    def _validate_spatial_vector(
        self,
        vector: Optional[np.ndarray],
        *,
        entity_id: int,
        vector_name: str,
    ) -> None:
        if vector is None:
            return

        if not isinstance(vector, np.ndarray):
            raise TypeError(
                f"Entity {entity_id} {vector_name} must be a NumPy array, "
                f"got {type(vector).__name__}"
            )

        if vector.ndim != 1:
            raise ValueError(
                f"Entity {entity_id} {vector_name} must be one-dimensional, "
                f"got shape {vector.shape}"
            )

        if len(vector) != int(self.spatial_dims):
            raise ValueError(
                f"Adapter declares spatial_dims={self.spatial_dims}, but "
                f"entity {entity_id} has {vector_name} length {len(vector)}"
            )

        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError(
                f"Entity {entity_id} {vector_name} must have numeric dtype, "
                f"got {vector.dtype}"
            )

        if np.iscomplexobj(vector):
            raise TypeError(
                f"Entity {entity_id} {vector_name} must contain real values"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Entity {entity_id} {vector_name} contains non-finite "
                f"values: {vector!r}"
            )

    @staticmethod
    def _validate_properties(entity: EntityState) -> None:
        """
        Validate arbitrary scalar properties without imposing a vocabulary.

        Property keys may vary across entities and observations.
        """
        if not isinstance(entity.properties, Mapping):
            raise TypeError(
                f"Entity {entity.entity_id} properties must be a mapping, "
                f"got {type(entity.properties).__name__}"
            )

        for property_name, value in entity.properties.items():
            if not isinstance(property_name, str):
                raise TypeError(
                    f"Entity {entity.entity_id} property names must be "
                    f"strings, got {type(property_name).__name__}"
                )

            if not property_name:
                raise ValueError(
                    f"Entity {entity.entity_id} has an empty property name"
                )

            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} must be a real scalar, not bool"
                )

            if not isinstance(value, Real):
                raise TypeError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} must be a real scalar, got "
                    f"{type(value).__name__}"
                )

            if not np.isfinite(float(value)):
                raise ValueError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} is non-finite: {value!r}"
                )

    @staticmethod
    def _validate_component_tags(observation: MonitorObservation) -> None:
        """Validate structural component tags against this observation."""
        component_tags = observation.component_tags
        if not isinstance(component_tags, Mapping):
            raise TypeError(
                "MonitorObservation.component_tags must be a mapping, got "
                f"{type(component_tags).__name__}"
            )

        entities = observation.perceived_entities
        for entity_id, property_tags in component_tags.items():
            if (
                isinstance(entity_id, (bool, np.bool_))
                or not isinstance(entity_id, Integral)
            ):
                raise TypeError(
                    "MonitorObservation.component_tags entity IDs must be "
                    f"integers, got {type(entity_id).__name__}"
                )
            if int(entity_id) < 0:
                raise ValueError(
                    "MonitorObservation.component_tags entity IDs must be "
                    f"nonnegative, got {entity_id}"
                )

            normalized_entity_id = int(entity_id)
            if normalized_entity_id not in entities:
                raise ValueError(
                    "MonitorObservation.component_tags entity ID "
                    f"{normalized_entity_id} does not name a present entity"
                )

            if not isinstance(property_tags, Mapping):
                raise TypeError(
                    "MonitorObservation.component_tags for entity "
                    f"{normalized_entity_id} must be a mapping, got "
                    f"{type(property_tags).__name__}"
                )

            properties = entities[normalized_entity_id].properties
            for component_name, tags in property_tags.items():
                if not isinstance(component_name, str):
                    raise TypeError(
                        "MonitorObservation.component_tags component names "
                        f"must be strings, got {type(component_name).__name__}"
                    )
                if not component_name:
                    raise ValueError(
                        "MonitorObservation.component_tags has an empty "
                        f"component name for entity {normalized_entity_id}"
                    )
                if component_name not in properties:
                    raise ValueError(
                        "MonitorObservation.component_tags component "
                        f"{component_name!r} for entity {normalized_entity_id} "
                        "does not name a property"
                    )
                if type(tags) is not tuple:
                    raise TypeError(
                        "MonitorObservation.component_tags tags must be a "
                        f"tuple, got {type(tags).__name__}"
                    )
                if not tags:
                    raise ValueError(
                        "MonitorObservation.component_tags tags must be "
                        "nonempty"
                    )
                for tag in tags:
                    if not isinstance(tag, str):
                        raise TypeError(
                            "MonitorObservation.component_tags tags must be "
                            f"strings, got {type(tag).__name__}"
                        )
                    if not tag:
                        raise ValueError(
                            "MonitorObservation.component_tags tags must be "
                            "nonempty strings"
                        )

    @staticmethod
    def _validate_property_values(
        entity: EntityState, validated_names: set
    ) -> set:
        """Validate current values and property names new to this episode.

        Same value rules as ``_validate_properties`` (real scalar, not
        bool, finite). Dynamic property names are checked once, when they
        first appear for the entity in an episode.
        ``math.isfinite`` over ``np.isfinite`` deliberately: identical
        verdicts for every Real, an order of magnitude cheaper on this
        per-property-per-step path.
        """
        if not isinstance(entity.properties, Mapping):
            raise TypeError(
                f"Entity {entity.entity_id} properties must be a mapping, "
                f"got {type(entity.properties).__name__}"
            )

        new_names = set()
        for property_name, value in entity.properties.items():
            if not isinstance(property_name, str):
                raise TypeError(
                    f"Entity {entity.entity_id} property names must be "
                    f"strings, got {type(property_name).__name__}"
                )
            if not property_name:
                raise ValueError(
                    f"Entity {entity.entity_id} has an empty property name"
                )
            if property_name not in validated_names:
                new_names.add(property_name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} must be a real scalar, not bool"
                )
            if not isinstance(value, Real):
                raise TypeError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} must be a real scalar, got "
                    f"{type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Entity {entity.entity_id} property "
                    f"{property_name!r} is non-finite: {value!r}"
                )
        return new_names

    # ------------------------------------------------------------------
    # State commit
    # ------------------------------------------------------------------

    def _record_observation(
        self,
        observation: MonitorObservation,
    ) -> None:
        """
        Commit an already validated observation to adapter bookkeeping.
        """
        for entity in observation.perceived_entities.values():
            self._entity_class_by_id.setdefault(
                entity.entity_id,
                entity.entity_class,
            )

        active_classes = {
            entity.entity_class
            for entity in observation.perceived_entities.values()
        }

        self._active_classes = frozenset(active_classes)
        self._seen_classes.update(active_classes)
        self._last_observation = observation
        self._accepted_extraction_scope = (
            None
            if self._extraction_scope is None
            else dict(self._extraction_scope)
        )
