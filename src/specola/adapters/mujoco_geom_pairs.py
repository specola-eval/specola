"""Geometry distance pairs as adapter entities and typed extraction outputs.

The shared MuJoCo adapter binds the isolated distance reader to its native
subject, allocates one ``geom_pair`` entity per universe pair, and emits the
two declared scalar projections of the native return. The bounded provider
uses the same records to declare the runtime population before the read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..adapter import PropertyMeta
from ..conformance.core_runner import EvidenceIdentity
from ..conformance.extraction import QuantityOutcome, QuantityRecord, SourceAddress
from ..conformance.instances import (
    ExpectedInstance, ExpectedOutcome, NativePathStep, QuantityDomain, QueryPopulationDomain,
    ScalarProjection, ValueConstraint,
)
from ..conformance.ledger import ExactnessKind
from ..conformance.query_bindings import (
    ArgumentSchema, QueryBindingError, bind_query_configuration, validate_query_arguments,
)
from ..extraction_observation import StructuredSample
from ..native_values import NativeValue
from .mujoco_completed_fields import _object, _record
from .mujoco_queries import GeomDistanceResult, IsolatedGeomDistanceReader, validate_mujoco_arguments

GEOM_PAIR_ENTITY_CLASS = "geom_pair"
CAPPED_KEY = "geom_distance_capped"
CLIPPED_KEY = "geom_distance_clipped"
GEOM_PAIR_PROPERTY_KEYS = (CAPPED_KEY, CLIPPED_KEY)
# ponytail: reset-time and contact-pair IDs count up from zero; pairs are
# allocated before reset registration, so they start above this base. A
# scene with more than a million other entities would collide; none does.
GEOM_PAIR_ENTITY_BASE = 1 << 20
EXECUTED_OUTCOMES = ("native_below_cutoff", "native_cutoff")
PRUNED_OUTCOME = "inferred_above_cutoff"
# Executed, returned above the cutoff: only on builds that do not clamp.
ABOVE_OUTCOME = "native_above_cutoff"
GEOM_PAIR_PROPERTY_META: Dict[str, PropertyMeta] = {
    CAPPED_KEY: PropertyMeta(
        group="geometry clearance",
        note="m; min(native mj_geomDistance return, distmax); a pruned pair reports distmax",
    ),
    CLIPPED_KEY: PropertyMeta(
        group="geometry clearance indicator",
        note="1 when the native return or the bounding-sphere bound is at or above distmax, else 0",
    ),
}


@dataclass(frozen=True)
class GeomPairRecord:
    """One universe pair with its reset-stable entity identity."""

    entity_id: int
    geom_ids: Tuple[int, int]
    body_ids: Tuple[int, int]
    label: str = ""
    tags: Tuple[str, ...] = ()


def pair_property_meta(record: GeomPairRecord) -> Dict[str, PropertyMeta]:
    """The metadata of one pair's two keys, carrying the pair's tags."""
    return {key: meta._replace(tags=tuple(meta.tags) + tuple(tag for tag in record.tags if tag not in meta.tags))
            for key, meta in GEOM_PAIR_PROPERTY_META.items()}


def bind_geometry_reader(model, data, configuration, *, guard, session) -> IsolatedGeomDistanceReader:
    """Construct and configure the distance reader on the adapter's native subject.

    Explicit query instances are outside the bounded provider and are refused
    here rather than silently dropped.
    """
    reader = IsolatedGeomDistanceReader(model, data, configuration, _guard=guard, _session=session,
                                        _defer_configuration=True)
    try:
        if configuration.instances():
            raise QueryBindingError("explicit geometry query instances are not part of the bounded provider")
        bindings, declarations = (reader.binding,), (reader.declaration,)
        validate_query_arguments(configuration, bindings, declarations, reader.binding.quantity_ids,
                                 validate_native_arguments=validate_mujoco_arguments)
        with session.observe(data):
            configured = bind_query_configuration(configuration, bindings, declarations,
                                                  reader.binding.quantity_ids, reader._resolve_operand,
                                                  validate_native_arguments=validate_mujoco_arguments)
        reader._install_configuration(configuration, configured)
    except BaseException:
        reader.close()
        raise
    return reader


def geom_pair_records(reader: IsolatedGeomDistanceReader, model) -> Dict[Tuple[int, int], GeomPairRecord]:
    """Allocate stable entity IDs over the reader's pair universe."""
    records = {}
    for index, (first, second) in enumerate(reader.population_pairs()):
        records[(first, second)] = GeomPairRecord(
            GEOM_PAIR_ENTITY_BASE + index, (first, second),
            (int(model.geom_bodyid[first]), int(model.geom_bodyid[second])),
        )
    return records


def configuration_identity(mujoco_module, model, configuration) -> EvidenceIdentity:
    """Identify the bound model bytes together with the query configuration."""
    buffer = np.zeros(mujoco_module.mj_sizeModel(model), dtype=np.uint8)
    mujoco_module.mj_saveModel(model, None, buffer)
    # The same digest without two copies of the model bytes (207 MB on LIBERO).
    digest = hashlib.sha256(buffer)
    digest.update(configuration.sha256.encode("ascii"))
    return EvidenceIdentity("mujoco-model-bytes-and-query-configuration", digest.hexdigest())


def cutoff(reader: IsolatedGeomDistanceReader) -> float:
    """The single population cutoff the bounded provider declares."""
    cutoffs = tuple(reader._cutoffs)
    if len(cutoffs) != 1:
        raise QueryBindingError("bounded provider expects exactly one geometry population cutoff")
    return float(cutoffs[0])


# Encoded NativeValue document layouts, independent of any captured value.
_FLOAT = _object({"kind": {"const": "float64"}, "data": {"type": "string"}})
_NULL = _object({"kind": {"const": "null"}})
_WITNESS = _object({"kind": {"const": "ndarray"}, "dtype": {"const": "<f8"},
                    "shape": {"const": [6]}, "data": {"type": "string"}})


def _return_layout(executed: bool) -> ArgumentSchema:
    return ArgumentSchema.from_document(_record({
        "native_return": _FLOAT if executed else _NULL, "lower_bound": _FLOAT, "distmax": _FLOAT,
    }))


def _witness_layout(executed: bool) -> ArgumentSchema:
    return ArgumentSchema.from_document(_WITNESS if executed else _NULL)


def _projections(field: str, distmax: float) -> Tuple[ScalarProjection, ...]:
    path = (NativePathStep("field", field),)
    return (ScalarProjection(CAPPED_KEY, path, "min_with_argument", distmax),
            ScalarProjection(CLIPPED_KEY, path, "ge_indicator", distmax))


def return_outcomes(distmax: float, clamps: bool = True) -> Tuple[ExpectedOutcome, ...]:
    """Admissible outcomes of the return quantity for one universe pair.

    A build that does not clamp (``clamps`` false) can also return above the
    cutoff. That outcome carries the same two projections, so the scalar
    keys read the cutoff and the clipped indicator on every build. Its
    constraint is ``ge`` because the constraint relations have no strict
    greater-than; the reader assigns it only for a return above the cutoff.
    """
    native = (NativePathStep("field", "native_return"),)
    bound = (NativePathStep("field", "lower_bound"),)
    above = () if clamps else (
        ExpectedOutcome(ABOVE_OUTCOME, _return_layout(True), _projections("native_return", distmax),
                        (ValueConstraint(native, "ge", distmax),)),
    )
    return (
        ExpectedOutcome("native_below_cutoff", _return_layout(True), _projections("native_return", distmax),
                        (ValueConstraint(native, "lt", distmax),)),
        ExpectedOutcome("native_cutoff", _return_layout(True), _projections("native_return", distmax),
                        (ValueConstraint(native, "eq", distmax),)),
        ExpectedOutcome(PRUNED_OUTCOME, _return_layout(False), _projections("lower_bound", distmax),
                        (ValueConstraint(bound, "ge", distmax),)),
    ) + above


def witness_outcomes(clamps: bool = True) -> Tuple[ExpectedOutcome, ...]:
    above = () if clamps else (ExpectedOutcome(ABOVE_OUTCOME, _witness_layout(True)),)
    return (
        ExpectedOutcome("native_below_cutoff", _witness_layout(True)),
        ExpectedOutcome("native_cutoff", _witness_layout(True)),
        ExpectedOutcome(PRUNED_OUTCOME, _witness_layout(False)),
    ) + above


_AVAILABILITY = {
    "native_below_cutoff": "mj_geomDistance executed on the completed scratch copy and returned below distmax",
    "native_cutoff": "mj_geomDistance executed and returned distmax; the reviewed 3.8.1 and 3.10.0 builds clamp beyond-cutoff pairs",
    PRUNED_OUTCOME: "not executed: the bounding-sphere lower bound |xpos_i-xpos_j|-rbound_i-rbound_j reached distmax",
    ABOVE_OUTCOME: ("mj_geomDistance executed and returned above distmax; the reviewed 3.3.0 and 3.3.1 builds do not "
                    "clamp the native CCD path and return a sentinel (1e10 on 3.3.0, the largest double on 3.3.1) "
                    "or, on 3.3.1, the unclamped distance"),
}
_RETURN_MEANING = ("Native signed distance between two collision-eligible geoms, bounded above by distmax; "
                   "the producer's native CCD pipeline can return zero for separated pairs at cutoffs of "
                   "0.2 m and above on 3.8.1 and 3.10.0 (session evidence 2026-09-07)")
_RETURN_MEANING_UNCLAMPED = ("Native signed distance between two collision-eligible geoms; on this build the native "
                             "CCD path does not bound the return at distmax and can return a sentinel or the "
                             "unclamped distance, which the capped projection bounds at distmax")
_WITNESS_MEANING = "Native witness segment from geom1 to geom2 in requested operand order; absent for a pruned pair"
_WITNESS_MEANING_UNCLAMPED = ("Native witness segment between geom1 and geom2; absent for a pruned pair. On this build "
                              "the native CCD path receives the operands in geometry-type order, so a mixed-type "
                              "pair's endpoints can be in that order instead of requested operand order, and the "
                              "buffer stays zero when the solver reports no witness")


def geometry_quantities(binding, distmax: float, clamps: bool = True) -> Tuple[QuantityRecord, ...]:
    """Manifest quantities for the geometry binding of the bounded provider."""
    callable_id = binding.callable_id
    provenance = binding.provenance
    return_id, witness_id = binding.quantity_ids
    common: Dict[str, Any] = dict(
        owner_resolver_id=binding.owner_resolver_id,
        index_domain=json.dumps({"owner": "geom_pair", "index": "canonical (min, max) native geom IDs",
                                 "population_resolver": binding.population_resolver_id}, sort_keys=True),
        exactness=ExactnessKind.EXACT,
        availability="one of the declared outcomes; a non-finite native return is inadmissible and fails the observation",
        freshness=binding.freshness, metadata_resolver_id="mujoco_bounded_geometry_binding",
        reader_id=binding.reader_id, provenance=provenance, owner_kind=GEOM_PAIR_ENTITY_CLASS,
    )
    return (
        QuantityRecord(
            quantity_id=return_id, sources=(SourceAddress(callable_id, "return"),),
            native_meaning=_RETURN_MEANING if clamps else _RETURN_MEANING_UNCLAMPED,
            unit=json.dumps({"system": "producer length unit", "symbol": "m"}),
            frame="world (scalar; frame-free)",
            transformation=("retain the native return with the bounding-sphere lower bound and distmax; "
                            "project min(value, distmax) and value >= distmax as scalar keys"),
            output_sink="structured_and_properties", property_pattern=", ".join(GEOM_PAIR_PROPERTY_KEYS),
            outcomes=tuple(QuantityOutcome(item.outcome_id, _AVAILABILITY[item.outcome_id], item.layout, provenance)
                           for item in return_outcomes(distmax, clamps)),
            **common,
        ),
        QuantityRecord(
            quantity_id=witness_id, sources=(SourceAddress(callable_id, "fromto"),),
            native_meaning=_WITNESS_MEANING if clamps else _WITNESS_MEANING_UNCLAMPED,
            unit=json.dumps({"system": "producer length unit", "symbol": "m", "components": 6}),
            frame="world Cartesian axes",
            transformation="copy the six native buffer components; null when the pair was not executed",
            output_sink="structured", property_pattern="structured-only",
            outcomes=tuple(QuantityOutcome(item.outcome_id, _AVAILABILITY[item.outcome_id], item.layout, provenance)
                           for item in witness_outcomes(clamps)),
            **common,
        ),
    )


def geometry_population(binding, records: Dict[Tuple[int, int], GeomPairRecord], distmax: float,
                        clamps: bool = True, scoped: bool = False):
    """Domains, expected instances and query domains for every given pair.

    ``scoped`` says ``records`` is the subset of the universe a
    specification scope selected (issue #136); the rule then says so.
    """
    return_id, witness_id = binding.quantity_ids
    rule = {"population_resolver": binding.population_resolver_id, "distmax": distmax, "pairs": len(records)}
    if scoped:
        rule["scope"] = "pairs a compiled specification slot can bind"
    rule = json.dumps(rule, sort_keys=True)
    empty = "" if records else ("no geom pair is in the specification scope" if scoped
                                else "no collision-eligible geom pair in this model")
    domains = tuple(QuantityDomain(quantity_id, rule, binding.provenance, empty)
                    for quantity_id in binding.quantity_ids)
    query_domains = tuple(QueryPopulationDomain(binding.binding_id, quantity_id, rule, binding.provenance, empty)
                          for quantity_id in binding.quantity_ids)
    returns, witnesses = return_outcomes(distmax, clamps), witness_outcomes(clamps)
    instances = []
    for pair, record in records.items():
        instances.append(ExpectedInstance(return_id, GEOM_PAIR_ENTITY_CLASS, pair, "", record.entity_id,
                                          returns, binding_id=binding.binding_id))
        instances.append(ExpectedInstance(witness_id, GEOM_PAIR_ENTITY_CLASS, pair, "", None,
                                          witnesses, binding_id=binding.binding_id))
    return domains, tuple(instances), query_domains


def geometry_samples(binding, result: GeomDistanceResult, record: GeomPairRecord) -> Tuple[StructuredSample, StructuredSample]:
    """Typed samples for one pair; the scalar values come from the same numbers."""
    return_id, witness_id = binding.quantity_ids
    value = NativeValue.capture({"native_return": result.native_return, "lower_bound": result.lower_bound,
                                 "distmax": result.distmax})
    witness = result.fromto if result.fromto is not None else NativeValue.capture(None)
    return (
        StructuredSample(return_id, GEOM_PAIR_ENTITY_CLASS, result.geom_ids, value, result.outcome,
                         record.entity_id, GEOM_PAIR_PROPERTY_KEYS, binding_id=binding.binding_id),
        StructuredSample(witness_id, GEOM_PAIR_ENTITY_CLASS, result.geom_ids, witness, result.outcome,
                         binding_id=binding.binding_id),
    )


def scalar_values(result: GeomDistanceResult) -> Optional[Dict[str, float]]:
    """The declared projections, or None when no outcome admits a scalar."""
    source = result.native_return if result.executed else result.lower_bound
    if source is None or (result.executed and not result.finite):
        return None
    value = float(source)
    return {CAPPED_KEY: min(value, result.distmax), CLIPPED_KEY: float(value >= result.distmax)}
