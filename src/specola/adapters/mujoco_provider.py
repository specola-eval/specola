"""Bounded migration extraction provider for the MuJoCo adapter family.

Maintainer decision 2026-09-07 (option 2 in the plan
``docs/superpowers/plans/2026-09-07-geometry-distance-prefilter-and-provider.md``):
declare the geometry distance and completed-field quantities in full over a
bounded producer root, and accept the adapter's remaining scalar emissions
as a legacy population under a rule backed by the adapter source identity.
This provider does not establish producer-surface completeness; its root
population rule says so, and no validator is relaxed beyond the rule.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Tuple

from ..conformance.core_runner import EvidenceIdentity
from ..conformance.extraction import (
    CoverageLink, ExtractionManifest, ResultDeclaration, ResultSurface, SourceAddress,
    SurfaceClassification, SurfaceDisposition, manifest_digest,
)
from ..conformance.instances import RuntimePopulation
from ..conformance.provider import (
    ExtractionGuardError, ExtractionProvider, PreparedExtraction, register_extraction_provider,
)
from ..conformance.query_bindings import (
    ArgumentSchema, CallableParameterSurface, ParameterRole, QueryBindingRecord, QueryParameter,
)
from ..conformance.surface import ApiMember, MemberKind, ProducerIdentity, ProducerSurface, SurfaceRoot, surface_digest
from .mujoco_completed_fields import COMPLETION_ROOT
from .mujoco_completed_population import completed_population
from .mujoco_geom_pairs import cutoff, geometry_population, geometry_quantities
from .mujoco_queries import _digest, _native_library_path

if TYPE_CHECKING:
    from .mujoco import MujocoAdapter

_ROOT_RULE = ("bounded migration root (maintainer decision 2026-09-07): the geometry distance callable and the "
              "completion stage callables named by the projection record; not the producer API denominator")
_LEGACY_RULE = ("scalar properties emitted by the reviewed Specola adapter sources named in the legacy provenance; "
                "accepted as the legacy population during the bounded migration, without typed declarations")
_NO_ARGUMENTS = ArgumentSchema.from_document({"type": "object", "properties": {}, "additionalProperties": False})


def _file_identity(identity: str, path: Path) -> EvidenceIdentity:
    return EvidenceIdentity(identity, hashlib.sha256(path.read_bytes()).hexdigest())


class MujocoBoundedProvider(ExtractionProvider):
    """Compose the adapter's bound native subject into a prepared extraction."""

    def __init__(self):
        self._library_identity: Dict[Path, EvidenceIdentity] = {}
        self._source_cache: Dict[Tuple[str, int, int], EvidenceIdentity] = {}
        self._prepared_cache: Dict[tuple, dict] = {}

    # -- custody -------------------------------------------------------------

    def _adapter_source_custody(self, adapter) -> Tuple[EvidenceIdentity, ...]:
        """Identify every bundled adapter source in the instance's lineage."""
        identities = []
        for cls in type(adapter).__mro__:
            module = getattr(cls, "__module__", "")
            if not module.startswith("specola.adapters."):
                continue
            source = inspect.getsourcefile(cls)
            if source is None:
                raise ExtractionGuardError("adapter_source_unavailable", (module,))
            path = Path(source)
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            identity = self._source_cache.get(key)
            if identity is None:
                identity = _file_identity("specola/adapters/" + path.name, path)
                if len(self._source_cache) > 32:
                    self._source_cache.clear()
                self._source_cache[key] = identity
            if identity not in identities:
                identities.append(identity)
        if not identities:
            raise ExtractionGuardError("adapter_source_unavailable", (type(adapter).__qualname__,))
        return tuple(identities)

    def guard_source(self, adapter) -> None:
        namespace = vars(adapter)
        if "_native_resources" not in namespace:
            raise ExtractionGuardError("native_subject_unavailable",
                                       ("adapter was built without the MuJoCo adapter initializer",))
        namespace["_bounded_provider_custody"] = self._adapter_source_custody(adapter)

    def guard_runtime(self, adapter) -> None:
        recorded = vars(adapter).get("_bounded_provider_custody")
        if recorded is None:
            raise ExtractionGuardError("extraction_reset_required", ("no adapter source custody",))
        if self._adapter_source_custody(adapter) != recorded:
            raise ExtractionGuardError("adapter_source_changed", ("adapter source differs from the reset custody",))

    # -- prepared subject ------------------------------------------------------

    def _producer(self, adapter) -> Tuple[ProducerIdentity, object]:
        from importlib import metadata, import_module

        mujoco = import_module("mujoco")
        distribution = metadata.distribution("mujoco")
        library = _native_library_path(distribution)
        identity = self._library_identity.get(library)
        if identity is None:
            identity = EvidenceIdentity("mujoco/" + library.name, _digest(library))
            self._library_identity = {library: identity}
        return ProducerIdentity("mujoco", distribution.version, "native", identity), mujoco

    def _prepared_pieces(self, adapter: "MujocoAdapter"):
        """Build the reset-pinned records once per bound native subject.

        Every input is fixed when the adapter binds its native subject: the
        model, the query configuration, the pair registry and the producer
        build. The pairs come from the adapter's specification scope (issue
        #136), whose subset object changes only with the registry or the
        scope, so a scope change builds, and validates, a new population.
        Per read, only the epochs and the post-read guard change.
        """
        from .mujoco import _raw_struct

        projection = adapter._native_projection
        reader = adapter._geometry_reader
        records = adapter._geom_pairs_in_scope()[0]
        configuration_identity = adapter._configuration_identity
        if projection is None or reader is None or records is None or configuration_identity is None:
            raise ExtractionGuardError("native_subject_not_bound", ("bind the native subject before preparing",))
        model = adapter._native_subject_model
        key = (id(adapter), id(model), configuration_identity.sha256, id(records))
        cached = self._prepared_cache.get(key)
        if cached is not None and cached["model"] is model and cached["records"] is records and cached["reader"] is reader:
            return cached
        producer, mujoco = self._producer(adapter)
        from mujoco.introspect import functions

        functions_identity = _file_identity("mujoco/introspect/functions.py", Path(functions.__file__))
        distmax = cutoff(reader)
        completed = tuple(projection._completed_fields.quantities)
        geometry = geometry_quantities(reader.binding, distmax, reader.clamps_beyond_cutoff)
        quantities = completed + geometry

        # Surface: one bounded root over the callables the quantities name.
        names = sorted({source.member_id.removeprefix(COMPLETION_ROOT + ".") for quantity in quantities
                        for source in quantity.sources})
        for source in (source for quantity in quantities for source in quantity.sources):
            if not source.member_id.startswith(COMPLETION_ROOT + "."):
                raise ExtractionGuardError("unbounded_quantity_source", (source.member_id,))
        members = tuple(ApiMember(COMPLETION_ROOT, name, MemberKind.CALLABLE, json.dumps({
            "parameters": [parameter.name for parameter in functions.FUNCTIONS[name].parameters],
            "return_type": str(functions.FUNCTIONS[name].return_type.decl()),
        }, sort_keys=True)) for name in names)
        census = EvidenceIdentity("mujoco-bounded-root-census", hashlib.sha256(json.dumps(
            {"version": producer.version, "members": names}, sort_keys=True).encode()).hexdigest())
        surface = ProducerSurface(
            producer, _file_identity("specola/adapters/mujoco_provider.py", Path(__file__)), census,
            (SurfaceRoot(COMPLETION_ROOT, functions_identity, _ROOT_RULE),), members,
        )
        digest = surface_digest(surface)
        results = {}
        for quantity in quantities:
            for source in quantity.sources:
                results.setdefault(source, ResultDeclaration(
                    source, "native result " + source.result_path + " of " + source.member_id,
                    quantity.availability, quantity.provenance))
        result_surface = ResultSurface(digest, tuple(results[key] for key in sorted(results)), (functions_identity,))
        classifications = tuple(SurfaceClassification(
            SourceAddress(member.member_id), SurfaceDisposition.REPRESENTED_ELSEWHERE,
            "bounded root callable; its observation is its declared results", (functions_identity,),
        ) for member in members) + tuple(SurfaceClassification(
            source, SurfaceDisposition.EXPOSED, "declared native result of a bounded root callable",
            results[source].provenance) for source in sorted(results))
        coverage = tuple(CoverageLink(source, "declared native result to quantity", quantity.provenance,
                                      target_quantity=quantity.quantity_id)
                         for quantity in quantities for source in quantity.sources)
        coverage += tuple(CoverageLink(SourceAddress(source.member_id), "callable to its declared result",
                                       results[source].provenance, target_source=source)
                          for source in sorted(results))
        manifest = ExtractionManifest("mujoco-bounded-migration-" + producer.version, digest,
                                      classifications, quantities, coverage)

        # Bindings: the reader's own, plus one per completion callable. The
        # crosswalk row that materializes the final value designates the
        # owner-producing callable of a quantity with several stages.
        producing: Dict[str, str] = {}
        for contract in projection._completed_fields._contracts.values():
            rows = [row for row in contract["completion"]["field_output_crosswalk"]
                    if row.get("materialized_path", "value") == "value"]
            if len(rows) != 1:
                raise ExtractionGuardError("completed_quantity_without_producer", (contract["projection_id"],))
            producing[contract["projection_id"]] = "mujoco.completion." + rows[0]["callable_id"].removeprefix("c:@F@")
        by_callable: Dict[str, list] = {}
        for quantity in completed:
            for source in quantity.sources:
                by_callable.setdefault(source.member_id, []).append(quantity.quantity_id)
        bindings, declarations = [reader.binding], [reader.declaration]
        for callable_id in sorted(by_callable):
            name = callable_id.removeprefix(COMPLETION_ROOT + ".")
            binding_id = "mujoco.completion." + name
            outputs = tuple(sorted(set(by_callable[callable_id])))
            produces = any(producing.get(quantity_id) == binding_id for quantity_id in outputs)
            provenance = tuple(dict.fromkeys(item for quantity in completed if quantity.quantity_id in outputs
                                             for item in quantity.provenance))
            bindings.append(QueryBindingRecord(
                binding_id, callable_id,
                (QueryParameter("m", ParameterRole.CONTEXT, "const mjModel*"),
                 QueryParameter("d", ParameterRole.CONTEXT, "mjData*")),
                _NO_ARGUMENTS, outputs, "mujoco_completed_native_root_field",
                "specola.adapters.mujoco_completed_fields.MujocoCompletedFields.capture",
                "completed on the owned scratch copy within the projection interval",
                "declared completion stage on detached scratch data; effects guarded by the projection",
                "copy the completed native arrays after the stage sequence", "single declared outcome",
                provenance, population_resolver_id="mujoco_completed_native_root_field" if produces else None,
                population_arguments=_NO_ARGUMENTS if produces else None,
            ))
            declarations.append(CallableParameterSurface(callable_id, ("m", "d"), (functions_identity,)))
        missing = [quantity.quantity_id for quantity in completed if quantity.quantity_id not in producing]
        if missing:
            raise ExtractionGuardError("completed_quantity_without_producer", tuple(missing))

        resolved = completed_population(model, _raw_struct(adapter._data, "_data"), projection._guard,
                                        completed, asdict(configuration_identity), producing)
        domains, instances, query_domains = geometry_population(
            reader.binding, records, distmax, reader.clamps_beyond_cutoff,
            scoped=records is not adapter._geom_pair_records)
        cached = {
            "model": model, "records": records, "reader": reader,
            "surface": surface, "results": result_surface, "manifest": manifest,
            "manifest_sha256": manifest_digest(manifest), "digest": digest,
            "bindings": tuple(bindings), "declarations": tuple(declarations),
            "domains": tuple(resolved.domains) + domains,
            "instances": tuple(resolved.instances) + instances,
            "query_domains": tuple(resolved.query_domains) + query_domains,
            "configuration_identity": configuration_identity,
        }
        if len(self._prepared_cache) > 8:
            self._prepared_cache.clear()
        self._prepared_cache[key] = cached
        return cached

    def prepare(self, adapter, *, reset: bool) -> PreparedExtraction:
        pieces = self._prepared_pieces(adapter)
        model, records, reader = pieces["model"], pieces["records"], pieces["reader"]
        population = RuntimePopulation(
            pieces["digest"], pieces["manifest_sha256"], adapter.query_configuration,
            adapter.extraction_episode_epoch, adapter.query_configuration_epoch,
            pieces["domains"], pieces["instances"],
            legacy_scalar_provenance=vars(adapter)["_bounded_provider_custody"],
            query_domains=pieces["query_domains"],
            legacy_scalar_rule=_LEGACY_RULE,
        )

        def guard_after_read():
            if adapter._native_subject_model is not model or adapter._geometry_reader is not reader:
                raise ExtractionGuardError("native_subject_changed", ("model or geometry reader changed during the read",))
            if adapter._geom_pairs_in_scope()[0] is not records:
                raise ExtractionGuardError("native_subject_changed", ("geom pair registry or scope changed during the read",))

        return PreparedExtraction(pieces["surface"], pieces["results"], pieces["manifest"], population, guard_after_read,
                                  configuration_identity=pieces["configuration_identity"],
                                  bindings=pieces["bindings"], declarations=pieces["declarations"])


BOUNDED_PROVIDER = MujocoBoundedProvider()


def install_bounded_provider(adapter_class: type) -> None:
    """Bind the shared bounded provider to one enrolled MuJoCo-family class."""
    register_extraction_provider(adapter_class, BOUNDED_PROVIDER)
