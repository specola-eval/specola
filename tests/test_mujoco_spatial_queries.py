"""Query binding contracts before execution of the separate native oracle."""

import pytest

from specola.adapters.mujoco_spatial_queries import spatial_bindings
from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.query_bindings import (
    CallableParameterSurface, QueryBindingError, QueryConfigurationRequired,
    validate_query_arguments,
)
from specola.query_config import QueryConfiguration


def declarations():
    # Independent C signatures from the pinned producer header. These are a
    # bounded binding fixture, not the producer-surface denominator.
    groups = {
        "mj_angmomMat": ("m", "d", "mat", "body"),
        "mj_jacBody": ("m", "d", "jacp", "jacr", "body"),
        "mj_jacBodyCom": ("m", "d", "jacp", "jacr", "body"),
        "mj_jacGeom": ("m", "d", "jacp", "jacr", "geom"),
        "mj_jacSite": ("m", "d", "jacp", "jacr", "site"),
        "mj_jacSubtreeCom": ("m", "d", "jacp", "body"),
        "mj_objectAcceleration": ("m", "d", "objtype", "objid", "res", "flg_local"),
        "mj_objectVelocity": ("m", "d", "objtype", "objid", "res", "flg_local"),
    }
    evidence = (EvidenceIdentity("owned independent declaration fixture", "a" * 64),)
    return {name: CallableParameterSurface("owned.c." + name, names, evidence)
            for name, names in groups.items()}


def test_binding_requires_each_native_object_frame_configuration():
    plans = spatial_bindings("3.8.1", declarations())
    assert len(plans) == 16  # Six matrices and two queries over five native types.
    bindings = tuple(plan.binding for plan in plans)
    quantities = tuple(dict.fromkeys(q for b in bindings for q in b.quantity_ids))
    with pytest.raises(QueryConfigurationRequired) as error:
        validate_query_arguments(QueryConfiguration.empty(), bindings, tuple(declarations().values()), quantities)
    assert str(error.value).count("flg_local") == 10
    for plan in plans:
        assert all(parameter.default_json is None for parameter in plan.binding.parameters)


def test_missing_source_declaration_is_not_invented():
    declared = declarations()
    del declared["mj_jacSite"]
    with pytest.raises(QueryBindingError, match="source declaration"):
        spatial_bindings("3.8.1", declared)


def test_dropped_native_result_parameter_rejects_binding():
    declared = declarations()
    source = declared["mj_jacBody"]
    declared["mj_jacBody"] = CallableParameterSurface(
        source.callable_id, ("m", "d", "jacp", "body"), source.provenance)
    with pytest.raises(QueryBindingError, match="parameter population"):
        spatial_bindings("3.8.1", declared)


@pytest.mark.parametrize("version", ("3.2.3", "3.3.0", "3.3.1", "3.8.1", "3.10.0"))
def test_result_mapping_retains_each_pinned_output(version):
    plans = spatial_bindings(version, declarations())
    for plan in plans:
        outputs = plan.contract["outputs"]
        assert len(outputs) == len(plan.binding.quantity_ids)
        assert {o["native_parameter_name"] for o in outputs} == {
            p.name for p in plan.binding.parameters if p.role.value == "output_buffer"}
        assert len(plan.binding.quantity_ids) == len(set(plan.binding.quantity_ids))
    # Source declarations explicitly distinguish BODY and XBODY. Their owner
    # domains and binding addresses must remain different even at the same id.
    body = next(p for p in plans if p.binding.binding_id.endswith("mj_objectVelocity.mjOBJ_BODY"))
    xbody = next(p for p in plans if p.binding.binding_id.endswith("mj_objectVelocity.mjOBJ_XBODY"))
    assert body.owner_kind != xbody.owner_kind
    assert body.native_object_type != xbody.native_object_type
