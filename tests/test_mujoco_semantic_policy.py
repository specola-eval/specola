"""Global field capability stays separate from MuJoCo simulator policy."""

from __future__ import annotations

from dataclasses import fields
import inspect
from types import SimpleNamespace
from typing import List

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.dmcontrol import DMControlAdapter
from specola.adapters.mamujoco import MaMujocoAdapter
from specola.adapters.mujoco import (
    _SIGNAL_ADDRESSES,
    BodyRule,
    DeclaredEntitySpec,
    MujocoAdapter,
    MujocoSemanticPolicy,
    RegionSpec,
)
from specola.adapters.robocasa import RoboCasaAdapter


def _xml(body_name: str, timestep: float = 0.002) -> str:
    return """
    <mujoco>
      <option timestep="{timestep}"/>
      <worldbody>
        <body name="{body}">
          <joint name="{body}_hinge" type="hinge"/>
          <geom name="{body}_geom" type="box" size="0.02 0.02 0.02"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="{body}_motor" joint="{body}_hinge"/>
      </actuator>
    </mujoco>
    """.format(body=body_name, timestep=timestep)


class _SwappableEnv:
    def __init__(self, body_name: str, timestep: float = 0.002) -> None:
        self.replace(body_name, timestep)

    def replace(self, body_name: str, timestep: float) -> None:
        self.model = mujoco.MjModel.from_xml_string(_xml(body_name, timestep))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


class _DynamicPolicyAdapter(MujocoAdapter):
    def __init__(self, env, **kwargs) -> None:
        self.policy_builds: List[str] = []
        super().__init__(env, **kwargs)

    def _default_body_rules(self):
        body_name = next(name for name in self._names("body") if name != "world")
        self.policy_builds.append(body_name)
        return (
            BodyRule(
                match=lambda name, selected=body_name: name == selected,
                cls="robot",
                priority=0,
                joint_scope="subtree",
            ),
        )

    def _discover_extra_regions(self):
        body_name = next(name for name in self._names("body") if name != "world")
        return (
            RegionSpec(
                name="{}_region".format(body_name),
                source="fixed",
                center=(0.0, 0.0, 0.0),
                half_extents=(0.1, 0.1, 0.1),
            ),
        )


def _classes_and_labels(adapter: MujocoAdapter):
    observation = adapter.reset()
    return {
        (entity.entity_class, adapter.entity_labels[entity_id])
        for entity_id, entity in observation.perceived_entities.items()
    }


def test_policy_surface_has_no_availability_or_materialisation_selector() -> None:
    # robot_bodies is semantic ownership (which pairs classify as
    # robot_contact), not an availability or materialisation selector:
    # both contact classes exist and materialize on every adapter
    # regardless of the robot set.
    assert {item.name for item in fields(MujocoSemanticPolicy)} == {
        "body_rules",
        "regions",
        "declared_entities",
        "robot_bodies",
    }
    signature = inspect.signature(MujocoAdapter._semantic_policy)
    assert set(signature.parameters) == {"self"}
    forbidden = (
        "field",
        "reader",
        "availability",
        "material",
        "extract",
        "spec",
    )
    assert not any(
        token in item.name
        for item in fields(MujocoSemanticPolicy)
        for token in forbidden
    )


def test_policy_rejects_conflicting_body_region_and_entity_declarations() -> None:
    with pytest.raises(ValueError, match="duplicate body-rule match"):
        MujocoSemanticPolicy(
            body_rules=(
                BodyRule(match="thing", cls="robot"),
                BodyRule(match="thing", cls="object"),
            )
        )

    region = RegionSpec(
        name="zone",
        source="fixed",
        center=(0.0, 0.0, 0.0),
        half_extents=(0.1, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="duplicate region name"):
        MujocoSemanticPolicy(regions=(region, region))

    first = DeclaredEntitySpec(
        declaration_key="target",
        name="target",
        entity_class="target",
        source_kind="task_declaration",
        source_name="_get_pos_goal",
        source_provenance="fixture:1",
    )
    second = DeclaredEntitySpec(
        declaration_key="target",
        name="other",
        entity_class="object",
        source_kind="task_declaration",
        source_name="_get_pos_objects",
        source_provenance="fixture:2",
    )
    with pytest.raises(ValueError, match="duplicate declaration key"):
        MujocoSemanticPolicy(declared_entities=(first, second))


def test_explicit_policy_overrides_compose_without_field_selection() -> None:
    env = _SwappableEnv("alpha")
    explicit_rule = BodyRule(match="alpha", cls="object")
    adapter = _DynamicPolicyAdapter(
        env,
        body_rules=(explicit_rule,),
        regions=(),
    )
    labels = _classes_and_labels(adapter)

    assert labels == {
        ("object", "alpha"),
        ("body", "world (MuJoCo body)"),
        ("geom", "alpha_geom (MuJoCo geom)"),
        ("simulation", "simulation (global state)"),
    }
    assert adapter.semantic_policy.body_rules == (explicit_rule,)
    assert adapter.semantic_policy.regions == ()
    assert adapter.field_surface_report.included
    assert adapter.field_surface_report.unclassified == ()


def test_reset_recomputes_dt_policy_capability_owners_and_vocabulary() -> None:
    env = _SwappableEnv("alpha", timestep=0.002)
    adapter = _DynamicPolicyAdapter(env)

    first = _classes_and_labels(adapter)
    first_report = adapter.field_surface_report
    assert ("robot", "alpha") in first
    assert ("region", "alpha_region") in first
    assert adapter.dt == pytest.approx(0.002)
    assert adapter.policy_builds[-1] == "alpha"

    env.replace("beta", timestep=0.007)
    with pytest.raises(RuntimeError, match="changed during an active"):
        adapter.get_observation()

    second = _classes_and_labels(adapter)
    assert ("robot", "beta") in second
    assert ("region", "beta_region") in second
    assert not any(label.endswith("alpha") for _, label in second)
    assert adapter.dt == pytest.approx(0.007)
    assert adapter.policy_builds[-1] == "beta"
    assert adapter.field_surface_report == first_report

    robot_id = adapter.name_to_id["beta"]
    metadata = adapter.property_metadata(robot_id)
    observation = adapter.get_observation()
    assert set(metadata) == set(observation.perceived_entities[robot_id].properties)
    assert any("beta_hinge" in key for key in metadata)
    assert not any("alpha_hinge" in key for key in metadata)


def test_joint_signal_subset_api_is_removed_with_deliberate_migration_error() -> None:
    assert "joint_signals" not in inspect.signature(
        MujocoAdapter.__init__
    ).parameters
    assert "joint_signals" not in inspect.signature(
        RoboCasaAdapter.__init__
    ).parameters

    env = _SwappableEnv("alpha")
    with pytest.raises(TypeError, match="joint_signals=.*global.*eager"):
        MujocoAdapter(
            env,
            body_rules=(BodyRule(match="alpha", cls="object"),),
            joint_signals=("qpos",),
        )
    with pytest.raises(TypeError, match="joint_signals=.*global.*eager"):
        RoboCasaAdapter(env, joint_signals=("qpos",))


def test_every_live_curated_joint_family_is_eager_and_read_only() -> None:
    env = _SwappableEnv("alpha")
    adapter = MujocoAdapter(
        env,
        body_rules=(
            BodyRule(match="alpha", cls="object", joint_scope="body"),
        ),
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    live = tuple(
        field for field in _SIGNAL_ADDRESSES if hasattr(env.data, field)
    )

    assert adapter.joint_signals == live
    for field in live:
        assert any(key.startswith("{}_".format(field)) for key in entity.properties)
    with pytest.raises(AttributeError):
        setattr(adapter, "joint_signals", ("qpos",))


class _Pointer:
    def __init__(self, ptr) -> None:
        self.ptr = ptr


class _Physics:
    def __init__(self, env: _SwappableEnv) -> None:
        self.model = _Pointer(env.model)
        self.data = _Pointer(env.data)


class _DynamicDMEnv:
    def __init__(self, body: str, control_dt: float) -> None:
        self.replace(body, control_dt)

    def replace(self, body: str, control_dt: float) -> None:
        raw = _SwappableEnv(body)
        self.physics = _Physics(raw)
        self._control_dt = control_dt

    def control_timestep(self) -> float:
        return self._control_dt


def test_dmcontrol_root_policy_follows_replaced_physics_model() -> None:
    env = _DynamicDMEnv("walker_a", 0.02)
    adapter = DMControlAdapter(env)
    assert ("robot", "walker_a") in _classes_and_labels(adapter)
    assert adapter.dt == pytest.approx(0.02)

    env.replace("walker_b", 0.04)
    assert ("robot", "walker_b") in _classes_and_labels(adapter)
    assert "walker_a" not in adapter.name_to_id
    assert adapter.dt == pytest.approx(0.04)


class _DynamicParallelEnv:
    def __init__(self, body: str) -> None:
        self.unwrapped = SimpleNamespace()
        self.replace(body)

    def replace(self, body: str) -> None:
        inner = _SwappableEnv(body)
        self.unwrapped.single_agent_env = SimpleNamespace(
            unwrapped=inner,
        )
        self.agents = []


def test_mamujoco_root_policy_follows_replaced_inner_model() -> None:
    env = _DynamicParallelEnv("agent_root_a")
    adapter = MaMujocoAdapter(env)
    assert ("robot", "agent_root_a") in _classes_and_labels(adapter)

    env.replace("agent_root_b")
    assert ("robot", "agent_root_b") in _classes_and_labels(adapter)
    assert "agent_root_a" not in adapter.name_to_id
