"""Controlled integration tests for the eager MuJoCo materializers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapter import PropertyMeta
from specola.adapters.mujoco import (
    BodyRule,
    DeclaredEntitySpec,
    MujocoAdapter,
    MujocoJointOwner,
    RegionSpec,
    _XAXIS_JOINT_TYPE_APPLICABILITY,
)
from specola.adapters.mujoco_contract import INCLUDED_FIELDS
from specola.entities import EntityState, TrackState


_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="tracked" pos="0 0 1">
      <inertial pos="0 0 0.1" mass="1"
                diaginertia="0.01 0.02 0.03"/>
      <joint name="slide_joint" type="slide" axis="1 0 0"/>
      <geom name="raw_geom" type="box" size="0.1 0.2 0.3"/>
      <site name="shared" type="sphere" size="0.02"/>
      <site name="box_site" type="box" size="0.01 0.02 0.03"/>
      <site name="capsule_site" type="capsule" size="0.01 0.02"/>
      <site name="marker_site" type="sphere" size="0.001"/>
    </body>
    <body name="mocap_body" mocap="true" pos="0 0 1">
      <geom name="mocap_geom" type="sphere" size="0.01"/>
    </body>
  </worldbody>
  <equality>
    <weld name="test_eq" body1="tracked" body2="mocap_body"
          active="false"/>
  </equality>
  <tendon>
    <fixed name="test_tendon">
      <joint joint="slide_joint" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="motor" joint="slide_joint"/>
  </actuator>
  <sensor>
    <jointpos name="joint_sensor" joint="slide_joint"/>
  </sensor>
</mujoco>
"""


def _env(xml: str = _XML):
    model = mujoco.MjModel.from_xml_string(xml)
    marker_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "marker_site"
    )
    if marker_id >= 0:
        model.site_size[marker_id] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def _dense_row_jacobian(
    env, values_name: str, row_count: int, column_count: int
) -> np.ndarray:
    """Return a dense (row, dof) Jacobian from either MuJoCo layout.

    MuJoCo chooses a dense or sparse Jacobian per ``mjOption.jacobian``,
    whose default is auto: the same field is a ``(rows, nv)`` array on one
    model and a flat array of nonzeros with CSR metadata on another. An
    oracle that indexes ``[row][dof]`` therefore only holds in the dense
    case, and silently depends on model size.

    This expands the sparse form directly from MuJoCo's own
    ``*_rownnz`` / ``*_rowadr`` / ``*_colind`` arrays rather than calling
    the adapter's expander, so the assertions using it stay independent of
    the code under test.
    """
    raw = np.asarray(getattr(env.data, values_name), dtype=float)
    if raw.shape == (row_count, column_count):
        return raw

    owner = next(
        (
            candidate
            for candidate in (env.data, env.model)
            if all(
                hasattr(candidate, "{}_{}".format(values_name, suffix))
                for suffix in ("rownnz", "rowadr", "colind")
            )
        ),
        None,
    )
    assert owner is not None, (
        "{} is {} but carries no sparse row metadata".format(
            values_name, raw.shape
        )
    )
    rownnz = np.asarray(
        getattr(owner, "{}_rownnz".format(values_name)), dtype=int
    ).reshape(-1)
    rowadr = np.asarray(
        getattr(owner, "{}_rowadr".format(values_name)), dtype=int
    ).reshape(-1)
    colind = np.asarray(
        getattr(owner, "{}_colind".format(values_name)), dtype=int
    ).reshape(-1)
    values = raw.reshape(-1)
    dense = np.zeros((row_count, column_count), dtype=float)
    for row in range(row_count):
        start = int(rowadr[row])
        for index in range(start, start + int(rownnz[row])):
            dense[row, int(colind[index])] = values[index]
    return dense


def _adapter(env=None, **kwargs) -> MujocoAdapter:
    if env is None:
        env = _env()
    return MujocoAdapter(
        env,
        body_rules=(
            BodyRule(
                match="tracked",
                cls="object",
                joint_scope="body",
            ),
        ),
        **kwargs
    )


def test_all_included_fields_are_dispatched_with_concrete_runtime_status() -> None:
    adapter = _adapter()
    observation = adapter.reset()

    assert set(adapter.field_materialization) == set(INCLUDED_FIELDS)
    assert {
        record.eager_materializer_id
        for record in INCLUDED_FIELDS.values()
    } == set(adapter._MATERIALIZER_METHODS)

    for name, status in adapter.field_materialization.items():
        assert status.field_name == name
        if status.scene_applicable:
            assert status.binding_present
            assert status.emitted_keys
        else:
            assert status.reason

    object_id = adapter.name_to_id["tracked"]
    properties = observation.perceived_entities[object_id].properties
    assert properties["qacc_smooth_slide_joint"] == pytest.approx(
        adapter._data.qacc_smooth[0]
    )
    assert properties["xipos_tracked_2"] == pytest.approx(
        adapter._data.xipos[1][2]
    )
    assert properties["xipos_tracked_2"] != pytest.approx(
        properties["xpos_tracked_2"]
    )
    assert properties["cinert_tracked_9"] == pytest.approx(
        adapter._data.cinert[1][9]
    )
    metadata = adapter.property_metadata(object_id)
    assert metadata["cdof_slide_joint_0_0"].note == "rad/m"
    assert metadata["cdof_slide_joint_0_3"].note == "1"
    assert metadata["cdof_dot_slide_joint_0_0"].note == "rad/(m·s)"
    assert metadata["cdof_dot_slide_joint_0_3"].note == "1/s"

    simulation_id = adapter.name_to_id["simulation:global"]
    simulation = observation.perceived_entities[simulation_id]
    assert {
        "energy_potential",
        "energy_kinetic",
        "inertia_matrix_dof_0_dof_0",
    } <= set(simulation.properties)
    assert simulation.properties[
        "inertia_matrix_dof_0_dof_0"
    ] == pytest.approx(adapter._data.qM[0])

    for entity_id, entity in observation.perceived_entities.items():
        assert set(entity.properties) == set(
            adapter.property_metadata(entity_id)
        )

    for field_name, owner_name in (
        ("geom_xpos", "geom:raw_geom"),
        ("mocap_pos", "mocap:mocap_body"),
    ):
        owner_id = adapter.name_to_id[owner_name]
        assert (
            "entity[{}].position".format(owner_id)
            in adapter.field_materialization[field_name].emitted_keys
        )


@pytest.mark.xfail(strict=True, reason=(
    "The native image guard rejects a spoofed mujoco.__version__ "
    "(MuJoCo module and distribution version differ); the 3.10 fullM "
    "signature branch needs a real 3.10 environment row."))
def test_inertia_matrix_uses_mujoco_310_fullm_signature(monkeypatch) -> None:
    env = _env()
    original_full_m = mujoco.mj_fullM
    seen = {}

    def full_m_310(model, data, destination):
        seen["data"] = data
        original_full_m(model, destination, np.asarray(data.qM))

    monkeypatch.setattr(mujoco, "__version__", "3.10.0")
    monkeypatch.setattr(mujoco, "mj_fullM", full_m_310)

    adapter = _adapter(env)
    observation = adapter.reset()

    assert seen["data"] is env.data
    simulation_id = adapter.name_to_id["simulation:global"]
    assert observation.perceived_entities[simulation_id].properties[
        "inertia_matrix_dof_0_dof_0"
    ] == pytest.approx(env.data.qM[0])


def test_classified_mocap_body_and_raw_input_have_distinct_positions() -> None:
    env = _env()
    mocap_body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "mocap_body",
    )
    env.data.mocap_pos[0] = np.asarray([1.0, 2.0, 3.0])
    adapter = MujocoAdapter(
        env,
        body_rules=(
            BodyRule("tracked", "object", joint_scope="body"),
            BodyRule("mocap_body", "robot", joint_scope="body"),
        ),
    )

    observation = adapter.reset()
    body_entity_id = adapter.name_to_id["mocap_body"]
    raw_entity_id = adapter.name_to_id["mocap:mocap_body"]
    assert body_entity_id != raw_entity_id

    body = observation.perceived_entities[body_entity_id]
    raw = observation.perceived_entities[raw_entity_id]
    # Positions are completed on the detached scratch copy (extraction
    # design 2026-09-04), so the classified mocap body already sits at its
    # input; the stored-stage scalar below still reads the live xpos.
    assert body.position == pytest.approx(env.data.mocap_pos[0])
    assert raw.position == pytest.approx(env.data.mocap_pos[0])
    assert body.properties["xpos_mocap_body_0"] == pytest.approx(
        env.data.xpos[mocap_body_id][0]
    )
    assert raw.properties["mocap_pos_mocap_body_0"] == pytest.approx(1.0)

    body_sink = "entity[{}].position".format(body_entity_id)
    raw_sink = "entity[{}].position".format(raw_entity_id)
    xpos_sinks = adapter.field_materialization["xpos"].emitted_keys
    mocap_sinks = adapter.field_materialization["mocap_pos"].emitted_keys
    assert body_sink in xpos_sinks
    assert raw_sink not in xpos_sinks
    assert raw_sink in mocap_sinks
    assert body_sink not in mocap_sinks


def test_additional_joint_owner_uses_shared_materialization_and_metadata():
    class _AdditionalOwnerAdapter(MujocoAdapter):
        def _additional_joint_owners(self):
            return (
                MujocoJointOwner(
                    entity_id=self.name_to_id["joint_alias"],
                    owner_body_id=self._name2id("body", "tracked"),
                    joint_ids=(self._name2id("joint", "slide_joint"),),
                    owner_name="joint_alias",
                ),
            )

    env = _env()
    adapter = _AdditionalOwnerAdapter(
        env,
        body_rules=(
            BodyRule(
                match="tracked",
                cls="object",
                joint_scope="body",
            ),
        ),
        regions=(
            RegionSpec(
                name="joint_alias",
                source="fixed",
                center=(0.0, 0.0, 0.0),
                half_extents=(0.1, 0.1, 0.1),
            ),
        ),
    )

    observation = adapter.reset()
    alias_id = adapter.name_to_id["joint_alias"]
    key = "qpos_slide_joint"
    alias = observation.perceived_entities[alias_id]

    assert alias.properties[key] == pytest.approx(env.data.qpos[0])
    assert key in adapter.property_metadata(alias_id)
    assert (
        "entity[{}].properties.{}".format(alias_id, key)
        in adapter.field_materialization["qpos"].emitted_keys
    )
    assert set(alias.properties) == set(adapter.property_metadata(alias_id))


def test_raw_site_and_semantic_region_coexist_without_reclassification() -> None:
    adapter = _adapter(
        regions=(
            RegionSpec(
                name="shared",
                source="fixed",
                center=(0.0, 0.0, 0.0),
                half_extents=(0.1, 0.1, 0.1),
            ),
        )
    )
    observation = adapter.reset()

    region_id = adapter.name_to_id["shared"]
    site_id = adapter.name_to_id["site:shared"]
    assert region_id != site_id
    assert observation.perceived_entities[region_id].entity_class == "region"
    assert observation.perceived_entities[site_id].entity_class == "site"
    assert adapter.entity_labels[site_id] == "shared (MuJoCo site)"
    assert "site_size_shared_0" in observation.perceived_entities[
        site_id
    ].properties
    expected_types = {
        "shared": "sphere",
        "box_site": "box",
        "capsule_site": "capsule",
        "marker_site": "marker",
    }
    for source_name, type_name in expected_types.items():
        raw_id = adapter.name_to_id["site:{}".format(source_name)]
        metadata = adapter.property_metadata(raw_id)
        assert any(
            "site_type:{}".format(type_name) in item.tags
            for item in metadata.values()
        )


def test_rotated_geom_and_site_regions_use_world_axis_aligned_bounds() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <geom name="rotated_box" type="box"
                  pos="1 2 3" size="0.1 0.2 0.3"
                  euler="0 0 90"/>
            <site name="rotated_site" type="box"
                  pos="-1 -2 -3" size="0.01"
                  euler="0 0 90"/>
          </worldbody>
        </mujoco>
        """
    )
    adapter = MujocoAdapter(
        env,
        regions=(
            RegionSpec(
                name="geom_region",
                source="geom",
                source_name="rotated_box",
            ),
            RegionSpec(
                name="site_region",
                source="site",
                source_name="rotated_site",
                half_extents=(0.4, 0.1, 0.2),
            ),
        ),
    )
    observation = adapter.reset()

    geom = observation.perceived_entities[
        adapter.name_to_id["geom_region"]
    ]
    assert geom.properties["hw"] == pytest.approx(0.2)
    assert geom.properties["hh"] == pytest.approx(0.1)
    assert geom.properties["hd"] == pytest.approx(0.3)
    assert geom.properties["x0"] == pytest.approx(0.8)
    assert geom.properties["x1"] == pytest.approx(1.2)
    assert geom.properties["y0"] == pytest.approx(1.9)
    assert geom.properties["y1"] == pytest.approx(2.1)

    site = observation.perceived_entities[
        adapter.name_to_id["site_region"]
    ]
    assert site.properties["hw"] == pytest.approx(0.1)
    assert site.properties["hh"] == pytest.approx(0.4)
    assert site.properties["hd"] == pytest.approx(0.2)
    assert site.properties["x0"] == pytest.approx(-1.1)
    assert site.properties["x1"] == pytest.approx(-0.9)
    assert site.properties["y0"] == pytest.approx(-2.4)
    assert site.properties["y1"] == pytest.approx(-1.6)


def test_perception_gate_filters_observation_not_scene_applicability() -> None:
    adapter = _adapter(
        perception_gate=lambda entity, _env: (
            entity.entity_class != "object"
        )
    )
    observation = adapter.reset()

    assert adapter.name_to_id["tracked"] not in observation.perceived_entities
    status = adapter.field_materialization["xpos"]
    assert status.binding_present
    assert status.scene_applicable
    assert status.emitted_keys
    assert not any(
        "xpos_tracked_" in sink for sink in status.emitted_keys
    )
    assert status.owner_omissions
    assert "filtered from this observation" in status.reason


def test_unnamed_positive_owners_use_compiled_model_index_identity() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint type="slide"/>
              <geom type="box" size="0.1 0.1 0.1"/>
              <site type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    observation = adapter.reset()

    for field_name in ("qpos", "geom_xpos", "site_xpos"):
        status = adapter.field_materialization[field_name]
        assert status.scene_applicable
        assert not status.owner_omissions
    object_id = adapter.name_to_id["tracked"]
    assert (
        "qpos_specola_unnamed_joint_index_0"
        in observation.perceived_entities[object_id].properties
    )
    geom_id = adapter.name_to_id[
        "geom:specola_unnamed_geom_index_0"
    ]
    site_id = adapter.name_to_id[
        "site:specola_unnamed_site_index_0"
    ]
    assert "geom_xpos_specola_unnamed_geom_index_0_0" in (
        observation.perceived_entities[geom_id].properties
    )
    assert "site_xpos_specola_unnamed_site_index_0_0" in (
        observation.perceived_entities[site_id].properties
    )
    assert "identity:compiled_model_index" in (
        adapter.property_metadata(geom_id)[
            "geom_xpos_specola_unnamed_geom_index_0_0"
        ].tags
    )


def test_unnamed_subtree_body_uses_compiled_model_index_key() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <geom name="root_shape" size="0.1"/>
              <body>
                <joint name="child_joint" type="hinge"/>
                <geom name="child_shape" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    adapter = MujocoAdapter(
        env,
        body_rules=(
            BodyRule(
                "tracked", "object", joint_scope="subtree"
            ),
        ),
    )
    observation = adapter.reset()
    object_id = adapter.name_to_id["tracked"]
    properties = observation.perceived_entities[object_id].properties

    assert "qpos_child_joint" in properties
    assert "xpos_specola_unnamed_body_index_2_0" in properties
    xpos = adapter.field_materialization["xpos"]
    assert xpos.scene_applicable
    assert not xpos.owner_omissions


def test_unnamed_index_identity_cannot_collide_with_model_name() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <geom type="sphere" size="0.1"/>
            <geom name="specola_unnamed_geom_index_0"
                  type="sphere" size="0.1"/>
          </worldbody>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    adapter.reset()

    unnamed = (
        "geom:specola_unnamed_geom_index_0_reserved"
    )
    named = "geom:specola_unnamed_geom_index_0"
    assert unnamed in adapter.name_to_id
    assert named in adapter.name_to_id
    assert adapter.name_to_id[unnamed] != adapter.name_to_id[named]


def test_unnamed_actuator_equality_and_camera_are_index_addressable() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="slide" type="slide"/>
              <geom name="shape" type="box" size="0.1 0.1 0.1"/>
              <camera/>
              <light/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
          <equality>
            <weld body1="tracked"/>
          </equality>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    observation = adapter.reset()

    object_id = adapter.name_to_id["tracked"]
    object_properties = observation.perceived_entities[
        object_id
    ].properties
    assert "ctrl_specola_unnamed_actuator_index_0" in object_properties
    assert (
        "actuator_force_specola_unnamed_actuator_index_0"
        in object_properties
    )
    moment_key = (
        "actuator_moment_actuator_0_dof_0"
    )
    assert object_properties[moment_key] == pytest.approx(1.0)
    assert adapter.property_metadata(object_id)[moment_key].note == "1"

    equality_id = adapter.name_to_id[
        "equality:specola_unnamed_equality_index_0"
    ]
    assert (
        observation.perceived_entities[equality_id].properties[
            "eq_active_specola_unnamed_equality_index_0_active"
        ]
        == 1.0
    )

    camera_id = adapter.name_to_id[
        "camera:specola_unnamed_camera_index_0"
    ]
    camera = observation.perceived_entities[camera_id]
    assert camera.position is not None
    assert (
        "cam_xpos_specola_unnamed_camera_index_0_0"
        in camera.properties
    )
    assert (
        "cam_xmat_specola_unnamed_camera_index_0_0"
        in camera.properties
    )
    assert "identity:compiled_model_index" in (
        adapter.property_metadata(camera_id)[
            "cam_xpos_specola_unnamed_camera_index_0_0"
        ].tags
    )

    light_id = adapter.name_to_id[
        "light:specola_unnamed_light_index_0"
    ]
    light = observation.perceived_entities[light_id]
    assert (
        "light_xpos_specola_unnamed_light_index_0_0"
        in light.properties
    )
    assert (
        "light_xdir_specola_unnamed_light_index_0_0"
        in light.properties
    )
    assert "sensor_configuration:rendering" in (
        adapter.property_metadata(light_id)[
            "light_xpos_specola_unnamed_light_index_0_0"
        ].tags
    )


def test_unselected_unnamed_body_and_joint_are_not_owner_omissions() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="tracked_joint" type="slide"/>
              <geom name="tracked_shape" type="sphere" size="0.1"/>
            </body>
            <body>
              <joint type="hinge"/>
              <geom name="unselected_shape" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    adapter.reset()

    unselected_body_id = int(env.model.nbody) - 1
    unselected_joint_id = int(env.model.njnt) - 1
    assert all(
        owner != "body[{}]".format(unselected_body_id)
        for owner, _reason in adapter.field_materialization[
            "xpos"
        ].owner_omissions
    )
    assert all(
        owner != "joint[{}]".format(unselected_joint_id)
        for owner, _reason in adapter.field_materialization[
            "qpos"
        ].owner_omissions
    )


def test_ball_joint_axis_is_outside_the_declared_owner_population() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="ball_joint" type="ball"/>
              <geom name="shape" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    adapter.reset()

    assert adapter.field_materialization["xanchor"].scene_applicable
    xaxis = adapter.field_materialization["xaxis"]
    assert not xaxis.scene_applicable
    assert xaxis.owner_omissions == ()
    assert (
        xaxis.reason
        == "current reset model has no resolvable hinge/slide joint owner"
    )


def test_xaxis_declares_and_materializes_the_exact_joint_type_partition() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="free_owner"><freejoint name="free_joint"/>
          <geom type="sphere" size="0.01"/></body>
        <body name="ball_owner"><joint name="ball_joint" type="ball"/>
          <geom type="sphere" size="0.01"/></body>
        <body name="slide_owner"><joint name="slide_joint" type="slide"
          axis="1 0 0"/><geom type="sphere" size="0.01"/></body>
        <body name="hinge_owner"><joint name="hinge_joint" type="hinge"
          axis="0 1 0"/><geom type="sphere" size="0.01"/></body>
      </worldbody>
    </mujoco>
    """
    env = _env(xml)
    adapter = MujocoAdapter(
        env,
        body_rules=(
            BodyRule(
                match=lambda name: name.endswith("_owner"),
                cls="object",
                joint_scope="body",
            ),
        ),
    )

    expected_partition = {
        int(mujoco.mjtJoint.mjJNT_FREE): False,
        int(mujoco.mjtJoint.mjJNT_BALL): False,
        int(mujoco.mjtJoint.mjJNT_SLIDE): True,
        int(mujoco.mjtJoint.mjJNT_HINGE): True,
    }
    assert _XAXIS_JOINT_TYPE_APPLICABILITY == expected_partition
    assert set(int(kind) for kind in env.model.jnt_type) == set(
        expected_partition
    )

    adapter.reset()

    xaxis = adapter.field_materialization["xaxis"]
    assert xaxis.scene_applicable
    assert xaxis.owner_omissions == ()
    assert xaxis.plan_refusal is None
    assert len(xaxis.emitted_keys) == 6
    assert all(
        "xaxis_{}_joint_".format(kind) not in sink
        for kind in ("free", "ball")
        for sink in xaxis.emitted_keys
    )
    assert sum("xaxis_slide_joint_" in sink for sink in xaxis.emitted_keys) == 3
    assert sum("xaxis_hinge_joint_" in sink for sink in xaxis.emitted_keys) == 3


def test_xaxis_rejects_an_unknown_joint_type() -> None:
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    env.model.jnt_type[0] = 99

    with pytest.raises(
        RuntimeError,
        match="Unknown MuJoCo joint type 99 while classifying xaxis",
    ):
        adapter._xaxis_owner_triples()


def test_mixed_tendon_units_omit_only_the_unresolvable_owner() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="slide_joint" type="slide"/>
              <joint name="hinge_joint" type="hinge"/>
              <geom name="shape" type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="mixed">
              <joint joint="slide_joint" coef="1"/>
              <joint joint="hinge_joint" coef="1"/>
            </fixed>
            <fixed name="slide_only">
              <joint joint="slide_joint" coef="1"/>
              <joint joint="hinge_joint" coef="0"/>
            </fixed>
          </tendon>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    observation = adapter.reset()

    status = adapter.field_materialization["ten_length"]
    assert status.scene_applicable
    assert status.owner_omissions == ()
    valid_id = adapter.name_to_id["tendon:slide_only"]
    assert "ten_length_slide_only" in observation.perceived_entities[
        valid_id
    ].properties
    mixed_id = adapter.name_to_id["tendon:mixed"]
    mixed = observation.perceived_entities[mixed_id].properties
    assert "ten_length_mixed" in mixed
    mixed_meta = adapter.property_metadata(mixed_id)["ten_length_mixed"]
    assert mixed_meta.note is None
    assert {"semantics:opaque", "unit:unresolved"} <= set(
        mixed_meta.tags
    )


def test_stateful_actuator_units_materialize_or_report_owner_capability() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="slide_joint" type="slide"/>
              <geom name="shape" type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <actuator>
            <intvelocity name="integrated" joint="slide_joint"
                         kp="10" actrange="-1 1"/>
            <general name="unresolved" joint="slide_joint"
                     dyntype="filter" dynprm="0.1"/>
          </actuator>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    observation = adapter.reset()
    object_id = adapter.name_to_id["tracked"]
    properties = observation.perceived_entities[object_id].properties
    metadata = adapter.property_metadata(object_id)

    assert "act_integrated" in properties
    assert "act_dot_integrated" in properties
    assert metadata["act_integrated"].note == "m"
    assert metadata["act_dot_integrated"].note == "m/s"
    for field_name in ("act", "act_dot"):
        status = adapter.field_materialization[field_name]
        assert status.scene_applicable
        assert status.owner_omissions == ()
        key = "{}_unresolved".format(field_name)
        assert key in properties
        assert metadata[key].note is None
        assert {"semantics:opaque", "unit:unresolved"} <= set(
            metadata[key].tags
        )


def test_compiled_flex_uses_model_addresses_and_populated_edge_jacobian() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1"
                      spacing=".05 .05 .05" radius=".01" dim="2">
              <edge damping="1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """
    )
    env.data.qvel[0] = 0.2
    mujoco.mj_forward(env.model, env.data)
    adapter = MujocoAdapter(env, body_rules=())
    observation = adapter.reset()

    flex_id = adapter.name_to_id["flex:cloth"]
    properties = observation.perceived_entities[flex_id].properties
    vertex_start = int(env.model.flex_vertadr[0])
    edge_start = int(env.model.flex_edgeadr[0])
    vertex_count = int(env.model.flex_vertnum[0])
    edge_count = int(env.model.flex_edgenum[0])

    assert vertex_count == 9
    assert edge_count == 16
    assert env.data.flexedge_J.size > 0
    assert np.count_nonzero(env.data.flexedge_velocity) > 0
    for local_index in range(vertex_count):
        for axis in range(3):
            assert properties[
                f"flexvert_xpos_cloth_{local_index}_{axis}"
            ] == pytest.approx(
                env.data.flexvert_xpos[vertex_start + local_index][axis]
            )
    edge_jacobian = _dense_row_jacobian(
        env,
        "flexedge_J",
        int(env.model.nflexedge),
        int(env.model.nv),
    )
    for local_index in range(edge_count):
        assert properties[
            f"flexedge_length_cloth_{local_index}"
        ] == pytest.approx(
            env.data.flexedge_length[edge_start + local_index]
        )
        assert properties[
            f"flexedge_velocity_cloth_{local_index}"
        ] == pytest.approx(
            env.data.flexedge_velocity[edge_start + local_index]
        )
        for dof_id in range(env.model.nv):
            key = (
                f"flexedge_jacobian_cloth_edge_{local_index}_dof_{dof_id}"
            )
            assert properties[key] == pytest.approx(
                edge_jacobian[edge_start + local_index][dof_id]
            )
            assert adapter.property_metadata(flex_id)[key].note == "1"
    for field_name in (
        "flexedge_J",
        "flexvert_xpos",
        "flexedge_length",
        "flexedge_velocity",
    ):
        status = adapter.field_materialization[field_name]
        assert status.scene_applicable
        assert status.emitted_keys
        assert status.owner_omissions == ()


def test_row_jacobian_oracle_reads_both_mujoco_layouts() -> None:
    """Guard the oracle the Jacobian assertions depend on.

    MuJoCo stores row Jacobians dense on some models and sparse on others.
    Measured on mujoco 3.8.1, ``flexedge_J`` is sparse for every flex model
    checked, including under ``jacobian="dense"`` — the option does not
    govern that field — which is what made a dense-only oracle silently
    wrong. A helper that lost its dense branch would still pass every
    Jacobian test today and fail only on some future MuJoCo, so both
    branches are pinned here against one known matrix.
    """
    expected = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    dense_env = SimpleNamespace(
        data=SimpleNamespace(row_J=expected.copy()),
        model=SimpleNamespace(),
    )
    assert np.array_equal(
        _dense_row_jacobian(dense_env, "row_J", 2, 3), expected
    )

    sparse_env = SimpleNamespace(
        data=SimpleNamespace(row_J=np.array([1.0, 2.0, 3.0])),
        model=SimpleNamespace(
            row_J_rownnz=np.array([2, 1]),
            row_J_rowadr=np.array([0, 2]),
            row_J_colind=np.array([0, 2, 1]),
        ),
    )
    assert np.array_equal(
        _dense_row_jacobian(sparse_env, "row_J", 2, 3), expected
    )


def test_unrequested_flex_edge_jacobian_is_not_exposed_as_physical_zero() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1"
                      spacing=".05 .05 .05" radius=".01" dim="2"/>
          </worldbody>
        </mujoco>
        """
    )
    adapter = MujocoAdapter(env, body_rules=())
    observation = adapter.reset()

    flex_id = adapter.name_to_id["flex:cloth"]
    assert not any(
        key.startswith("flexedge_jacobian_")
        for key in observation.perceived_entities[flex_id].properties
    )
    status = adapter.field_materialization["flexedge_J"]
    assert not status.scene_applicable
    assert status.owner_omissions == (
        ("cloth", "MuJoCo did not request this flex's edge Jacobian"),
    )


def test_spatial_tendon_length_and_velocity_are_metres_and_metres_per_second() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <site name="anchor" pos="0 0 0"/>
            <body name="tracked" pos="0.5 0 0">
              <joint name="slide_joint" type="slide" axis="1 0 0"/>
              <geom name="shape" type="sphere" size="0.05"/>
              <site name="moving"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="rope">
              <site site="anchor"/>
              <site site="moving"/>
            </spatial>
          </tendon>
        </mujoco>
        """
    )
    env.data.qvel[0] = 0.2
    mujoco.mj_forward(env.model, env.data)
    adapter = _adapter(env)
    observation = adapter.reset()

    tendon_id = adapter.name_to_id["tendon:rope"]
    properties = observation.perceived_entities[tendon_id].properties
    metadata = adapter.property_metadata(tendon_id)
    assert properties["ten_length_rope"] == pytest.approx(
        env.data.ten_length[0]
    )
    assert properties["ten_velocity_rope"] == pytest.approx(
        env.data.ten_velocity[0]
    )
    assert properties["ten_velocity_rope"] == pytest.approx(0.2)
    assert metadata["ten_length_rope"].note == "m"
    assert metadata["ten_velocity_rope"].note == "m/s"
    jacobian_key = "tendon_jacobian_tendon_0_dof_0"
    tendon_jacobian = _dense_row_jacobian(
        env, "ten_J", int(env.model.ntendon), int(env.model.nv)
    )
    assert properties[jacobian_key] == pytest.approx(
        tendon_jacobian[0][0]
    )
    assert properties[jacobian_key] == pytest.approx(1.0)
    assert metadata[jacobian_key].note == "1"
    assert adapter.field_materialization["ten_length"].owner_omissions == ()
    assert adapter.field_materialization["ten_J"].owner_omissions == ()
    assert adapter.field_materialization["ten_velocity"].owner_omissions == ()


def test_multicomponent_user_activation_is_exposed_with_opaque_semantics() -> None:
    env = _env(
        """
        <mujoco>
          <worldbody>
            <body name="tracked">
              <joint name="hinge_joint" type="hinge"/>
              <geom name="shape" type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <actuator>
            <general name="plugin_shaped" joint="hinge_joint"
                     dyntype="user" actdim="3"/>
          </actuator>
        </mujoco>
        """
    )
    adapter = _adapter(env)
    observation = adapter.reset()
    object_id = adapter.name_to_id["tracked"]

    properties = observation.perceived_entities[object_id].properties
    metadata = adapter.property_metadata(object_id)
    expected = {
        "{}_plugin_shaped_{}".format(field_name, component)
        for field_name in ("act", "act_dot")
        for component in range(3)
    }
    assert expected <= set(properties)
    for key in expected:
        assert metadata[key].note is None
        assert {"semantics:opaque", "unit:unresolved"} <= set(
            metadata[key].tags
        )
    for field_name in ("act", "act_dot"):
        status = adapter.field_materialization[field_name]
        assert status.scene_applicable
        assert status.owner_omissions == ()


class _DeclaredAdapter(MujocoAdapter):
    omit_states = False

    def _declare_task_entities(self):
        return (
            DeclaredEntitySpec(
                declaration_key="goal",
                name="declared_goal",
                entity_class="goal",
                source_kind="task_declaration",
                source_name="goal_position",
                source_provenance="controlled fixture",
                tags=("task",),
            ),
        )

    def _declared_entity_states(
        self,
        registrations: Mapping[str, int],
    ):
        if self.omit_states:
            return {}
        return {
            "goal": EntityState(
                entity_id=registrations["goal"],
                entity_class="goal",
                position=None,
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={"goal_value": 0.25},
            )
        }

    def _declared_entity_property_metadata(self, declaration_key):
        assert declaration_key == "goal"
        return {
            "goal_value": PropertyMeta(
                group="declared goal value",
                note="1",
                tags=("task",),
            )
        }


def test_declared_entities_reconcile_state_and_metadata_every_observation() -> None:
    adapter = _DeclaredAdapter(
        _env(),
        body_rules=(BodyRule("tracked", "object"),),
    )
    first = adapter.reset()
    entity_id = adapter.name_to_id["declared:goal"]
    assert first.perceived_entities[entity_id].properties == {
        "goal_value": 0.25
    }
    assert set(adapter.property_metadata(entity_id)) == {"goal_value"}

    second = adapter.get_observation()
    assert second.perceived_entities[entity_id].track_age == 1


def test_declared_entity_count_mismatch_fails_loudly() -> None:
    adapter = _DeclaredAdapter(
        _env(),
        body_rules=(BodyRule("tracked", "object"),),
    )
    adapter.omit_states = True
    with pytest.raises(
        ValueError, match="Declared entity state reconciliation mismatch"
    ):
        adapter.reset()


class _CollidingDeclaredAdapter(_DeclaredAdapter):
    def _declare_task_entities(self):
        original = super()._declare_task_entities()[0]
        return (
            DeclaredEntitySpec(
                declaration_key=original.declaration_key,
                name="shared",
                entity_class=original.entity_class,
                source_kind=original.source_kind,
                source_name=original.source_name,
                source_provenance=original.source_provenance,
                tags=original.tags,
            ),
        )


def test_declared_entity_and_semantic_identity_coexist_by_namespace() -> None:
    adapter = _CollidingDeclaredAdapter(
        _env(),
        body_rules=(BodyRule("tracked", "object"),),
        regions=(
            RegionSpec(
                name="shared",
                source="fixed",
                center=(0.0, 0.0, 0.0),
                half_extents=(0.1, 0.1, 0.1),
            ),
        ),
    )
    observation = adapter.reset()
    assert adapter.name_to_id["shared"] != adapter.name_to_id[
        "declared:goal"
    ]
    declared_id = adapter.name_to_id["declared:goal"]
    assert adapter.entity_labels[declared_id].startswith("shared (")
    assert observation.perceived_entities[
        declared_id
    ].entity_class == "goal"
