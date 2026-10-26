"""Metadata address reconciliation; native fidelity uses separate fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from specola.adapters.mujoco_query_metadata import MujocoQueryMetadata
from specola.conformance.query_bindings import QueryBindingError


def _subject():
    # A bounded mixed-joint declaration fixture. Expected column conventions
    # below are source-derived test inputs, not a producer/API census.
    model = SimpleNamespace(
        nv=11, njnt=4, nbody=5, ngeom=0, nsite=0, ncam=0, nnames=6,
        names=b"\0free\0", name_bodyadr=np.zeros(5, dtype=np.int32),
        name_jntadr=np.array([1, 0, 0, 0], dtype=np.int32),
        name_geomadr=np.empty(0, dtype=np.int32), name_siteadr=np.empty(0, dtype=np.int32),
        name_camadr=np.empty(0, dtype=np.int32),
        dof_jntid=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3], dtype=np.int32),
        dof_bodyid=np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32),
        jnt_type=np.array([0, 1, 2, 3], dtype=np.int32),
        jnt_dofadr=np.array([0, 6, 9, 10], dtype=np.int32),
        jnt_qposadr=np.array([0, 7, 11, 12], dtype=np.int32),
        jnt_bodyid=np.array([1, 2, 3, 4], dtype=np.int32),
        jnt_axis=np.array([[0., 0., 1.]] * 4),
    )
    producer = SimpleNamespace(__version__="3.8.1", mjtJoint=SimpleNamespace(
        mjJNT_FREE=0, mjJNT_BALL=1, mjJNT_SLIDE=2, mjJNT_HINGE=3))
    return model, producer


def test_native_qvel_columns_keep_joint_names_frames_and_component_units():
    model, producer = _subject()
    metadata = MujocoQueryMetadata(model, producer)
    columns = metadata.value.restore()["columns"]
    assert [c["column"] for c in columns] == list(range(11))
    assert [c["joint_component"] for c in columns] == [0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 0]
    assert [c["component_frame"] for c in columns] == ["world"] * 3 + ["child_body"] * 6 + ["native_joint_axis"] * 2
    assert [c["coordinate_dimension"] for c in columns] == ["L"] * 3 + ["1"] * 6 + ["L", "1"]
    assert [c["axis_component"] for c in columns] == [0, 1, 2] * 3 + [None, None]
    assert metadata.owner_name("mjOBJ_XBODY", 0) is None
    assert columns[0]["joint_name"] == b"free" and columns[-1]["joint_name"] is None
    with pytest.raises(ValueError, match="read-only"):
        columns[0]["native_joint_axis"][2] = 9
    model.jnt_axis[0, 2] = 9
    assert metadata.value.restore()["columns"][0]["native_joint_axis"][2] == 1


def test_public_metadata_copies_cannot_change_names_or_column_conventions():
    model, producer = _subject()
    metadata = MujocoQueryMetadata(model, producer)
    columns, names = metadata.columns, metadata.owner_names
    columns[0]["component_frame"] = "wrong"
    names["joint"] = (b"wrong",)
    assert metadata.columns[0]["component_frame"] == "world"
    assert metadata.owner_name("joint", 0) == b"free"
    metadata.validate()


@pytest.mark.parametrize("index", (-1, 5, 1.0, True))
def test_owner_name_keeps_native_integer_bounds(index):
    model, producer = _subject()
    with pytest.raises(QueryBindingError, match="out of bounds"):
        MujocoQueryMetadata(model, producer).owner_name("body", index)


@pytest.mark.parametrize("field,index,value", (("jnt_dofadr", 1, 7), ("dof_jntid", 6, 0),
                                             ("dof_bodyid", 0, 2), ("jnt_type", 3, 100),
                                             ("name_jntadr", 1, 100)))
def test_inconsistent_or_omitted_native_addresses_fail(field, index, value):
    model, producer = _subject()
    getattr(model, field)[index] = value
    with pytest.raises(QueryBindingError):
        MujocoQueryMetadata(model, producer)


@pytest.mark.parametrize("field,index,value", (("jnt_axis", (1, 0), 0.5), ("name_jntadr", 1, 1),
                                             ("dof_jntid", 0, 1)))
def test_same_dimensions_do_not_hide_changed_metadata(field, index, value):
    model, producer = _subject()
    metadata = MujocoQueryMetadata(model, producer)
    getattr(model, field)[index] = value
    with pytest.raises(QueryBindingError, match="metadata changed"):
        metadata.validate()


def test_zero_dof_domain_retains_empty_columns_and_unnamed_world():
    model, producer = _subject()
    model.nv = model.njnt = 0
    for field in ("dof_jntid", "dof_bodyid", "jnt_type", "jnt_dofadr", "jnt_qposadr", "jnt_bodyid", "name_jntadr"):
        setattr(model, field, np.empty(0, dtype=np.int32))
    model.jnt_axis = np.empty((0, 3))
    metadata = MujocoQueryMetadata(model, producer)
    assert metadata.columns == ()
    assert metadata.owner_name("body", 0) is None
