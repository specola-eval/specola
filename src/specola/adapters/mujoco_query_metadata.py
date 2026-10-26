"""Own native joint, tangent-column and owner-name metadata for query replay."""

import json
from pathlib import Path

import numpy as np

from ..conformance.query_bindings import QueryBindingError
from ..native_values import NativeValue


# These are semantic joins from the versioned native name-dispatch source.
# Counts and address arrays belong to the model, including unnamed owners.
_NAME_DOMAINS = {
    "body": ("nbody", "name_bodyadr"),
    "joint": ("njnt", "name_jntadr"),
    "geom": ("ngeom", "name_geomadr"),
    "site": ("nsite", "name_siteadr"),
    "camera": ("ncam", "name_camadr"),
}


class MujocoQueryMetadata:
    """Reconcile native tangent columns against independent joint declarations.

    The owner data is captured once per metadata identity and linked by digest.
    It describes the query's model addresses; it is not the full model census.
    The enclosing adapter retains the broader model and observation identities.
    """

    def __init__(self, model, mujoco):
        path = Path(__file__).parent.parent / "data" / ("mujoco-query-metadata-" + mujoco.__version__ + ".json")
        if not path.is_file():
            raise QueryBindingError("unresolved native query metadata version")
        self._profile = json.loads(path.read_text())
        self._model = model
        self._fields = tuple(name for name in self._profile["source"]["field_declarations"]
                             if not name.startswith("mjData."))
        self.native = self._capture_model()
        self.identity = self.native.sha256
        self._values = self.native.restore()
        self._branches = {}
        for name, branch in self._profile["joint_branches"].items():
            if int(getattr(mujoco.mjtJoint, name)) != branch["enum_value"]:
                raise QueryBindingError("native joint enum and metadata source differ")
            self._branches[branch["enum_value"]] = dict(branch, name=name)
        self._owner_names = self._names()
        columns = self._columns()
        self.value = NativeValue.capture({
            "native_model_fields": self._values,
            "owner_names": self._owner_names,
            "columns": columns,
            "source_version": mujoco.__version__,
            "source_artifact": self._profile["source_artifact"],
            "unit_system": "consistent native length L, mass M and time T; radians for angular presentation",
        })

    @property
    def columns(self):
        """Return a detached tangent-column table from the owned record."""
        return self.value.restore()["columns"]

    @property
    def owner_names(self):
        """Return detached name domains while retaining raw name bytes."""
        return self.value.restore()["owner_names"]

    def _capture_model(self):
        return NativeValue.capture({name: getattr(self._model, name) for name in self._fields})

    def validate(self):
        """Refuse changed names, joint axes or native address mappings."""
        if self._capture_model().canonical_json != self.native.canonical_json:
            raise QueryBindingError("native query metadata changed; reset required")

    def _array(self, name, shape):
        value = self._values[name]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise QueryBindingError("native query metadata shape differs: " + name)
        return value

    def _names(self):
        raw = self._values["names"]
        if type(raw) is not bytes or len(raw) != int(self._values["nnames"]):
            raise QueryBindingError("native name buffer declaration differs")
        domains = {}
        for kind, (count_field, address_field) in _NAME_DOMAINS.items():
            count = int(self._values[count_field])
            addresses = self._array(address_field, (count,))
            names = []
            for address in addresses:
                address = int(address)
                if not 0 <= address < len(raw):
                    raise QueryBindingError("native owner name address is out of bounds")
                end = raw.find(b"\0", address)
                if end < 0:
                    raise QueryBindingError("native owner name is not terminated")
                names.append(raw[address:end] or None)
            domains[kind] = tuple(names)
        return domains

    def _columns(self):
        nv, njnt, nbody = (int(self._values[name]) for name in ("nv", "njnt", "nbody"))
        joints = self._array("dof_jntid", (nv,))
        dof_bodies = self._array("dof_bodyid", (nv,))
        types = self._array("jnt_type", (njnt,))
        starts = self._array("jnt_dofadr", (njnt,))
        bodies = self._array("jnt_bodyid", (njnt,))
        axes = self._array("jnt_axis", (njnt, 3))
        columns, next_column = [], 0
        for joint in range(njnt):
            branch = self._branches.get(int(types[joint]))
            if branch is None:
                raise QueryBindingError("native joint type has no source-backed DOF semantics")
            start, extent, body = int(starts[joint]), branch["native_dof_extent"], int(bodies[joint])
            end = int(starts[joint + 1]) if joint + 1 < njnt else nv
            if start != next_column or end - start != extent or not 0 <= body < nbody:
                raise QueryBindingError("native joint intervals do not cover the DOF domain")
            for column in range(start, end):
                if int(joints[column]) != joint or int(dof_bodies[column]) != body:
                    raise QueryBindingError("native DOF owner differs from its declared joint interval")
                component = column - start
                kind = branch["name"]
                translation = kind == "mjJNT_SLIDE" or (kind == "mjJNT_FREE" and component < 3)
                dimension = "L" if translation else "1"
                if kind in ("mjJNT_FREE", "mjJNT_BALL"):
                    axis = component if translation or kind == "mjJNT_BALL" else component - 3
                    frame = "world" if translation else "child_body"
                else:
                    axis, frame = None, "native_joint_axis"
                columns.append({
                    "column": column, "joint_id": joint, "joint_type": int(types[joint]),
                    "joint_type_name": kind, "joint_component": component, "body_id": body,
                    "joint_name": self._owner_names["joint"][joint], "coordinate_dimension": dimension,
                    "motion": "translation" if translation else "rotation", "axis_component": axis,
                    "component_frame": frame, "native_joint_axis": axes[joint],
                    "qvel_unit": "L/T" if translation else "rad/T",
                    "qacc_unit": "L/T^2" if translation else "rad/T^2",
                    "jacp_unit": "L/" + dimension, "jacr_unit": "1/" + dimension,
                    "angmom_unit": "M L^2/" + dimension,
                })
            next_column = end
        if next_column != nv or len(columns) != nv:
            raise QueryBindingError("native joint intervals omit a DOF")
        return tuple(columns)

    def owner_name(self, kind, index):
        """Retain raw name bytes or absence without changing native identity."""
        if kind.startswith("mjOBJ_"):
            kind = kind[len("mjOBJ_"):].lower()
        if kind == "xbody":
            kind = "body"
        names = self._owner_names.get(kind)
        if names is None or type(index) is not int or not 0 <= index < len(names):
            raise QueryBindingError("native query metadata owner is absent or out of bounds")
        return names[index]

    def completed_axes(self, snapshot):
        """Retain the actual completed joint axes and body orientation bases."""
        return NativeValue.capture({"xaxis": snapshot.data.xaxis, "xmat": snapshot.data.xmat,
                                    "completed": snapshot.completed, "model_metadata": self.value.sha256})
