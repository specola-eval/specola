"""Complete legacy scalar projections on detached native MuJoCo data.

Stored native extraction retains the source data. These completed fields are
separate projections of that stored state, including its existing qacc and
constraint forces. They do not represent a new dynamics solution.
"""

from contextlib import contextmanager
import ctypes

import numpy as np

from .mujoco_scratch import ScratchObservation
from .mujoco_completed_fields import MujocoCompletedFields
from .mujoco_timer import MujocoTimerGuard
from ..conformance.query_bindings import QueryBindingError


class MujocoProjectionData:
    """Borrow native projection data with the existing wrapper pose aliases.

    Robosuite's body_xpos/body_xmat spellings refer to native xpos/xmat.
    This internal view retains that established lossless alias during scalar
    extraction. It must not escape the enclosing scratch observation.
    """

    def __init__(self, snapshot):
        self._snapshot = snapshot

    @property
    def _data(self):
        return self._snapshot.data

    @property
    def body_xpos(self):
        return self._snapshot.data.xpos

    @property
    def body_xmat(self):
        return self._snapshot.data.xmat

    def __getattr__(self, name):
        return getattr(self._snapshot.data, name)


class MujocoProjection:
    """Share existing scratch and effect guards for scalar completion stages."""

    def __init__(self, model, mujoco, session, guard):
        self._model = model
        self._mujoco = mujoco
        self._session = session
        self._guard = guard
        self._timer = MujocoTimerGuard(guard)
        self._closed = False
        self._functions = {}
        try:
            self._completed_fields = MujocoCompletedFields(model, guard)
            declared = self._timer.policy["completion"]["stage_sources"]
            for name in ("mj_flex", "mj_tendon", "mj_comVel", "mj_subtreeVel", "mj_energyPos", "mj_energyVel"):
                if not declared.get(name):
                    raise QueryBindingError("projection stage has no native source declaration: " + name)
                self._functions[name] = guard.function(name)
            self._functions["mj_rnePostConstraint"] = guard.function("mj_rnePostConstraint")
            self._mass = tuple(stage.split(" ", 1)[0] for stage in self._timer.policy["completion"]["mass_preparation"])
            for name in self._mass:
                if name == "mj_tendonArmature":
                    # The exact 3.3.1 internal-header MJAPI signature is
                    # void(const mjModel*, mjData*); it has no Python binding.
                    self._armature = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
                        guard._function_address(name))
                else:
                    self._functions[name] = guard.function(name)
        except BaseException:
            self.close()
            raise

    def _energy_stages(self):
        model, mujoco = self._model, self._mujoco
        version = self._timer.policy["version"]
        energy = (version != "3.2.3" or
                  int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ENERGY))
        spring_flag = getattr(mujoco.mjtDisableBit, self._timer.policy["completion"]["spring_gate"])
        spring = energy and not int(model.opt.disableflags) & int(spring_flag)
        flex = False
        if spring:
            for index in range(int(model.nflex)):
                if (bool(model.flex_rigid[index]) or not model.flex_edgestiffness[index]
                        or int(model.flex_dim[index]) > 1):
                    continue
                start, count = int(model.flex_edgeadr[index]), int(model.flex_edgenum[index])
                if not 0 <= start <= start + count <= int(model.nflexedge):
                    raise QueryBindingError("native flex energy edge interval is out of bounds")
                if np.any(model.flexedge_rigid[start:start + count] == 0):
                    flex = True
        tendon = int(model.ntendon) > 0 and (spring or (energy and version not in ("3.2.3", "3.3.0")))
        stages = []
        if flex:
            stages.append("mj_flex")
        if tendon:
            stages.append("mj_tendon")
        if energy:
            stages.extend(self._mass)
        return tuple(stages)

    @contextmanager
    def observe(self, data):
        """Yield completed scalar data under the caller's outer fenv scope."""
        if self._closed:
            raise QueryBindingError("native scalar projection is closed")
        self._session._validate_subject(data)
        energy_stages = self._energy_stages()
        for name, function in self._functions.items():
            if self._guard.function(name) is not function:
                raise QueryBindingError("native scalar completion reader changed")
        # Kinematics is included in the enclosing logging policy so wake paths
        # cannot evade the 3.10 lazy-initialization and debug-topic guards.
        initial = ("mj_kinematics", "mj_comPos", "mj_camlight")
        later = energy_stages + ("mj_comVel", "mj_subtreeVel", "mj_rnePostConstraint", "mj_energyPos", "mj_energyVel")
        with self._timer.preserve(), self._session.observe(data, completion=initial) as borrowed:
            stage_results = {}
            for index, name in enumerate(later):
                warning_start, log_start = len(borrowed.warnings), len(borrowed.logs)
                self._call(name, borrowed.data)
                for field, value in self._completed_fields.capture_stage(
                        name, borrowed, initial + later[:index + 1], warning_start, log_start):
                    stage_results.setdefault(field, {})[name] = value
            snapshot = ScratchObservation(data, borrowed.data, initial + later,
                                          borrowed._warnings, borrowed._logs, stage_results)
            try:
                yield snapshot
            finally:
                snapshot._data = None

    def _call(self, name, data):
        if name == "mj_tendonArmature":
            self._armature(self._model._address, data._address)
        else:
            self._functions[name](self._model, data)

    def prepare_mass(self, data):
        """Run this build's mass matrix stages (``crb`` and ``qM``) on ``data``.

        ``mj_crb`` on 3.3.0, ``mj_crb`` and ``mj_tendonArmature`` on 3.3.1,
        ``mj_makeM`` on 3.8.1 and 3.10.0: what the forward pass runs between
        kinematics and collision detection (``mass_preparation`` of the
        completion policy).
        """
        for name in self._mass:
            self._call(name, data)

    def object_linear_velocity(self, snapshot, owner_type, owner_id):
        """Project world-frame native linear velocity at the chosen origin."""
        if snapshot.data is not self._session._scratch:
            raise QueryBindingError("native velocity projection requires active scratch data")
        result = np.empty(6, dtype=snapshot.data.qpos.dtype)
        self._guard.function("mj_objectVelocity")(
            self._model, snapshot.data, owner_type, owner_id, result, 0)
        return result[3:].copy()

    def capture(self, snapshot, configuration_identity):
        """Retain completed arrays through the existing typed structured sink."""
        if self._closed or snapshot.data is not self._session._scratch:
            raise QueryBindingError("completed field capture requires an active projection")
        return self._completed_fields.capture(snapshot, configuration_identity)

    def close(self):
        """Retire timer custody; the enclosing adapter owns scratch and fenv."""
        if not self._closed:
            self._timer.close()
            self._closed = True
