"""Evaluate explicit native completion stages on detached MuJoCo data.

The old error interceptor owns a thread-local jump buffer. An incoming error
handler may denote an active interceptor, so that condition remains unresolved.
Warning handlers are preserved and messages are retained without invoking them.
Caller synchronization of the live model/data remains required. Observation
must start outside any native call or native log callback; an empty TLS handler
does not establish that the producer's logging recursion flag is clear.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import threading

from .mujoco_native import MujocoNativeGuard
from .mujoco_logging import MujocoStructuredLogging
from ..conformance.query_bindings import QueryBindingError
from ..conformance.native_images import PinnedNativeFile


# These are explicit semantic dependencies of the named completion APIs, not
# a producer census or permission to execute additional pipeline operations.
_COMPLETION = {
    "mj_kinematics": (),
    "mj_comPos": ("mj_kinematics",),
    "mj_comVel": ("mj_kinematics", "mj_comPos"),
    "mj_camlight": ("mj_kinematics", "mj_comPos"),
    "mj_rnePostConstraint": ("mj_kinematics", "mj_comPos", "mj_comVel"),
}
_OBSERVING_THREAD = threading.local()


@dataclass(frozen=True)
class NativeWarning:
    """One native C-string warning, captured before its lifetime ends."""

    message: bytes


class ScratchObservation:
    """Borrow detached query data only inside its observation context.

    Readers must copy their outputs before returning. They must not retain
    native data or array views beyond the context. Public adapter observations
    carry owned values and do not expose this internal borrowed data handle.
    """

    def __init__(self, source, data, completed, warnings, logs=(), stage_results=None):
        self.source = source
        self._data = data
        self.completed = completed
        self._warnings = warnings
        self._logs = logs
        self.completion_warnings = tuple(warnings)
        self.completion_logs = tuple(logs)
        self.stage_results = {} if stage_results is None else {
            field: dict(stages) for field, stages in stage_results.items()}

    @property
    def data(self):
        if self._data is None:
            raise QueryBindingError("native scratch borrow has expired")
        return self._data

    @property
    def warnings(self):
        return tuple(self._warnings)

    @property
    def logs(self):
        return tuple(self._logs)


class MujocoScratchSession:
    """Own query scratch data and preserve native callback slots per interval."""

    def __init__(self, model, data, guard=None):
        mujoco = import_module("mujoco")

        self._mujoco = mujoco
        self._model = model
        self._guard = guard if guard is not None else MujocoNativeGuard(mujoco)
        self._owns_guard = guard is None
        self._scratch = None
        self._lock = threading.Lock()
        self._closed = False
        self._stack_header = None
        try:
            if mujoco.__version__ == "3.10.0":
                path = Path(__file__).parent.parent / "data" / "mujoco-stack-3.10.0.json"
                self._stack_profile = json.loads(path.read_text())
                field = self._stack_profile["field"]
                native = field["native"]
                self._threadlock_offset = native["offset_bits"] // 8
                if (self._stack_profile["version"] != mujoco.__version__
                        or field["id"] != "c:@S@mjData_@FI@threadlock"
                        or native["bitfield"] or native["offset_bits"] % 8
                        or native["type"]["canonical_kind"] != "BOOL"
                        or native["type"]["size"] != 1
                        or not 0 <= self._threadlock_offset < self._stack_profile["record_size"]):
                    raise QueryBindingError("unresolved native stack flag layout")
                self._stack_header = PinnedNativeFile(self._guard._distribution.locate_file(
                    "mujoco/include/mujoco/mjdata.h"), field["declaration"]["sha256"])
                self._stack_header.__enter__()
            self._validate_subject(data)
            self._logging = None
            if mujoco.__version__ == "3.10.0":
                self._logging = MujocoStructuredLogging(mujoco, model, self._guard)
            elif mujoco.__version__ in ("3.2.3", "3.3.0", "3.3.1", "3.8.1"):
                getter = ctypes.CFUNCTYPE(ctypes.c_void_p)
                setter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
                self._get_error = getter(self._guard._function_address("_mjPRIVATE__get_tls_error_fn"))
                self._set_error = setter(self._guard._function_address("_mjPRIVATE__set_tls_error_fn"))
                self._get_warning = getter(self._guard._function_address("_mjPRIVATE__get_tls_warning_fn"))
                self._set_warning = setter(self._guard._function_address("_mjPRIVATE__set_tls_warning_fn"))
                self._warning_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
            else:
                raise QueryBindingError("unresolved MuJoCo scratch logging source version")
            self._copy = type(data).__dict__["__copy__"]
            self._copy_data = (self._guard.function("mj_copyData")
                               if "mj_copyData" in vars(mujoco._functions) else None)
            self._completion = {name: self._guard.function(name) for name in _COMPLETION}
            self._guard.check()
        except BaseException:
            if self._stack_header is not None:
                self._stack_header.close()
            if self._owns_guard:
                self._guard.close()
            raise

    def _validate_subject(self, data):
        self._guard.check()
        if type(self._model) is not self._mujoco.MjModel or type(data) is not self._mujoco.MjData:
            raise QueryBindingError("scratch needs exact native MuJoCo model and data types")
        if data.model is not self._model:
            raise QueryBindingError("scratch data model identity differs")
        if int(self._model.nplugin):
            raise QueryBindingError("unresolved scratch plugin initialization and copy effects")
        if int(data.threadpool):
            raise QueryBindingError("unresolved scratch shared native thread pool")
        if self._stack_header is not None:
            self._stack_header.validate()
            address = data._address
            if type(address) is not int or address <= 0 or address % self._stack_profile["record_alignment"]:
                raise QueryBindingError("native data address does not satisfy its header declaration")
            # The exact native wrapper owns this struct. Its pinned descriptor
            # supplies the base; the independent header supplies the field
            # offset. No arbitrary pointer or nested allocation is followed.
            if ctypes.string_at(address + self._threadlock_offset, 1) != b"\x00":
                raise QueryBindingError("unresolved scratch native threadlock stack policy")
        if int(data.pstack):
            raise QueryBindingError("cannot copy native data while its stack is in use")
        for name in ("mju_user_malloc", "mju_user_free"):
            if self._guard.read_global_pointer(name):
                raise QueryBindingError("unresolved scratch allocation callback: " + name)

    @contextmanager
    def _capture_legacy_warnings(self):
        old_error, old_warning = self._get_error(), self._get_warning()
        if old_error:
            raise QueryBindingError("unresolved active native TLS error handler")
        warnings, recorder_errors = [], []

        def record(message):
            try:
                if type(message) is not bytes:
                    raise QueryBindingError("native warning did not contain a C string")
                warnings.append(NativeWarning(message))
            except BaseException as error:
                recorder_errors.append(error)

        recorder = self._warning_type(record)
        self._set_warning(ctypes.cast(recorder, ctypes.c_void_p).value)
        try:
            yield warnings, ()
            if recorder_errors:
                raise QueryBindingError("native warning recorder failed") from recorder_errors[0]
        finally:
            self._set_error(old_error)
            self._set_warning(old_warning)
            if self._get_error() != old_error or self._get_warning() != old_warning:
                raise QueryBindingError("native TLS callback restoration failed")
            self._guard.check()

    @contextmanager
    def observe(self, data, *, completion=()):
        """Yield detached data after the explicitly requested dependency closure.

        No simulator step, control callback, reward or verdict is requested.
        Completion uses stored copied inputs, including stored acceleration;
        it does not establish that those inputs describe one physical stage.
        """
        if self._closed:
            raise QueryBindingError("MuJoCo scratch session is closed")
        if type(completion) is not tuple:
            raise QueryBindingError("native completion requires an explicit ordered tuple")
        done = set()
        for name in completion:
            if (type(name) is not str or name not in _COMPLETION or name in done
                    or not set(_COMPLETION[name]).issubset(done)):
                raise QueryBindingError("native completion has an unsupported operation or dependency order")
            done.add(name)
        if getattr(_OBSERVING_THREAD, "active", False) or not self._lock.acquire(blocking=False):
            raise QueryBindingError("reentrant or concurrent native scratch observation is unresolved")
        _OBSERVING_THREAD.active = True
        try:
            self._validate_subject(data)
            capture = (self._logging.capture(data, completion) if self._logging is not None
                       else self._capture_legacy_warnings())
            with capture as (warnings, logs):
                if self._scratch is None or self._copy_data is None:
                    self._scratch = self._copy(data)
                else:
                    self._copy_data(self._scratch, self._model, data)
                for name in completion:
                    self._completion[name](self._model, self._scratch)
                snapshot = ScratchObservation(data, self._scratch, completion, warnings, logs)
                try:
                    yield snapshot
                finally:
                    snapshot._data = None
        finally:
            _OBSERVING_THREAD.active = False
            self._lock.release()

    def close(self):
        """Retire scratch before releasing its producer image ownership."""
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise QueryBindingError("cannot close an active native scratch observation")
        try:
            self._guard.check()
            if self._scratch is not None:
                self._validate_subject(self._scratch)
            self._scratch = None
            self._closed = True
            if self._stack_header is not None:
                self._stack_header.close()
            if self._owns_guard:
                self._guard.close()
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise QueryBindingError("MuJoCo scratch session is closed")
        return self

    def __exit__(self, *exception):
        self.close()
