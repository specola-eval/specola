"""Capture MuJoCo 3.10 logs without initializing or invoking global sinks."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from ..conformance.query_bindings import QueryBindingError


class _MjLogMessage(ctypes.Structure):
    """The pinned mjtype.h declaration; native fixtures verify its full ABI."""

    _fields_ = [("level", ctypes.c_int), ("topic", ctypes.c_int),
                ("subject", ctypes.c_char * 1024), ("body", ctypes.c_char_p),
                ("func", ctypes.c_char_p), ("file", ctypes.c_char_p),
                ("line", ctypes.c_int), ("timestamp", ctypes.c_bool)]


@dataclass(frozen=True)
class NativeLog:
    """Owned native log semantics, with null and empty strings distinguished."""

    level: int
    topic: int
    subject: bytes
    body: Optional[bytes]
    func: Optional[bytes]
    file: Optional[bytes]
    line: int
    timestamp: bool

    @property
    def message(self):
        """Expose subject bytes through the existing warning interface."""
        return self.subject


class MujocoStructuredLogging:
    """Own one structured TLS recorder inside a synchronized native interval."""

    def __init__(self, mujoco, model, guard):
        self._mujoco, self._model, self._guard = mujoco, model, guard
        path = Path(__file__).parent.parent / "data" / "mujoco-logging-3.10.0.json"
        self._profile = json.loads(path.read_text())
        if mujoco.__version__ != self._profile["version"]:
            raise QueryBindingError("native structured logging source version differs")
        if ctypes.sizeof(_MjLogMessage) != 1064 or ctypes.alignment(_MjLogMessage) != 8:
            raise QueryBindingError("native structured logging ABI is unresolved")
        self._set_tls = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(
            guard._function_address("_mjPRIVATE_setTlsLogHandler"))
        self._recorder_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(_MjLogMessage))
        self._warning_level = int(mujoco.mjtLogLevel.mjLOG_WARNING)
        self._error_level = int(mujoco.mjtLogLevel.mjLOG_ERROR)
        self._sleep_flag = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        self._sleep_topic_bit = 1 << (int(mujoco.mjtLogTopic.mjTOPIC_SLEEP) - 1)

    def _state(self):
        state = self._guard._read_declared_local_objects(self._profile)
        state["legacy_warning"] = self._guard.read_global_pointer("mju_user_warning")
        state["legacy_error"] = self._guard.read_global_pointer("mju_user_error")
        return state

    def _validate_branch(self, data, completion, state):
        if not completion or not int(self._model.opt.enableflags) & self._sleep_flag:
            return
        count = int(self._model.ntree)
        if data.tree_asleep.shape != (count,):
            raise QueryBindingError("native sleeping-tree population differs from model declaration")
        if all(int(value) < 0 for value in data.tree_asleep):
            return
        if state["env_checked"] != b"\x01":
            raise QueryBindingError("unresolved native sleep logging would initialize global configuration")
        topics = int.from_bytes(state["log_config"][1028:1032], "little", signed=True)
        if topics & self._sleep_topic_bit:
            raise QueryBindingError("unresolved native sleep debug formatter bounds")

    @contextmanager
    def capture(self, data, completion):
        """Retain logs and restore TLS while leaving global configuration alone."""
        before = self._state()
        self._validate_branch(data, completion, before)
        warnings, logs, failures = [], [], []

        def record(pointer):
            try:
                if not pointer:
                    raise QueryBindingError("native log message pointer is null")
                message = pointer.contents
                subject = ctypes.string_at(ctypes.addressof(message) + _MjLogMessage.subject.offset, 1024)
                if b"\0" not in subject:
                    raise QueryBindingError("native log subject has no declared string terminator")
                owned = NativeLog(message.level, message.topic, subject.split(b"\0", 1)[0],
                                  message.body, message.func, message.file, message.line, message.timestamp)
                logs.append(owned)
                if owned.level == self._error_level:
                    raise QueryBindingError("native error escaped the maintained error interceptor")
                if owned.level == self._warning_level:
                    warnings.append(owned)
            except BaseException as error:
                failures.append(error)

        recorder = self._recorder_type(record)
        address = ctypes.cast(recorder, ctypes.c_void_p).value
        previous = self._set_tls(address)
        if previous:
            replaced = self._set_tls(previous)
            if replaced != address:
                raise QueryBindingError("native structured TLS restoration failed before refusal")
            raise QueryBindingError("unresolved active native TLS log handler")
        try:
            yield warnings, logs
            if failures:
                raise QueryBindingError("native structured log recorder failed") from failures[0]
        finally:
            replaced = self._set_tls(previous)
            current = self._set_tls(previous)
            if replaced != address or current != previous:
                raise QueryBindingError("native structured TLS restoration failed")
            if self._state() != before:
                raise QueryBindingError("native global logging state changed during observation")
