"""Ordered, backend-neutral batching over isolated scalar monitor engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from .capabilities import AdapterCapabilities
from .components import ComponentCatalog
from .engine import MonitorEngine
from .results import EngineResult


@dataclass(frozen=True)
class BatchEpisodeContext:
    """Scalar episode-compilation arguments retained for one batch slot."""

    dt: float
    capabilities: Optional[AdapterCapabilities] = None
    catalog: Optional[ComponentCatalog] = None
    require_confirmed: bool = True
    skip_unsatisfiable: bool = False


class BatchStepError(RuntimeError):
    """A reset or update failure associated with one environment slot."""

    def __init__(self, index: int, operation: str, cause: Exception) -> None:
        self.index = index
        self.operation = operation
        self.cause = cause
        super().__init__(
            "Batched monitoring failed for environment index "
            f"{index} during {operation}: {type(cause).__name__}: {cause}"
        )


class BatchedMonitorEngine:
    """A fixed ordered set of completely independent monitor engines.

    This implementation deliberately performs the same scalar work as an
    ordered user-side loop. It establishes the API and equivalence boundary;
    it does not claim vectorized execution or acceleration.

    Instances are mutable and are not safe for concurrent ``step_batch``
    calls.
    """

    def __init__(
        self,
        prototype: MonitorEngine,
        contexts: Sequence[BatchEpisodeContext],
    ) -> None:
        self._prototype = prototype
        ordered_contexts = tuple(contexts)
        for index, context in enumerate(ordered_contexts):
            if not isinstance(context, BatchEpisodeContext):
                raise TypeError(
                    f"contexts[{index}] must be BatchEpisodeContext, got "
                    f"{type(context).__module__}."
                    f"{type(context).__qualname__}."
                )
        self._contexts = list(ordered_contexts)
        self._engines: list[Optional[MonitorEngine]] = [
            None for _ in self._contexts
        ]

    @classmethod
    def load(
        cls,
        spec_file: str,
        contexts: Sequence[BatchEpisodeContext],
    ) -> "BatchedMonitorEngine":
        """Parse one spec file and create an unstarted fixed-size batch."""
        return cls(MonitorEngine.load(spec_file), contexts)

    @property
    def batch_size(self) -> int:
        return len(self._contexts)

    def engine(self, index: int) -> MonitorEngine:
        """Return one started slot engine as a power-user escape hatch."""
        if index < 0 or index >= self.batch_size:
            raise IndexError(
                f"Environment index {index} is outside batch size "
                f"{self.batch_size}."
            )
        engine = self._engines[index]
        if engine is None:
            raise RuntimeError(
                f"Environment slot {index} has not started; pass "
                "reset_mask[index]=True with its first observation."
            )
        return engine

    def step_batch(
        self,
        observations: Sequence[Any],
        reset_mask: Sequence[bool],
        reset_contexts: Optional[
            Sequence[Optional[BatchEpisodeContext]]
        ] = None,
    ) -> Tuple[EngineResult, ...]:
        """Update every slot in order, resetting flagged slots first.

        Length and reset-flag validation is atomic. Execution thereafter has
        the same fail-stop progress semantics as an increasing-index scalar
        loop: earlier slots may have advanced when a later slot fails.
        """
        ordered_observations = tuple(observations)
        ordered_resets = tuple(reset_mask)
        ordered_contexts = (
            (None,) * self.batch_size
            if reset_contexts is None
            else tuple(reset_contexts)
        )
        if (
            len(ordered_observations) != self.batch_size
            or len(ordered_resets) != self.batch_size
            or len(ordered_contexts) != self.batch_size
        ):
            raise ValueError(
                f"Batch size is {self.batch_size}, got "
                f"{len(ordered_observations)} observations and "
                f"{len(ordered_resets)} reset flags and "
                f"{len(ordered_contexts)} reset contexts."
            )
        for index, reset in enumerate(ordered_resets):
            if type(reset) is not bool:
                raise TypeError(
                    f"reset_mask[{index}] must be bool, got "
                    f"{type(reset).__module__}.{type(reset).__qualname__}."
                )
            context = ordered_contexts[index]
            if context is not None and not isinstance(
                context, BatchEpisodeContext
            ):
                raise TypeError(
                    f"reset_contexts[{index}] must be "
                    "BatchEpisodeContext or None, got "
                    f"{type(context).__module__}."
                    f"{type(context).__qualname__}."
                )
            if context is not None and not reset:
                raise ValueError(
                    f"reset_contexts[{index}] was provided while "
                    f"reset_mask[{index}] is False."
                )

        results = []
        for index, (observation, reset) in enumerate(
            zip(ordered_observations, ordered_resets)
        ):
            if reset:
                replacement_context = ordered_contexts[index]
                self._engines[index] = None
                context = (
                    replacement_context
                    if replacement_context is not None
                    else self._contexts[index]
                )
                try:
                    candidate = self._prototype._fork_unstarted()
                    candidate.start_episode(
                        dt=context.dt,
                        capabilities=context.capabilities,
                        catalog=context.catalog,
                        require_confirmed=context.require_confirmed,
                        skip_unsatisfiable=context.skip_unsatisfiable,
                    )
                except Exception as exc:
                    raise BatchStepError(index, "reset", exc) from exc
                if replacement_context is not None:
                    self._contexts[index] = replacement_context
                self._engines[index] = candidate

            try:
                result = self.engine(index).update(observation)
            except Exception as exc:
                raise BatchStepError(index, "update", exc) from exc
            results.append(result)

        return tuple(results)
