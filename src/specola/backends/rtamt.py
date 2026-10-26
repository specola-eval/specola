"""
rtamt.py — RTAMT-backed temporal evaluation.

Owns RTAMT health probing and the discrete-time specification wrapper.
Availability is a compile-time concern: semantic compilation calls
``rtamt_available()`` when a backend plan resolves to RTAMT and fails
loudly if it is missing or its C++ backend is broken.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional


def _probe_rtamt() -> bool:
    """True only if RTAMT imports and its online update() is functional.

    The broken pure-Python fallback returns 0.0 from update(), so the
    probe checks actual values.
    """
    try:
        import rtamt

        spec = rtamt.StlDiscreteTimeSpecification()
        spec.declare_var("mu", "float")
        spec.spec = "historically[0,1](mu)"
        spec.parse()
        r0 = spec.update(0, [("mu", 3.7)])
        r1 = spec.update(1, [("mu", 1.2)])
        return (r0 == 3.7) and (r1 == 1.2)
    except Exception:
        return False


_PROBE_RESULT: Optional[bool] = None


def rtamt_available() -> bool:
    """Memoized RTAMT health check."""
    global _PROBE_RESULT
    if _PROBE_RESULT is None:
        _PROBE_RESULT = _probe_rtamt()
    return _PROBE_RESULT


def _parse_specification(
    spec_text: str,
    variable_names: List[str],
) -> Any:
    import rtamt

    specification = rtamt.StlDiscreteTimeSpecification()
    for name in variable_names:
        specification.declare_var(name, "float")
    specification.spec = spec_text
    specification.parse()
    return specification


class RtamtBackend:
    """TemporalBackend over one RTAMT discrete-time specification."""

    def __init__(self, spec_text: str, variable_names: List[str]) -> None:
        self._spec_text = spec_text
        self._variable_names = list(variable_names)
        self._specification = _parse_specification(
            spec_text, self._variable_names
        )
        self._step = 0

    def update(
        self,
        timestep: int,
        signals: Mapping[str, float],
    ) -> float:
        del timestep  # RTAMT consumes a local consecutive step index
        robustness = self._specification.update(
            self._step,
            [(name, signals[name]) for name in self._variable_names],
        )
        self._step += 1
        return float(robustness)

    def reset(self) -> None:
        # RTAMT offers no online reset; rebuild the specification.
        self._specification = _parse_specification(
            self._spec_text, self._variable_names
        )
        self._step = 0


def make_legacy_spec(formula: Any) -> Optional[Any]:
    """RTAMT specification for legacy hand-constructed formulas.

    Kept for SpecTemplate objects built without the semantic
    compiler (rtamt_spec text and/or single-mu temporal_op form). New
    code paths use backend factories chosen at compilation.
    """
    if not rtamt_available():
        return None

    import rtamt

    b = max(1, formula.window_steps - 1)

    specification = rtamt.StlDiscreteTimeSpecification()

    if formula.rtamt_spec is not None:
        if not formula.predicate_fns:
            raise ValueError(
                "formula.rtamt_spec requires formula.predicate_fns"
            )
        for name in sorted(formula.predicate_fns.keys()):
            specification.declare_var(name, "float")
        specification.spec = formula.rtamt_spec
    else:
        specification.declare_var("mu", "float")
        op = formula.temporal_op
        if op == "always":
            specification.spec = f"historically[0,{b}](mu)"
        elif op == "eventually":
            specification.spec = f"once[0,{b}](mu)"
        else:
            specification.spec = "mu"

    specification.parse()
    return specification
