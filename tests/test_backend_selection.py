"""
Temporal backend selection (design stage 6, §5.8).

Selection happens at compile time: ``native`` supports the complete causal
past-time language, ``rtamt`` requires a working RTAMT, and ``auto`` prefers
native. An explicit RTAMT request fails at compilation when unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from specola.entities import SpecConfigError
from specola.stl_parser import parse_stl
from test_parser_semantics import make_traces, run_pipeline


FRAGMENT_FORMULAS = [
    "sig_1[a] >= 0.5",
    "not (sig_1[a] >= 0) and (sig_1[b] >= 0 or sig_1[a] >= 1)",
    "(sig_1[a] >= 0) <-> (sig_1[b] >= 0)",
    "G[0,5](sig_1[a] >= 0)",
    "F[0,4](sig_1[a] >= 0 and sig_1[b] >= 0)",
    "G[0,4](F[0,3](sig_1[a] >= 0))",
    "G[0,3](sig_1[a] >= 0) and F[0,5](sig_1[b] >= 0)",
    "not G[0,3](sig_1[a] >= 0)",
    "(sig_1[a] >= 0) since[0,6] (sig_1[b] >= 0)",
    "G[1,5](sig_1[a] >= 0)",
    "historically(sig_1[a] >= 0)",
    "once(sig_1[a] >= 0)",
    "prev(sig_1[a] >= 0)",
    "rise(sig_1[a] >= 0)",
    "fall(sig_1[a] >= 0)",
]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_auto_selects_native_for_fragment():
    for formula in FRAGMENT_FORMULAS:
        parsed = parse_stl(formula, dt=1.0)
        assert parsed.backend == "native", formula


def test_requested_native_accepts_complete_causal_fragment():
    for formula in FRAGMENT_FORMULAS:
        parsed = parse_stl(formula, dt=1.0, backend="native")
        assert parsed.backend == "native"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        parse_stl("sig_1[a] >= 0", dt=1.0, backend="magic")


# ---------------------------------------------------------------------------
# Missing RTAMT: compile-time failure, and native independence
# ---------------------------------------------------------------------------

def test_missing_rtamt_fails_compilation_when_required(monkeypatch):
    from specola.backends import rtamt as rtamt_backend

    monkeypatch.setattr(rtamt_backend, "rtamt_available", lambda: False)

    with pytest.raises(SpecConfigError, match="RTAMT"):
        parse_stl("sig_1[a] >= 0", dt=1.0, backend="rtamt")


def test_native_fragment_runs_without_rtamt(monkeypatch):
    from specola.backends import rtamt as rtamt_backend

    monkeypatch.setattr(rtamt_backend, "rtamt_available", lambda: False)

    sig = make_traces()
    parsed, rhos = run_pipeline("G[0,5](sig_1[a] >= 0)", sig)

    assert parsed.backend == "native"
    assert rhos[10] == pytest.approx(min(sig["a"][5:11]))


# ---------------------------------------------------------------------------
# Backend parity: native vs RTAMT on identical traces, every step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("formula", FRAGMENT_FORMULAS)
def test_native_matches_rtamt_on_fragment(formula):
    sig = make_traces(seed=3, steps=50)

    _, native_rhos = run_pipeline(formula, sig, backend="native")
    _, rtamt_rhos = run_pipeline(formula, sig, backend="rtamt")

    for t, (native_rho, rtamt_rho) in enumerate(
        zip(native_rhos, rtamt_rhos)
    ):
        assert native_rho == pytest.approx(rtamt_rho, abs=1e-9), (
            f"{formula}: backend divergence at step {t}: "
            f"native={native_rho} rtamt={rtamt_rho}"
        )


def test_native_matches_rtamt_on_random_traces():
    rng = np.random.default_rng(42)
    for trial in range(20):
        sig = {
            "a": np.round(rng.uniform(-2, 2, 30), 3),
            "b": np.round(rng.uniform(-2, 2, 30), 3),
        }
        formula = "G[0,4](F[0,3](sig_1[a] >= 0)) and (sig_1[b] >= 0)"
        _, native_rhos = run_pipeline(formula, sig, backend="native")
        _, rtamt_rhos = run_pipeline(formula, sig, backend="rtamt")
        assert native_rhos == pytest.approx(rtamt_rhos, abs=1e-9)


# ---------------------------------------------------------------------------
# Iff/Xor over temporal subformulas (reviewer finding): RTAMT evaluates
# duplicated stateful subformulas incorrectly; the encoding is rejected.
# ---------------------------------------------------------------------------

def test_iff_over_temporal_native_matches_oracle():
    import numpy as np
    from test_parser_semantics import hist, once

    sig = make_traces(seed=7, steps=40)
    formula = "G[0,3](sig_1[a] >= 0) <-> F[0,2](sig_1[b] >= 0)"

    parsed, rhos = run_pipeline(formula, sig)  # auto → native
    assert parsed.backend == "native"

    left = hist(sig["a"], 0, 3)
    right = once(sig["b"], 0, 2)
    expected = np.minimum(
        np.maximum(-left, right), np.maximum(-right, left)
    )
    for t in range(3, 40):
        assert rhos[t] == pytest.approx(expected[t], abs=1e-9)


def test_iff_over_temporal_rejected_for_rtamt_backend():
    with pytest.raises(SpecConfigError, match="duplicates"):
        parse_stl(
            "G[0,3](sig_1[a] >= 0) <-> (sig_1[b] >= 0)",
            dt=1.0,
            backend="rtamt",
        )


def test_xor_over_since_auto_uses_native():
    parsed = parse_stl(
        "((sig_1[a] >= 0) since[0,3] (sig_1[b] >= 0)) "
        "xor (sig_1[b] >= 1)",
        dt=1.0,
    )
    assert parsed.backend == "native"
