"""
Quantitative semantics of parsed STL formulas, checked against an oracle.

The oracle computes past-time STL robustness directly over recorded signal
traces (brute force, no RTAMT). The pipeline under test is the real one:
``parse_stl`` → ``IncrementalBindingManager`` → RTAMT.

Interval convention (documented in stl_parser):
an authored bound ``[0,N]`` (steps) covers exactly max(N, 1) samples, i.e.
RTAMT receives ``[0, N-1]``. The oracle mirrors that conversion, then
applies textbook past-time semantics over the converted inclusive window.

Robustness is compared only at steps where every operand's full history
exists (``t >= horizon``); warmup values are provisional by design.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from specola.entities import EntityState, TrackState
from specola.monitor import IncrementalBindingManager
from specola.stl_parser import parse_stl


# ---------------------------------------------------------------------------
# Pipeline harness: one pinned entity of class 'sig' carrying the signals
# ---------------------------------------------------------------------------

def run_pipeline(formula, signals, dt=1.0, constants=None, backend=None):
    """Run the full parse→monitor pipeline over per-step property traces.

    signals: {property_name: sequence of floats}, equal lengths.
    Returns (parsed_formula, [rho at each step]).
    """
    lengths = {len(v) for v in signals.values()}
    assert len(lengths) == 1
    (steps,) = lengths

    kwargs = {} if backend is None else {"backend": backend}
    parsed = parse_stl(formula, dt=dt, constants=constants, **kwargs)
    manager = IncrementalBindingManager(parsed)

    rhos = []
    for t in range(steps):
        entity = EntityState(
            entity_id=1,
            entity_class="sig",
            position=np.zeros(3),
            velocity=np.zeros(3),
            track_age=t,
            track_state=TrackState.CONFIRMED,
            properties={name: float(trace[t]) for name, trace in signals.items()},
        )
        observation = SimpleNamespace(
            timestep=t, time=float(t), perceived_entities={1: entity}
        )
        results = manager.update(observation)
        assert len(results) == 1, "exactly one binding expected"
        rho = next(iter(results.values()))
        assert rho is not None, "binding must stay active"
        rhos.append(rho)

    return parsed, rhos


# ---------------------------------------------------------------------------
# Oracle: textbook past-time robustness over converted inclusive windows
# ---------------------------------------------------------------------------

def convert_interval(lower, upper):
    """Mirror the parser's sample-count convention: [0,N] → [0, N-1]."""
    upper = upper - 1 if upper > 0 else 0
    if upper < lower:
        upper = lower
    return lower, upper


def hist(arr, lower, upper):
    """historically[lower,upper]: min over samples t-upper .. t-lower."""
    lower, upper = convert_interval(lower, upper)
    out = np.full(len(arr), np.nan)
    for t in range(len(arr)):
        if t - upper < 0:
            continue
        out[t] = min(arr[t - upper : t - lower + 1])
    return out


def once(arr, lower, upper):
    """once[lower,upper]: max over samples t-upper .. t-lower."""
    lower, upper = convert_interval(lower, upper)
    out = np.full(len(arr), np.nan)
    for t in range(len(arr)):
        if t - upper < 0:
            continue
        out[t] = max(arr[t - upper : t - lower + 1])
    return out


def since(left, right, lower, upper):
    """left since[lower,upper] right (converted inclusive window).

    rho(t) = max over t' in [t-upper, t-lower] of
             min( right[t'], min over t'' in (t', t] of left[t''] )
    """
    lower, upper = convert_interval(lower, upper)
    out = np.full(len(left), np.nan)
    for t in range(len(left)):
        if t - upper < 0:
            continue
        candidates = []
        for tp in range(t - upper, t - lower + 1):
            left_tail = left[tp + 1 : t + 1]
            hold = min(left_tail) if len(left_tail) else math.inf
            candidates.append(min(right[tp], hold))
        out[t] = max(candidates)
    return out


def assert_matches_oracle(rhos, expected, horizon):
    """Compare pipeline robustness with the oracle for all t >= horizon."""
    assert len(rhos) > horizon, "trace too short to leave the warmup region"
    for t in range(horizon, len(rhos)):
        assert rhos[t] == pytest.approx(expected[t], abs=1e-9), (
            f"step {t}: pipeline={rhos[t]!r} oracle={expected[t]!r}"
        )


def make_traces(seed=0, steps=40):
    rng = np.random.default_rng(seed)
    return {
        "a": np.round(rng.uniform(-2.0, 2.0, steps), 3),
        "b": np.round(rng.uniform(-2.0, 2.0, steps), 3),
    }


# ---------------------------------------------------------------------------
# Atomic and Boolean nesting
# ---------------------------------------------------------------------------

def test_bare_atomic_is_signed_distance():
    sig = make_traces()
    _, rhos = run_pipeline("sig_1[a] >= 0.5", sig)
    assert_matches_oracle(rhos, sig["a"] - 0.5, horizon=0)


def test_negation():
    sig = make_traces()
    _, rhos = run_pipeline("not (sig_1[a] >= 0.5)", sig)
    assert_matches_oracle(rhos, -(sig["a"] - 0.5), horizon=0)


def test_and_or_precedence_or_binds_looser():
    sig = make_traces()
    _, rhos = run_pipeline(
        "sig_1[a] >= 0 or sig_1[a] >= 1 and sig_1[b] >= 0", sig
    )
    expected = np.maximum(
        sig["a"], np.minimum(sig["a"] - 1.0, sig["b"])
    )
    assert_matches_oracle(rhos, expected, horizon=0)


def test_implication():
    sig = make_traces()
    _, rhos = run_pipeline("(sig_1[a] >= 0) -> (sig_1[b] >= 0)", sig)
    expected = np.maximum(-sig["a"], sig["b"])
    assert_matches_oracle(rhos, expected, horizon=0)


def test_implication_is_right_associative():
    sig = make_traces()
    _, rhos = run_pipeline(
        "sig_1[a] >= 0 -> sig_1[b] >= 0 -> sig_1[a] >= 1", sig
    )
    # a -> (b -> c)  ==  max(-a, max(-b, c))
    expected = np.maximum(
        -sig["a"], np.maximum(-sig["b"], sig["a"] - 1.0)
    )
    assert_matches_oracle(rhos, expected, horizon=0)


def test_equivalence():
    sig = make_traces()
    _, rhos = run_pipeline("(sig_1[a] >= 0) <-> (sig_1[b] >= 0)", sig)
    # (l -> r) and (r -> l)
    expected = np.minimum(
        np.maximum(-sig["a"], sig["b"]),
        np.maximum(-sig["b"], sig["a"]),
    )
    assert_matches_oracle(rhos, expected, horizon=0)


def test_xor():
    sig = make_traces()
    _, rhos = run_pipeline("(sig_1[a] >= 0) xor (sig_1[b] >= 0)", sig)
    # not(l <-> r)
    expected = -np.minimum(
        np.maximum(-sig["a"], sig["b"]),
        np.maximum(-sig["b"], sig["a"]),
    )
    assert_matches_oracle(rhos, expected, horizon=0)


def test_equality_comparison():
    sig = make_traces()
    _, rhos = run_pipeline("sig_1[a] == sig_1[b]", sig)
    assert_matches_oracle(rhos, -np.abs(sig["a"] - sig["b"]), horizon=0)


# ---------------------------------------------------------------------------
# Temporal windows, plain and nested
# ---------------------------------------------------------------------------

def test_trailing_always():
    sig = make_traces()
    _, rhos = run_pipeline("G[0,5](sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos, hist(sig["a"], 0, 5), horizon=5)


def test_trailing_eventually():
    sig = make_traces()
    _, rhos = run_pipeline("F[0,5](sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos, once(sig["a"], 0, 5), horizon=5)


def test_temporal_over_boolean():
    sig = make_traces()
    _, rhos = run_pipeline(
        "G[0,4](sig_1[a] >= 0 and sig_1[b] >= 0)", sig
    )
    expected = hist(np.minimum(sig["a"], sig["b"]), 0, 4)
    assert_matches_oracle(rhos, expected, horizon=4)


def test_boolean_over_temporal():
    sig = make_traces()
    _, rhos = run_pipeline(
        "G[0,3](sig_1[a] >= 0) and F[0,5](sig_1[b] >= 0)", sig
    )
    expected = np.minimum(hist(sig["a"], 0, 3), once(sig["b"], 0, 5))
    assert_matches_oracle(rhos, expected, horizon=5)


def test_negated_temporal():
    sig = make_traces()
    _, rhos = run_pipeline("not G[0,3](sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos, -hist(sig["a"], 0, 3), horizon=3)


def test_nested_always_of_eventually():
    sig = make_traces()
    _, rhos = run_pipeline("G[0,4](F[0,3](sig_1[a] >= 0))", sig)
    expected = hist(once(sig["a"], 0, 3), 0, 4)
    # horizon: (4-1) + (3-1) after conversion = 5 steps of history
    assert_matches_oracle(rhos, expected, horizon=5)


def test_nested_eventually_of_always():
    sig = make_traces()
    _, rhos = run_pipeline("F[0,4](G[0,3](sig_1[a] >= 0))", sig)
    expected = once(hist(sig["a"], 0, 3), 0, 4)
    assert_matches_oracle(rhos, expected, horizon=5)


def test_since():
    sig = make_traces()
    _, rhos = run_pipeline(
        "(sig_1[a] >= 0) since[0,6] (sig_1[b] >= 0)", sig
    )
    expected = since(sig["a"], sig["b"], 0, 6)
    assert_matches_oracle(rhos, expected, horizon=5)


def test_seconds_bounds_convert_via_dt():
    sig = make_traces()
    # dt=0.5 → 2s = 4 steps: identical to authored [0,4]
    _, rhos = run_pipeline("G[0,2s](sig_1[a] >= 0)", sig, dt=0.5)
    assert_matches_oracle(rhos, hist(sig["a"], 0, 4), horizon=4)


def test_window_steps_covers_nested_horizon():
    sig = make_traces()
    parsed, _ = run_pipeline("G[0,4](F[0,3](sig_1[a] >= 0))", sig)
    # 4 samples of a signal that itself needs 3 samples → 6 total
    assert parsed.window_steps == 6


# ---------------------------------------------------------------------------
# Structure and error handling
# ---------------------------------------------------------------------------

def test_parameter_slots_merge_across_nested_subformulas():
    parsed = parse_stl(
        "G[0,3](sig_1[a] >= 0) and F[0,2](sig_1[b] >= 0)", dt=1.0
    )
    assert len(parsed.params) == 1
    assert parsed.params[0].required_components == ["a", "b"]


def test_chained_comparison_rejected():
    with pytest.raises(ValueError, match="[Cc]hained"):
        parse_stl("0 <= sig_1[a] <= 1", dt=1.0)


def test_not_equal_rejected_with_guidance():
    with pytest.raises(ValueError, match="!="):
        parse_stl("sig_1[a] != 0", dt=1.0)


def test_empty_temporal_body_rejected():
    with pytest.raises(ValueError):
        parse_stl("G[0,5]()", dt=1.0)


def test_unbalanced_parentheses_rejected():
    with pytest.raises(ValueError):
        parse_stl("G[0,5](sig_1[a] >= 0", dt=1.0)


def test_since_requires_two_operands():
    with pytest.raises(ValueError):
        parse_stl("since[0,5] (sig_1[a] >= 0)", dt=1.0)


# ---------------------------------------------------------------------------
# Exact finality boundaries and warmup provisional semantics
# ---------------------------------------------------------------------------

def test_always_final_at_exact_window_boundary():
    sig = make_traces()
    _, rhos = run_pipeline("G[0,5](sig_1[a] >= 0)", sig)
    # authored [0,5] = 5 samples: first full window at t=4, not t=5
    assert_matches_oracle(rhos, hist(sig["a"], 0, 5), horizon=4)


def test_warmup_is_robustness_over_available_prefix():
    sig = make_traces()
    _, rhos = run_pipeline("G[0,5](sig_1[a] >= 0)", sig)
    for t in range(4):
        assert rhos[t] == pytest.approx(min(sig["a"][: t + 1]), abs=1e-9)


def test_nonzero_lower_bound_always():
    sig = make_traces()
    _, rhos = run_pipeline("G[1,5](sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos, hist(sig["a"], 1, 5), horizon=4)


def test_nonzero_lower_bound_since():
    sig = make_traces()
    _, rhos = run_pipeline(
        "(sig_1[a] >= 0) since[1,4] (sig_1[b] >= 0)", sig
    )
    assert_matches_oracle(rhos, since(sig["a"], sig["b"], 1, 4), horizon=3)


def test_single_sample_windows():
    sig = make_traces()
    _, rhos0 = run_pipeline("G[0,0](sig_1[a] >= 0)", sig)
    _, rhos1 = run_pipeline("G[0,1](sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos0, sig["a"], horizon=0)
    assert_matches_oracle(rhos1, sig["a"], horizon=0)


def test_triple_nesting():
    sig = make_traces(steps=60)
    _, rhos = run_pipeline(
        "F[0,3](G[0,2](F[0,4](sig_1[a] >= 0)))", sig
    )
    expected = once(hist(once(sig["a"], 0, 4), 0, 2), 0, 3)
    assert_matches_oracle(rhos, expected, horizon=7)


def test_since_nested_inside_always():
    sig = make_traces(steps=60)
    _, rhos = run_pipeline(
        "G[0,3]((sig_1[a] >= 0) since[0,4] (sig_1[b] >= 0))", sig
    )
    expected = hist(since(sig["a"], sig["b"], 0, 4), 0, 3)
    assert_matches_oracle(rhos, expected, horizon=6)


# ---------------------------------------------------------------------------
# Operator spelling variants
# ---------------------------------------------------------------------------

def test_word_and_symbol_operator_variants():
    sig = make_traces()
    variants = [
        "historically[0,4](sig_1[a] >= 0) && once[0,3](sig_1[b] >= 0)",
        "always[0,4](sig_1[a] >= 0) || eventually[0,3](sig_1[b] >= 0)",
        "H[0,4](sig_1[a] >= 0) and P[0,3](sig_1[b] >= 0)",
    ]
    expected = [
        np.minimum(hist(sig["a"], 0, 4), once(sig["b"], 0, 3)),
        np.maximum(hist(sig["a"], 0, 4), once(sig["b"], 0, 3)),
        np.minimum(hist(sig["a"], 0, 4), once(sig["b"], 0, 3)),
    ]
    for formula, exp in zip(variants, expected):
        _, rhos = run_pipeline(formula, sig)
        assert_matches_oracle(rhos, exp, horizon=3)


def test_double_negation():
    sig = make_traces()
    _, rhos = run_pipeline("not not (sig_1[a] >= 0)", sig)
    assert_matches_oracle(rhos, sig["a"], horizon=0)


# ---------------------------------------------------------------------------
# Constants (design-spec defect 5)
# ---------------------------------------------------------------------------

def test_constant_named_like_sympy_builtin():
    sig = make_traces()
    _, rhos = run_pipeline(
        "sig_1[a] <= limit", sig, constants={"limit": 0.5}
    )
    assert_matches_oracle(rhos, 0.5 - sig["a"], horizon=0)


def test_constant_colliding_with_generated_signal_rejected():
    with pytest.raises(ValueError, match="sig_1_a"):
        parse_stl(
            "sig_1[a] >= 0", dt=1.0, constants={"sig_1_a": 99.0}
        )
