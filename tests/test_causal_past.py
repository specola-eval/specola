"""Causal past-time syntax and native/RTAMT quantitative parity."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from specola.spec import ast, render
from specola.spec.parser import parse
from specola.stl_parser import parse_stl
from test_lifecycle_rewarm import obs, only_instance
from test_parser_semantics import make_traces, run_pipeline
from specola.monitor import IncrementalBindingManager


ATOM_A = "sig_1[a] >= 0"
ATOM_B = "sig_1[b] >= 0"


STRICT_ORDER_CANDIDATES = {
    "state_trigger": (
        f"historically(({ATOM_B}) -> prev(once({ATOM_A})))"
    ),
    "rising_edge_trigger": (
        f"historically(rise({ATOM_B}) -> prev(once({ATOM_A})))"
    ),
}


def _strict_before_oracle(a_trace, b_trace):
    """Independent Boolean oracle: every true B has a strictly earlier A."""
    return all(
        not b_value or any(a_trace[:step])
        for step, b_value in enumerate(b_trace)
    )


def _completed_bounded_response_oracle(trigger, response, bound):
    """Inclusive [0, bound] response over obligations complete by trace end."""
    rises = [
        step
        for step, value in enumerate(trigger)
        if value and (step == 0 or not trigger[step - 1])
    ]
    final_step = len(trigger) - 1
    return all(
        any(response[start : start + bound + 1])
        for start in rises
        if start + bound <= final_step
    )


def _rise_onset_oracle(trace):
    """Independent Boolean origin-inclusive false-to-true edge oracle."""
    return tuple(
        value and (step == 0 or not trace[step - 1])
        for step, value in enumerate(trace)
    )


@pytest.mark.parametrize(
    ("text", "node_type", "bounded"),
    [
        (f"once({ATOM_A})", ast.Eventually, False),
        (f"historically({ATOM_A})", ast.Always, False),
        (f"once[1,4]({ATOM_A})", ast.Eventually, True),
        (f"historically[1,4]({ATOM_A})", ast.Always, True),
        (f"prev({ATOM_A})", ast.Previous, False),
        (f"rise({ATOM_A})", ast.Rise, False),
        (f"fall({ATOM_A})", ast.Fall, False),
        (f"{ATOM_A} since {ATOM_B}", ast.Since, False),
        (f"{ATOM_A} since[1,4] {ATOM_B}", ast.Since, True),
    ],
)
def test_canonical_syntax_parses_and_round_trips(text, node_type, bounded):
    node = parse(text)
    assert isinstance(node, node_type)
    if isinstance(node, (ast.Always, ast.Eventually, ast.Since)):
        assert (node.bounds is not None) is bounded

    canonical = render.formula_text(node)
    assert render.formula_text(parse(canonical)) == canonical
    assert parse(canonical) == node


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        (f"G[0,4]({ATOM_A})", f"historically[0,4]({ATOM_A})"),
        (f"H[0,4]({ATOM_A})", f"historically[0,4]({ATOM_A})"),
        (f"F[0,4]({ATOM_A})", f"once[0,4]({ATOM_A})"),
        (f"P[0,4]({ATOM_A})", f"once[0,4]({ATOM_A})"),
    ],
)
def test_legacy_bounded_aliases_render_canonically(legacy, canonical):
    assert render.formula_text(parse(legacy)) == canonical


def test_since_is_non_associative_without_parentheses():
    with pytest.raises(ValueError, match="non-associative"):
        parse(f"{ATOM_A} since {ATOM_B} since sig_1[c] >= 0")

    left_grouped = parse(f"({ATOM_A} since {ATOM_B}) since sig_1[c] >= 0")
    right_grouped = parse(f"{ATOM_A} since ({ATOM_B} since sig_1[c] >= 0)")
    assert isinstance(left_grouped, ast.Since)
    assert isinstance(left_grouped.left, ast.Since)
    assert isinstance(right_grouped, ast.Since)
    assert isinstance(right_grouped.right, ast.Since)


@pytest.mark.parametrize("spacing", ["  ", "\t ", " \t\t"])
def test_one_since_is_not_mistaken_for_chain_after_whitespace_run(spacing):
    node = parse(f"{ATOM_A}{spacing}since {ATOM_B}")
    assert isinstance(node, ast.Since)


def test_since_binds_tighter_than_and_and_looser_than_prefix_unary():
    node = parse(f"{ATOM_A} and rise({ATOM_B}) since sig_1[c] >= 0")
    assert isinstance(node, ast.And)
    assert isinstance(node.right, ast.Since)
    assert isinstance(node.right.left, ast.Rise)


@pytest.mark.parametrize("backend", ["native", "rtamt"])
def test_operator_origin_semantics(backend):
    signals = {
        "a": np.array([2.0, -3.0, 4.0]),
        "b": np.array([-1.0, 5.0, -2.0]),
    }
    cases = [
        (f"once({ATOM_A})", [2.0, 2.0, 4.0]),
        (f"historically({ATOM_A})", [2.0, -3.0, -3.0]),
        (f"prev({ATOM_A})", [-math.inf, 2.0, -3.0]),
        (f"rise({ATOM_A})", [2.0, -3.0, 3.0]),
        (f"fall({ATOM_A})", [-2.0, 2.0, -4.0]),
        (f"{ATOM_A} since {ATOM_B}", [-1.0, 5.0, 4.0]),
    ]
    for formula, expected in cases:
        _, rhos = run_pipeline(formula, signals, backend=backend)
        assert rhos == pytest.approx(expected), formula


@pytest.mark.parametrize("backend", ["native", "rtamt"])
def test_nonzero_lower_bound_uses_identity_before_eligibility(backend):
    signals = {"a": np.array([2.0, -3.0, 4.0, 1.0])}
    _, once_rho = run_pipeline(f"once[2,4]({ATOM_A})", signals, backend=backend)
    _, hist_rho = run_pipeline(
        f"historically[2,4]({ATOM_A})", signals, backend=backend
    )
    assert once_rho[:2] == [-math.inf, -math.inf]
    assert hist_rho[:2] == [math.inf, math.inf]
    assert once_rho[2:] == pytest.approx([2.0, 2.0])
    assert hist_rho[2:] == pytest.approx([2.0, -3.0])


CAUSAL_FORMULAS = [
    f"once({ATOM_A})",
    f"historically({ATOM_A})",
    f"once[0,5]({ATOM_A})",
    f"historically[2,6]({ATOM_A})",
    f"prev({ATOM_A})",
    f"rise({ATOM_A})",
    f"fall({ATOM_A})",
    f"{ATOM_A} since {ATOM_B}",
    f"{ATOM_A} since[0,6] {ATOM_B}",
    f"{ATOM_A} since[2,6] {ATOM_B}",
    f"historically[0,4](once({ATOM_A}) or rise({ATOM_B}))",
    f"once(prev({ATOM_A}) and ({ATOM_A} since {ATOM_B}))",
]


@pytest.mark.parametrize("formula", CAUSAL_FORMULAS)
def test_native_matches_rtamt_every_step(formula):
    for seed in range(5):
        signals = make_traces(seed=seed, steps=50)
        _, native = run_pipeline(formula, signals, backend="native")
        _, rtamt = run_pipeline(formula, signals, backend="rtamt")
        assert native == pytest.approx(rtamt, abs=1e-9), (formula, seed)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(40),
        np.ones(40),
        np.resize(np.array([-1.0, 0.0, 1.0]), 40),
    ],
)
def test_native_matches_rtamt_on_ties_and_constant_traces(values):
    signals = {"a": values, "b": values[::-1]}
    for formula in CAUSAL_FORMULAS:
        _, native = run_pipeline(formula, signals, backend="native")
        _, rtamt = run_pipeline(formula, signals, backend="rtamt")
        assert native == pytest.approx(rtamt, abs=1e-9), formula


@pytest.mark.parametrize(
    ("candidate_name", "formula"),
    STRICT_ORDER_CANDIDATES.items(),
)
@pytest.mark.parametrize("backend_name", ["native", "rtamt"])
def test_strict_order_candidates_match_independent_exhaustive_boolean_oracle(
    candidate_name,
    formula,
    backend_name,
):
    """Both backends match strict-before on all 1,024 five-step A/B traces."""
    for values in product((False, True), repeat=10):
        a_trace = values[:5]
        b_trace = values[5:]
        signals = {
            "a": np.asarray([1.0 if value else -1.0 for value in a_trace]),
            "b": np.asarray([1.0 if value else -1.0 for value in b_trace]),
        }
        _, robustness = run_pipeline(formula, signals, backend=backend_name)
        observed = robustness[-1] >= 0.0
        expected = _strict_before_oracle(a_trace, b_trace)
        assert observed is expected, (
            "independent strict-before oracle mismatch: "
            f"candidate={candidate_name} backend={backend_name} "
            f"a={a_trace} b={b_trace} rho={robustness[-1]}"
        )


@pytest.mark.parametrize("backend_name", ["native", "rtamt"])
def test_rise_matches_independent_exhaustive_onset_oracle(backend_name):
    """Both backends expose exactly the origin-inclusive onset indices."""
    for trace in product((False, True), repeat=5):
        signals = {
            "a": np.zeros(5),
            "b": np.asarray([1.0 if value else -1.0 for value in trace]),
        }
        _, robustness = run_pipeline(
            f"rise({ATOM_B})", signals, backend=backend_name
        )
        observed = tuple(value >= 0.0 for value in robustness)
        expected = _rise_onset_oracle(trace)
        assert observed == expected, (
            "independent rise-onset oracle mismatch: "
            f"backend={backend_name} trace={trace} "
            f"observed={observed} expected={expected}"
        )


@pytest.mark.parametrize("backend_name", ["native", "rtamt"])
@pytest.mark.parametrize("bound", range(1, 5))
def test_delayed_response_endpoint_matches_inclusive_oracle_with_plus_delta(
    backend_name,
    bound,
):
    """Authored [0,b+1] covers samples 0..b under sample-count bounds."""
    trigger_step = 1
    final_step = trigger_step + bound
    trigger = [False] * (final_step + 1)
    trigger[trigger_step] = True

    for response_delay in range(bound + 2):
        response = [False] * (final_step + 1)
        response_step = trigger_step + response_delay
        if response_step <= final_step:
            response[response_step] = True

        signals = {
            "a": np.asarray(
                [1.0 if value else -1.0 for value in trigger]
            ),
            "b": np.asarray(
                [1.0 if value else -1.0 for value in response]
            ),
        }
        formula = (
            "historically("
            f"once[{bound},{bound}](rise({ATOM_A})) -> "
            f"once[0,{bound + 1}]({ATOM_B})"
            ")"
        )
        _, robustness = run_pipeline(formula, signals, backend=backend_name)
        observed = robustness[-1] >= 0.0
        expected = _completed_bounded_response_oracle(
            trigger, response, bound
        )
        assert observed is expected, (
            "independent inclusive bounded-response oracle mismatch: "
            f"backend={backend_name} b={bound} "
            f"response_delay={response_delay} rho={robustness[-1]}"
        )


@pytest.mark.parametrize("backend_name", ["native", "rtamt"])
@pytest.mark.parametrize("bound", range(1, 5))
def test_delayed_response_without_plus_delta_omits_immediate_response_sample(
    backend_name,
    bound,
):
    """The naive authored [0,b] endpoint excludes trigger-time response."""
    trigger_step = 1
    final_step = trigger_step + bound
    trigger = [False] * (final_step + 1)
    response = [False] * (final_step + 1)
    trigger[trigger_step] = True
    response[trigger_step] = True
    assert _completed_bounded_response_oracle(trigger, response, bound)

    signals = {
        "a": np.asarray([1.0 if value else -1.0 for value in trigger]),
        "b": np.asarray([1.0 if value else -1.0 for value in response]),
    }
    naive_formula = (
        "historically("
        f"once[{bound},{bound}](rise({ATOM_A})) -> "
        f"once[0,{bound}]({ATOM_B})"
        ")"
    )
    _, robustness = run_pipeline(
        naive_formula, signals, backend=backend_name
    )
    assert robustness[-1] < 0.0, (
        "naive endpoint unexpectedly retained trigger-time response: "
        f"backend={backend_name} b={bound} rho={robustness[-1]}"
    )


def test_auto_selects_native_for_complete_causal_fragment():
    for formula in CAUSAL_FORMULAS:
        assert parse_stl(formula, dt=1.0).backend == "native", formula


@pytest.mark.parametrize(
    ("formula", "first", "after_reset"),
    [
        (f"once({ATOM_A})", -4.0, 3.0),
        (f"historically({ATOM_A})", -4.0, 3.0),
        (f"prev({ATOM_A})", -math.inf, -math.inf),
        (f"rise({ATOM_A})", -4.0, 3.0),
        (f"fall({ATOM_A})", 4.0, -3.0),
    ],
)
def test_reset_and_rewarm_discards_history(formula, first, after_reset):
    parsed = parse_stl(formula, dt=1.0, valid_classes=["sig"])
    manager = IncrementalBindingManager(parsed)

    initial = manager.update(obs(0, -4.0))
    assert next(iter(initial.values())) == pytest.approx(first)
    manager.update(obs(1, None))
    resumed = manager.update(obs(2, 3.0))

    assert next(iter(resumed.values())) == pytest.approx(after_reset)
    assert only_instance(manager).verdict_final


def test_rtamt_reset_reparses_once_and_restores_clean_history(monkeypatch):
    from specola.backends import rtamt as rtamt_backend

    calls = 0
    original = rtamt_backend._parse_specification

    def counted(spec_text, variable_names):
        nonlocal calls
        calls += 1
        return original(spec_text, variable_names)

    monkeypatch.setattr(rtamt_backend, "_parse_specification", counted)
    backend = rtamt_backend.RtamtBackend("once[0,1] (mu)", ["mu"])
    assert backend.update(0, {"mu": 1.0}) == pytest.approx(1.0)
    assert backend.update(1, {"mu": 2.0}) == pytest.approx(2.0)
    backend.reset()
    assert backend.update(0, {"mu": -3.0}) == pytest.approx(-3.0)
    assert calls == 2


@pytest.mark.parametrize(
    "formula",
    [
        f"{ATOM_A} since {ATOM_B}",
        f"{ATOM_A} since[2,6] {ATOM_B}",
        f"historically[2,6]({ATOM_A})",
    ],
)
def test_native_and_rtamt_resets_restore_origin_semantics(formula):
    outputs = {}
    for backend_name in ("native", "rtamt"):
        compiled = parse_stl(formula, dt=1.0, backend=backend_name)
        backend = compiled.make_backend()
        for step, (left, right) in enumerate(
            [(2.0, -1.0), (-3.0, 5.0), (4.0, -2.0)]
        ):
            backend.update(step, {"mu_0": left, "mu_1": right})
        backend.reset()
        outputs[backend_name] = backend.update(
            0, {"mu_0": 7.0, "mu_1": -4.0}
        )

    assert outputs["native"] == pytest.approx(outputs["rtamt"])
