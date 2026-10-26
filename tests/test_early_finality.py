"""
Early verdict finality: a decided verdict is reported as decided.

Reporting a settled answer the moment it settles is the point of runtime
monitoring, and specola used to withhold it until the temporal window
filled. The two safe cases are monotone:

- ``G[0,b]`` reports the running minimum over the available prefix while
  warming up. Once that minimum is negative it can only fall further
  before the window fills, so FAIL is settled.
- ``F[0,b]`` reports the running maximum. Once it is non-negative it can
  only rise before the window fills, so PASS is settled.

The converse never settles early: a satisfied ``G`` can still be violated
by a later sample, and an unsatisfied ``F`` can still be satisfied.

Scope is deliberately narrow. Only a top-level ``G``/``F`` with a zero
lower bound directly over a comparison qualifies, which is the shape the
guide teaches and the one whose prefix semantics both backends agree on.
Everything else keeps waiting for the window.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from specola.adapters.dataset import DatasetAdapter
from specola.capabilities import reset_adapter
from specola.engine import MonitorEngine


def _dataset(tmp_path: Path, values) -> DatasetAdapter:
    path = tmp_path / "trace.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "entity_id", "entity_class", "px", "py", "sig"])
        for index, value in enumerate(values):
            writer.writerow([index * 0.1, 1, "object", 0.0, 0.0, value])
    return DatasetAdapter(str(path))


def _run(tmp_path: Path, values, specs_yaml: str):
    """Return [{spec_name: (rho, verdict_final)}] per step."""
    spec_file = tmp_path / "specs.yaml"
    spec_file.write_text(specs_yaml)

    adapter = _dataset(tmp_path, values)
    engine = MonitorEngine.load(str(spec_file))
    reset = reset_adapter(adapter)
    engine.start_episode(
        dt=adapter.dt,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )

    steps = []
    observation = reset.initial_observation
    for index in range(len(values)):
        if observation is None:
            break
        result = engine.update(observation)
        steps.append({
            evaluation.spec_name: (
                evaluation.robustness,
                evaluation.verdict_final,
            )
            for evaluation in result.evaluations
        })
        if index + 1 < len(values):
            observation = adapter.get_observation()
    return steps


_ALWAYS = 'specs:\n- name: g\n  formula: "G[0,4](object_*[sig] >= 0)"\n'
_EVENTUALLY = 'specs:\n- name: f\n  formula: "F[0,4](object_*[sig] >= 0)"\n'


def test_always_is_final_the_step_it_is_violated(tmp_path):
    # Window is 4 samples, so without early finality nothing is final
    # before step 3. The violation lands at step 1 and cannot leave the
    # window before it fills, so the FAIL is settled there.
    steps = _run(tmp_path, [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], _ALWAYS)

    assert steps[0]["g"] == (pytest.approx(1.0), False)

    rho, final = steps[1]["g"]
    assert rho == pytest.approx(-1.0)
    assert final is True, "a violated G is decided the moment it violates"

    # ...and stays decided through the window filling
    assert steps[2]["g"][1] is True
    assert steps[3]["g"][1] is True


def test_eventually_is_final_the_step_it_is_satisfied(tmp_path):
    steps = _run(tmp_path, [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0], _EVENTUALLY)

    assert steps[0]["f"] == (pytest.approx(-1.0), False)

    rho, final = steps[1]["f"]
    assert rho == pytest.approx(1.0)
    assert final is True, "a satisfied F is decided the moment it is satisfied"

    assert steps[2]["f"][1] is True


def test_satisfied_always_still_waits_for_the_window(tmp_path):
    # The unsafe direction: a G that is satisfied so far can still be
    # violated by a later sample, so it must not settle early.
    steps = _run(tmp_path, [1.0, 1.0, 1.0, 1.0, 1.0], _ALWAYS)

    assert [steps[i]["g"][1] for i in range(4)] == [False, False, False, True]


def test_unsatisfied_eventually_still_waits_for_the_window(tmp_path):
    steps = _run(tmp_path, [-1.0, -1.0, -1.0, -1.0, -1.0], _EVENTUALLY)

    assert [steps[i]["f"][1] for i in range(4)] == [False, False, False, True]


def test_early_verdict_agrees_with_the_verdict_at_window_fill(tmp_path):
    """
    The guarantee that makes this sound: whatever is reported early is
    what the full window would have said. Asserting the sign is stable
    from the decision step through the fill step is what would catch a
    rule that fires on a value which can still cross zero.
    """
    steps = _run(tmp_path, [1.0, -1.0, 1.0, 1.0, 1.0], _ALWAYS)

    decided_at = next(i for i, s in enumerate(steps) if s["g"][1])
    assert decided_at == 1

    for index in range(decided_at, 4):
        assert steps[index]["g"][0] < 0, index


def test_backends_agree_on_when_a_verdict_settles(tmp_path):
    # native and rtamt compute the warm-up prefix identically, so an
    # early verdict must not depend on which backend ran.
    both = (
        'specs:\n'
        '- name: g_native\n'
        '  formula: "G[0,4](object_*[sig] >= 0)"\n'
        '  backend: native\n'
        '- name: g_rtamt\n'
        '  formula: "G[0,4](object_*[sig] >= 0)"\n'
        '  backend: rtamt\n'
    )
    steps = _run(tmp_path, [1.0, -1.0, 1.0, 1.0, 1.0], both)

    for index, step in enumerate(steps):
        assert step["g_native"][1] == step["g_rtamt"][1], index


def test_nested_window_does_not_settle_early(tmp_path):
    # Out of scope by design: the inner window's warm-up value is itself
    # provisional, so the outer running minimum is not a bound on the
    # settled answer. Waiting is the conservative choice.
    nested = (
        'specs:\n- name: nested\n'
        '  formula: "G[0,4](F[0,2](object_*[sig] >= 0))"\n'
    )
    steps = _run(tmp_path, [-1.0, -1.0, -1.0, -1.0, -1.0], nested)

    assert steps[0]["nested"][1] is False
    assert steps[1]["nested"][1] is False


@pytest.mark.parametrize("operator", ["prev", "rise", "fall"])
def test_event_operator_waits_for_previous_bounded_child_value(
    tmp_path, operator
):
    """Both child samples consumed by an edge operator must be settled."""
    nested = (
        "specs:\n- name: nested\n"
        f'  formula: "{operator}(historically[0,4]('
        'object_*[sig] >= 0))"\n'
    )
    steps = _run(tmp_path, [4.0, 3.0, 2.0, 1.0, 0.0], nested)

    # historically[0,4] first settles at t=3. At that boundary the
    # event operator still consumes its provisional t=2 child value.
    assert [step["nested"][1] for step in steps] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_non_temporal_spec_is_final_immediately(tmp_path):
    # Unchanged behavior, pinned here because it is the boundary case:
    # no window to fill, so nothing to settle early.
    bare = 'specs:\n- name: bare\n  formula: "object_*[sig] >= 0"\n'
    steps = _run(tmp_path, [1.0, 1.0], bare)

    assert steps[0]["bare"][1] is True


def test_lower_bound_qualifies_on_converted_steps_not_authored_text(tmp_path):
    """
    The gate is the *converted* lower bound, not the authored one.
    `bound_to_steps` rounds half-up, so `G[0.01s,...]` at dt=0.1 converts
    to a lower bound of 0 steps and both backends then evaluate a genuine
    zero-lower-bound window. Settling early there matches what is
    actually computed; gating on the authored text would refuse it for a
    difference that does not exist by evaluation time.
    """
    specs = (
        'specs:\n'
        '- name: authored_zero\n'
        '  formula: "G[0,0.4s](object_*[sig] >= 0)"\n'
        '- name: rounds_to_zero\n'
        '  formula: "G[0.01s,0.4s](object_*[sig] >= 0)"\n'
    )
    steps = _run(tmp_path, [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], specs)

    assert steps[1]["authored_zero"][1] is True
    assert steps[1]["rounds_to_zero"][1] is True


def test_nonzero_lower_bound_waits_for_the_window(tmp_path):
    """
    A window that does not reach back to the current sample is excluded.
    Before any sample has aged into it the backend reports an infinite
    sentinel rather than a running extremum, which is not a bound on
    anything: `G` reads +inf and `F` reads -inf. Neither can trip the
    settling rule as written, but the exclusion is what makes that a
    guarantee rather than a coincidence of sign.
    """
    specs = (
        'specs:\n'
        '- name: g_shifted\n'
        '  formula: "G[0.2s,0.5s](object_*[sig] >= 0)"\n'
    )
    steps = _run(tmp_path, [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], specs)

    # violated from step 3 once the sample ages in, still not final
    assert steps[3]["g_shifted"][0] < 0
    assert steps[3]["g_shifted"][1] is False
    # final only when the window fills
    assert steps[4]["g_shifted"][1] is True


def test_spec_template_positional_construction_still_means_what_it_did():
    """
    ``SpecTemplate`` is exported in ``specola.__all__``, so its field
    order is public. Adding ``settles_early`` anywhere but the end would
    silently re-map an existing positional construction: the backend name
    would land in the new field, the factory in ``backend``, and the
    desugared text in ``make_backend``, so the instance would later try
    to call a string.
    """
    from specola import SpecTemplate
    from specola.monitor import TemporalOp

    def predicate(binding, states):
        return 1.0

    def factory():
        return None

    template = SpecTemplate(
        [], predicate, TemporalOp.NONE, 1, 1.0, "x",
        {"mu_0": predicate}, "mu_0", None, None, False, None,
        "native", factory, "expanded",
    )

    assert template.backend == "native"
    assert template.make_backend is factory
    assert template.desugared_text == "expanded"
    assert template.settles_early is None


def test_zero_robustness_boundary(tmp_path):
    """
    The boundary where the settling rule would break if the comparisons
    were loosened. A `G` at exactly 0 is satisfied, not violated, so it
    must not settle: the next sample can still take it negative. An `F`
    at exactly 0 is satisfied, so it must settle.

    Both backends, because an early verdict must not depend on which one
    ran.
    """
    specs = (
        'specs:\n'
        '- name: g_native\n'
        '  formula: "G[0,4](object_*[sig] >= 0)"\n'
        '  backend: native\n'
        '- name: g_rtamt\n'
        '  formula: "G[0,4](object_*[sig] >= 0)"\n'
        '  backend: rtamt\n'
        '- name: f_native\n'
        '  formula: "F[0,4](object_*[sig] >= 0)"\n'
        '  backend: native\n'
        '- name: f_rtamt\n'
        '  formula: "F[0,4](object_*[sig] >= 0)"\n'
        '  backend: rtamt\n'
    )
    steps = _run(tmp_path, [0.0, -1.0, 1.0, 1.0, 1.0], specs)

    for name in ("g_native", "g_rtamt"):
        assert steps[0][name] == (pytest.approx(0.0), False), name
        # the following negative sample is what a `<= 0` rule would have
        # already committed to the wrong side of
        assert steps[1][name][1] is True, name

    for name in ("f_native", "f_rtamt"):
        rho, final = steps[0][name]
        assert rho == pytest.approx(0.0)
        assert final is True, name
