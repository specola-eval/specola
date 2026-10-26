"""Declared outcomes of the geometry distance binding per producer build.

MuJoCo 3.8.1 and 3.10.0 clamp a beyond-cutoff pair to the cutoff. 3.3.0
and 3.3.1 do not: they return a sentinel, and 3.3.1 also the unclamped
distance. The
outcome for that case is declared only where the build can produce it, so
the manifests of the clamping builds are unchanged (issue #128).
"""

import hashlib

import numpy as np
import pytest

from specola.adapters import mujoco_geom_pairs as pairs
from specola.adapters.mujoco_queries import GeomDistanceResult, reviewed_custody
from specola.native_values import NativeValue

CLAMPING = ("native_below_cutoff", "native_cutoff", "inferred_above_cutoff")
ABOVE = "native_above_cutoff"


def test_configuration_identity_digests_saved_model_bytes_then_configuration():
    mujoco = pytest.importorskip("mujoco")
    from specola.query_config import QueryConfiguration

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    configuration = QueryConfiguration.empty()
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    expected = hashlib.sha256(buffer.tobytes() + configuration.sha256.encode("ascii")).hexdigest()
    assert pairs.configuration_identity(mujoco, model, configuration).sha256 == expected


def test_clamping_builds_keep_their_three_outcomes():
    assert tuple(o.outcome_id for o in pairs.return_outcomes(0.05)) == CLAMPING
    assert tuple(o.outcome_id for o in pairs.witness_outcomes()) == CLAMPING
    assert pairs.return_outcomes(0.05) == pairs.return_outcomes(0.05, clamps=True)


def test_non_clamping_builds_declare_the_above_cutoff_outcome():
    outcomes = {o.outcome_id: o for o in pairs.return_outcomes(0.05, clamps=False)}
    assert set(outcomes) == set(CLAMPING) | {ABOVE}
    above = outcomes[ABOVE]
    assert [(c.relation, c.argument) for c in above.constraints] == [("ge", 0.05)]
    assert {p.key for p in above.projections} == set(
        pairs.GEOM_PAIR_PROPERTY_KEYS
    )
    witnesses = {o.outcome_id for o in pairs.witness_outcomes(clamps=False)}
    assert witnesses == set(CLAMPING) | {ABOVE}


def test_custody_rows_say_whether_the_build_clamps():
    assert reviewed_custody("3.8.1")["clamps"] is True
    assert reviewed_custody("3.10.0")["clamps"] is True
    assert reviewed_custody("3.3.0")["clamps"] is False
    assert reviewed_custody("3.3.1")["clamps"] is False


def _result(returned):
    witness = NativeValue.capture(np.zeros(6))
    return GeomDistanceResult((0, 1), (0, 1), 0.05, returned, witness,
                              NativeValue.capture({}), True)


def test_sentinel_and_unclamped_returns_project_to_the_cutoff():
    for returned in (10000000000.0, 1.7976931348623157e+308, 0.4415):
        result = _result(returned)
        assert result.outcome == ABOVE
        assert pairs.scalar_values(result) == {
            pairs.CAPPED_KEY: 0.05, pairs.CLIPPED_KEY: 1.0,
        }


def test_availability_text_exists_for_every_declared_outcome():
    for outcome in pairs.return_outcomes(0.05, clamps=False):
        assert pairs._AVAILABILITY[outcome.outcome_id]
