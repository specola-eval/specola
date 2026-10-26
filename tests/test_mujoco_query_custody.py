"""Custody table of the MuJoCo geometry distance reader (issue #128)."""

import re

import pytest

from specola.adapters import mujoco_queries
from specola.conformance.query_bindings import QueryBindingError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("version", ["3.3.0", "3.3.1", "3.8.1", "3.10.0"])
def test_reviewed_versions_carry_a_complete_custody_row(version):
    row = mujoco_queries.reviewed_custody(version)
    assert _COMMIT.match(row["source_commit"])
    for name in ("functions", "structs", "library", "support", "copy"):
        assert _SHA256.match(row[name]), name
    assert type(row["clamps"]) is bool


def test_reviewed_rows_do_not_share_digests():
    digests = [
        value
        for row in mujoco_queries._REVIEWED.values()
        for name, value in row.items() if type(value) is str
    ]
    assert len(digests) == len(set(digests))


@pytest.mark.parametrize("version", ["3.2.3", "3.9.0", "4.0.0", ""])
def test_unreviewed_version_is_refused_by_name(version):
    with pytest.raises(QueryBindingError) as refusal:
        mujoco_queries.reviewed_custody(version)
    message = str(refusal.value)
    assert "unresolved MuJoCo query effect contract" in message
    assert "3.10.0" in message and "3.8.1" in message
    if version:
        assert version in message


def test_iteration_limit_refuses_only_the_build_that_has_one():
    mujoco_queries.check_ccd_iterations("3.3.1", 50)
    mujoco_queries.check_ccd_iterations("3.3.1", 170)
    with pytest.raises(QueryBindingError, match="ccd_iterations=171"):
        mujoco_queries.check_ccd_iterations("3.3.1", 171)
    for version in ("3.3.0", "3.8.1", "3.10.0"):
        mujoco_queries.check_ccd_iterations(version, 10000)
