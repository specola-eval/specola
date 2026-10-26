"""Shared test configuration.

The accepted query amendment (docs/superpowers/specs/2026-09-05) makes an
explicit geometry distance cutoff a prerequisite for observing any
MuJoCo-family adapter. Tests that exercise other behavior install this
fixed configuration on construction; a test that needs the unconfigured
path installs ``QueryConfiguration.empty()`` explicitly before ``reset``.
The cutoff is the measured safe range for the reviewed producer builds
(session capture 2026-09-07); it is a test fixture, not a product default.
"""

import pytest

from specola.query_config import QueryConfiguration

GEOMETRY_QUERIES = QueryConfiguration.from_document({
    "schema": "specola.queries.v1",
    "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": 0.05}}],
    "instances": [],
})


@pytest.fixture(autouse=True)
def default_geometry_queries(monkeypatch):
    try:
        from specola.adapters import mujoco as module
    except ImportError:
        yield
        return
    original = module.MujocoAdapter.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.query_configuration == QueryConfiguration.empty():
            self.set_query_configuration(GEOMETRY_QUERIES)

    monkeypatch.setattr(module.MujocoAdapter, "__init__", __init__)
    yield
