"""Turn the episode's own reset into a durable discovery document.

The subtlety this module exists for: a caller can reset its adapter
before ``Monitor.watch()`` (``specola run`` did, in ``load_sim()``, until
2026-10-12), and ``Monitor.watch()`` resets again, and *that* second
reset is the one ``MonitorEngine.start_episode()`` compiles against. For a simulator that
randomizes per reset the two describe different scenes — RoboCasa draws
a fresh kitchen each time, so five consecutive resets of the same task
produced five different object sets and five different instructions.

A sidecar written from the first reset would therefore name objects the
episode never spawned, and an offline check compiled against it would
resolve keys the run never had. Everything here is driven from the
second reset, handed over by ``Monitor.watch``'s private reset hook
before the compiled specs narrow the extraction scope. That reset is
catalog-only: entities whose reads are deferred to the first step (the
MuJoCo family's geometry pairs) are not in the document, by maintainer
decision of 2026-10-13, which took RoboCasa's ``--web`` startup from
146 s to the 24 s of a run without the dashboard.

The document is the unfiltered one — ``include_raw=True``, no CLI
filters — because a filter is a question the person asked once, and the
recording has to answer questions nobody has asked yet.
"""

from __future__ import annotations

import gzip
from typing import Optional


def discovery_payload(
    adapter,
    reset,
    *,
    simulator: str,
    task: Optional[str],
    effective_dt: float,
    task_description: Optional[str] = None,
) -> bytes:
    """Build and compress the episode's discovery document.

    ``reset`` is an ``AdapterResetResult`` from the episode's own reset.
    Its property metadata is reused rather than re-read: reading it
    again would ask a live adapter a second time, and the answer can
    differ once the extraction scope narrows what the adapter reports.

    Measured on RoboCasa PickPlaceCounterToCabinet: 55.13 MiB of JSON,
    0.73 s to dump and 0.18 s to gzip, arriving at 1.66 MiB — an eighth
    of the geometry payload this recording already writes once per
    episode.
    """
    from .discovery_filter import DiscoveryQuery, filter_discovery
    from .discovery_model import build_discovery_model
    from .discovery_serialize import discovery_json

    model = build_discovery_model(
        adapter,
        reset.initial_observation.perceived_entities,
        simulator=simulator,
        task=task,
        effective_dt=effective_dt,
        task_description=task_description,
        property_metadata=dict(reset.property_metadata or {}),
        capabilities=reset.capabilities,
        extraction=reset.initial_observation.extraction,
    )
    view = filter_discovery(model, DiscoveryQuery(include_raw=True))
    return gzip.compress(discovery_json(view).encode("utf-8"), 6)
