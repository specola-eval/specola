"""
specola — STL runtime monitoring for multi-entity robotic systems.

Quick start (facade)
---------------------
For dropping the monitor into an existing eval loop, ``Monitor``/``Session``
wrap the pieces below into five lines:

    from specola import Monitor

    monitor = Monitor("specs.yaml")
    for _ in range(n_episodes):
        observation, info = env.reset()
        session = monitor.watch(env, meta={"task": task.name})
        for _ in range(horizon):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            result = session.step()
        record = session.end(success=env.check_success())

``watch()`` auto-detects the adapter for a raw env (or accepts an
already-constructed ``SimulatorAdapter``); ``session.end()`` writes one
Parquet file per episode when ``results_dir`` is set on ``Monitor`` and
PyArrow is available, or CSV otherwise. The caller remains responsible for
resetting the simulator before ``watch()``.

Quick start (engine)
---------------------
For lower-level control over episode lifecycle and adapter construction:

    from specola import MonitorEngine
    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = RoboCasaAdapter(env)

    # Discover and validate the initial monitor-facing observation.
    obs = adapter.reset()

    engine = MonitorEngine.load("specs.yaml")
    reset = reset_adapter(adapter)
    engine.start_episode(
        dt=adapter.dt,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )

    result = engine.update(reset.initial_observation)

    for _ in range(horizon):
        env.step(action)
        result = engine.update(adapter.get_observation())

Console scripts
---------------
    specola discover --sim robocasa --task PickPlaceCounterToCabinet
    specola check --sim robocasa --task PickPlaceCounterToCabinet --specs my_specs.yaml
    specola write-specs --sim robocasa --task PickPlaceCounterToCabinet
    specola run --sim robocasa --specs my_specs.yaml --mode random
    specola report results/ --by train_step,task
    specola view results/<run_id>/recording_000
"""

from __future__ import annotations

from .entities import (
    EntityState,
    TrackState,
    SpecConfigError,
)
from .adapter import (
    SimulatorAdapter,
    MonitorObservation,
)
from .stl_parser import (
    parse_stl,
    describe_formula,
)
from .monitor import (
    SpecTemplate,
    ParameterSpec,
    TemporalOp,
    MonitorInstance,
    IncrementalBindingManager,
    Binding,
    ParamKey,
    binding_label,
    enumerate_bindings,
)
from .engine import MonitorEngine
from .capabilities import (
    AdapterCapabilities,
    AdapterResetResult,
    VocabularyMode,
    reset_adapter,
)
from .results import BindingEvaluation, EngineResult, EvaluationStatus
from .batch import BatchEpisodeContext, BatchStepError, BatchedMonitorEngine
from .session import Monitor, Session, StepResult, EpisodeRecord


__version__ = "0.2.0"


__all__ = [
    # Entity representation
    "EntityState",
    "TrackState",
    "SpecConfigError",

    # Adapter interface
    "SimulatorAdapter",
    "MonitorObservation",

    # Formula parsing
    "parse_stl",
    "describe_formula",

    # Monitoring
    "SpecTemplate",
    "ParameterSpec",
    "TemporalOp",
    "MonitorInstance",
    "IncrementalBindingManager",
    "Binding",
    "ParamKey",
    "binding_label",
    "enumerate_bindings",

        # Engine, capabilities, structured results
    "MonitorEngine",
    "AdapterCapabilities",
    "AdapterResetResult",
    "VocabularyMode",
    "reset_adapter",
    "BindingEvaluation",
    "EngineResult",
    "EvaluationStatus",
    "BatchEpisodeContext",
    "BatchStepError",
    "BatchedMonitorEngine",

    # Researcher-facing facade
    "Monitor",
    "Session",
    "StepResult",
    "EpisodeRecord",
]
