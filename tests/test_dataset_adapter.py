"""Sim-free dataset adapter: CSV trajectories → observations → monitors."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from specola.adapters.dataset import DatasetAdapter
from specola.capabilities import reset_adapter
from specola.engine import MonitorEngine


CSV = textwrap.dedent("""\
    time,entity_id,entity_class,px,py,pz,battery
    0.0,1,robot,0.0,0.0,0.5,0.9
    0.0,2,object,1.0,0.0,0.1,
    0.1,1,robot,0.1,0.0,0.5,0.8
    0.1,2,object,1.0,0.0,0.1,
    0.2,1,robot,0.2,0.0,0.5,0.7
    0.2,2,object,1.0,0.0,0.1,
    0.3,1,robot,0.3,0.0,0.5,0.6
    0.3,2,object,1.0,0.0,0.1,
""")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "traj.csv"
    path.write_text(CSV)
    return str(path)


def test_frames_dt_velocities_and_properties(csv_path):
    adapter = DatasetAdapter(csv_path)

    assert adapter.n_frames == 4
    assert adapter.dt == pytest.approx(0.1)
    assert adapter.spatial_dims == 3

    obs0 = adapter.reset()
    assert obs0.perceived_entities[1].properties["battery"] == 0.9
    assert "battery" not in obs0.perceived_entities[2].properties

    obs1 = adapter.get_observation()
    # finite-differenced velocity: (0.1 - 0.0) / 0.1
    assert obs1.perceived_entities[1].velocity[0] == pytest.approx(1.0)
    assert obs1.perceived_entities[2].velocity[0] == pytest.approx(0.0)


def test_engine_monitors_dataset(csv_path, tmp_path):
    specs = tmp_path / "specs.yaml"
    specs.write_text(
        "specs:\n"
        "- name: battery_ok\n"
        "  formula: \"G[0,2](robot_*[battery] >= 0.5)\"\n"
    )
    adapter = DatasetAdapter(csv_path)
    reset = reset_adapter(adapter)
    engine = MonitorEngine.load(str(specs))
    engine.start_episode(
        dt=adapter.dt,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )

    result = engine.update(reset.initial_observation)
    for _ in range(adapter.n_frames - 1):
        result = engine.update(adapter.get_observation())

    evaluation = result.evaluations[0]
    # min(battery) over last 2 samples at t=0.3: min(0.7, 0.6) - 0.5
    assert evaluation.robustness == pytest.approx(0.1)
    assert evaluation.verdict_final


def test_cli_end_to_end(csv_path, tmp_path):
    specs = tmp_path / "specs.yaml"
    specs.write_text(
        "specs:\n"
        "- name: battery_ok\n"
        "  formula: \"G[0,2](robot_*[battery] >= 0.5)\"\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "dataset", "--task", csv_path,
         "--specs", str(specs)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_advancer_plays_the_file_and_refuses_replay_and_demo(csv_path):
    """Issue #132: the CSV is the recording. random and hold both play it;
    replay and --demo are refused instead of accepted and ignored."""
    adapter = DatasetAdapter(csv_path)
    with pytest.raises(SystemExit, match="not supported by the dataset adapter"):
        adapter.make_advancer("replay", None, 0)
    for mode in ("replay", "random", "hold"):
        with pytest.raises(SystemExit, match="The dataset adapter has no --demo replay"):
            adapter.make_advancer(mode, "does-not-exist.hdf5", 0)

    for mode in ("random", "hold"):
        adapter.reset()
        advance = adapter.make_advancer(mode, None, 0)
        played = 0
        while advance(played):
            adapter.get_observation()
            played += 1
        assert played == adapter.n_frames - 1


def test_extraction_scope_projects_recorded_properties_exactly(csv_path):
    """Recorded values survive unchanged while unrequested keys are absent."""
    adapter = DatasetAdapter(csv_path)

    def frames():
        observation = adapter.reset()
        rows = [
            {
                entity_id: dict(entity.properties)
                for entity_id, entity in
                observation.perceived_entities.items()
            }
        ]
        for _ in range(adapter.n_frames - 1):
            observation = adapter.get_observation()
            rows.append({
                entity_id: dict(entity.properties)
                for entity_id, entity in
                observation.perceived_entities.items()
            })
        return rows

    before = frames()
    adapter.set_extraction_scope({"robot": frozenset({"battery"})})
    after = frames()

    assert before[0][1]["battery"] == 0.9
    assert after[0][1] == {"battery": 0.9}
    assert after[1][1] == {"battery": 0.8}
    assert after[0][2] == {}


def test_position_columns_are_properties_now(csv_path):
    adapter = DatasetAdapter(csv_path)
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    assert entity.properties["px"] == pytest.approx(entity.position[0])
    assert "battery" in entity.properties


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("time, time ,entity_class,px,py", "duplicate normalized header"),
        ("time,,entity_class,px,py", "empty header column"),
    ],
)
def test_rejects_invalid_normalized_headers(tmp_path, header, message):
    path = tmp_path / "invalid_header.csv"
    path.write_text(
        header + "\n"
        "0.0,1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(ValueError, match=message):
        DatasetAdapter(str(path))


def test_rejects_duplicate_time_entity_rows(tmp_path):
    path = tmp_path / "duplicate_row.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0.0,1,robot,0.0,0.0\n"
        "0.0,1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(ValueError, match=r"duplicate \(time, entity_id\)"):
        DatasetAdapter(str(path))


def test_duplicate_detection_uses_numeric_time_and_entity_identity(tmp_path):
    path = tmp_path / "duplicate_numeric_alias.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0,01,robot,0.0,0.0\n"
        "0.0,+1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(ValueError, match=r"duplicate \(time, entity_id\)"):
        DatasetAdapter(str(path))


def test_rejects_frame_times_returning_to_an_earlier_time(tmp_path):
    path = tmp_path / "out_of_order.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0.0,1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
        "0.0,2,object,1.0,0.0\n"
    )

    with pytest.raises(ValueError, match="out-of-order frame times"):
        DatasetAdapter(str(path))


def test_rejects_nonuniform_frame_times(tmp_path):
    path = tmp_path / "nonuniform.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0.0,1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
        "0.3,1,robot,0.3,0.0\n"
    )

    with pytest.raises(ValueError, match="nonuniform frame times"):
        DatasetAdapter(str(path))


def test_uniform_decimal_spacing_is_independent_of_large_time_offset(tmp_path):
    path = tmp_path / "large_offset_uniform.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "1000000000.000,1,robot,0.0,0.0\n"
        "1000000000.001,1,robot,0.1,0.0\n"
        "1000000000.002,1,robot,0.2,0.0\n"
    )

    adapter = DatasetAdapter(str(path))

    assert adapter.dt == pytest.approx(0.001)


def test_nonuniform_subpicosecond_decimal_spacing_is_rejected(tmp_path):
    path = tmp_path / "tiny_nonuniform.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0,1,robot,0.0,0.0\n"
        "1e-15,1,robot,0.1,0.0\n"
        "3e-15,1,robot,0.2,0.0\n"
    )

    with pytest.raises(ValueError, match="nonuniform frame times"):
        DatasetAdapter(str(path))


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        ("0.0,1,robot,0.0,0.0,extra", "row 2 has extra cells"),
        ("0.0,1,robot,0.0", "row 2 is missing column 'py'"),
    ],
)
def test_rejects_ragged_rows_with_row_context(tmp_path, bad_row, message):
    path = tmp_path / "ragged.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        + bad_row
        + "\n0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(ValueError, match=message):
        DatasetAdapter(str(path))


@pytest.mark.parametrize("time_value", ["nan", "inf", "-inf", "bad"])
def test_rejects_invalid_time_with_path_row_and_column_context(
    tmp_path, time_value
):
    path = tmp_path / "invalid_time.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        f"{time_value},1,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(
        ValueError,
        match=r"invalid_time\.csv.*row 2 column 'time'.*(finite|decimal)",
    ):
        DatasetAdapter(str(path))


def test_rejects_invalid_entity_id_with_path_row_and_column_context(tmp_path):
    path = tmp_path / "invalid_id.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "0.0,one,robot,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0\n"
    )

    with pytest.raises(
        ValueError,
        match=r"invalid_id\.csv.*row 2 column 'entity_id'.*integer",
    ):
        DatasetAdapter(str(path))


def test_finite_decimal_time_must_be_representable_as_monitor_float(tmp_path):
    path = tmp_path / "decimal_overflow.csv"
    path.write_text(
        "time,entity_id,entity_class,px,py\n"
        "1e1000000,1,robot,0.0,0.0\n"
        "2e1000000,1,robot,0.1,0.0\n"
    )

    with pytest.raises(
        ValueError,
        match=r"decimal_overflow\.csv.*finite, strictly increasing binary floats",
    ):
        DatasetAdapter(str(path))


def test_whitespace_headers_and_equal_time_distinct_entities_are_valid(tmp_path):
    path = tmp_path / "normalized.csv"
    path.write_text(
        " time , entity_id , entity_class , px , py , vx , vy \n"
        "0.0,1,robot,0.0,0.0,1.0,0.0\n"
        "0.0,2,object,1.0,0.0,0.0,0.0\n"
        "0.1,1,robot,0.1,0.0,1.0,0.0\n"
        "0.1,2,object,1.0,0.0,0.0,0.0\n"
    )

    adapter = DatasetAdapter(str(path))
    observation = adapter.reset()

    assert set(observation.perceived_entities) == {1, 2}
    assert observation.perceived_entities[1].velocity[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "velocity_columns",
    ["vx", "vx,vy,vz", "pz,vx,vy"],
)
def test_rejects_incomplete_velocity_column_group(tmp_path, velocity_columns):
    path = tmp_path / "incomplete_velocity.csv"
    values = ",".join(
        "0.0" for _ in velocity_columns.split(",")
    )
    path.write_text(
        "time,entity_id,entity_class,px,py," + velocity_columns + "\n"
        "0.0,1,robot,0.0,0.0," + values + "\n"
        "0.1,1,robot,0.1,0.0," + values + "\n"
    )

    with pytest.raises(ValueError, match="incomplete velocity column group"):
        DatasetAdapter(str(path))


CSV_EXPLICIT_VELOCITY = textwrap.dedent("""\
    time,entity_id,entity_class,px,py,pz,vx,vy,vz,battery
    0.0,1,robot,0.0,0.0,0.5,1.0,0.0,0.0,0.9
    0.0,2,object,1.0,0.0,0.1,0.0,0.0,0.0,
    0.1,1,robot,0.1,0.0,0.5,1.0,0.0,0.0,0.8
    0.1,2,object,1.0,0.0,0.1,0.0,0.0,0.0,
""")


@pytest.fixture
def csv_explicit_velocity_path(tmp_path):
    path = tmp_path / "traj_explicit_velocity.csv"
    path.write_text(CSV_EXPLICIT_VELOCITY)
    return str(path)


def test_explicit_velocity_columns_are_properties(csv_explicit_velocity_path):
    adapter = DatasetAdapter(csv_explicit_velocity_path)

    obs0 = adapter.reset()
    robot0 = obs0.perceived_entities[1]
    assert robot0.properties["vx"] == pytest.approx(robot0.velocity[0])
    assert robot0.properties["vy"] == pytest.approx(robot0.velocity[1])
    assert robot0.properties["vz"] == pytest.approx(robot0.velocity[2])
    assert robot0.velocity[0] == pytest.approx(1.0)

    obs1 = adapter.get_observation()
    robot1 = obs1.perceived_entities[1]
    assert robot1.properties["vx"] == pytest.approx(robot1.velocity[0])
    assert robot1.properties["vy"] == pytest.approx(robot1.velocity[1])
    assert robot1.properties["vz"] == pytest.approx(robot1.velocity[2])


CSV_2D = textwrap.dedent("""\
    time,entity_id,entity_class,px,py,battery
    0.0,1,robot,0.0,0.0,0.9
    0.0,2,object,1.0,0.0,
    0.1,1,robot,0.1,0.0,0.8
    0.1,2,object,1.0,0.0,
""")


@pytest.fixture
def csv_2d_path(tmp_path):
    path = tmp_path / "traj_2d.csv"
    path.write_text(CSV_2D)
    return str(path)


def test_2d_dataset_exposes_only_2d_properties(csv_2d_path):
    adapter = DatasetAdapter(csv_2d_path)
    assert adapter.spatial_dims == 2

    obs0 = adapter.reset()
    robot0 = obs0.perceived_entities[1]
    assert robot0.properties["px"] == pytest.approx(robot0.position[0])
    assert robot0.properties["py"] == pytest.approx(robot0.position[1])
    assert robot0.properties["vx"] == pytest.approx(robot0.velocity[0])
    assert robot0.properties["vy"] == pytest.approx(robot0.velocity[1])
    assert "pz" not in robot0.properties
    assert "vz" not in robot0.properties
    # Frame 0 has no prior frame, so derived velocity is zero.
    assert robot0.velocity[0] == pytest.approx(0.0)
    assert robot0.velocity[1] == pytest.approx(0.0)
    assert robot0.properties["vx"] == pytest.approx(0.0)
    assert robot0.properties["vy"] == pytest.approx(0.0)

    obs1 = adapter.get_observation()
    robot1 = obs1.perceived_entities[1]
    assert robot1.properties["px"] == pytest.approx(robot1.position[0])
    assert robot1.properties["py"] == pytest.approx(robot1.position[1])
    assert robot1.properties["vx"] == pytest.approx(robot1.velocity[0])
    assert robot1.properties["vy"] == pytest.approx(robot1.velocity[1])
    assert "pz" not in robot1.properties
    assert "vz" not in robot1.properties
