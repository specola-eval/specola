"""Declaration-backed RoboCasa task roles.

Covers the role contract of
docs/superpowers/specs/2026-07-28-robocasa-task-roles.md:

- the packaged role table loads, is schema-checked, and answers exact-name
  lookups only (sim-free);
- declared-name reading and declaration/entity reconciliation are pure and
  fail loudly with both sides named (sim-free, fake env);
- entity discovery is declaration-driven, roles and bases are structured
  adapter metadata, role tags ride the property keys, generic ``object``
  binding survives, resets are stable, mismatches fail, and no
  success/reward API is called (live, marked ``robocasa``).
"""

from __future__ import annotations

import numpy as np
import pytest

from specola.adapters import robocasa as rc

ROLE_VOCABULARY = {"target", "distractor", "container", "auxiliary"}
BASIS_VOCABULARY = {"source_adjudicated", "structural_derivation"}

# Audited facts for the live task (Task 4 artifact,
# benchmarks/diagnostic_catalogue/evidence/robocasa_role_audit.json):
# PickPlaceCounterToCabinet declares obj (target), distr_counter and
# distr_cab (distractors), no dynamic constructs, no conflicts.
LIVE_TASK = "PickPlaceCounterToCabinet"


# ---------------------------------------------------------------------------
# Packaged role table (sim-free)
# ---------------------------------------------------------------------------


class TestRoleTable:
    def test_loads_with_expected_schema(self):
        table = rc._load_role_table()
        assert table["schema_version"] == rc._ROLE_TABLE_SCHEMA_VERSION
        tasks = table["tasks"]
        assert table["task_count"] == len(tasks)
        assert len(tasks) >= 350

    def test_closed_vocabularies_and_no_globs(self):
        table = rc._load_role_table()
        for task, entry in table["tasks"].items():
            for name, info in entry["roles"].items():
                assert info["role"] in ROLE_VOCABULARY, (task, name)
                assert info["basis"] in BASIS_VOCABULARY, (task, name)
                assert "*" not in name, (task, name)
            # A conflict has no role: it awaits adjudication.
            for name in entry["conflicts"]:
                assert name not in entry["roles"], (task, name)

    def test_exact_lookup_only(self):
        entry = rc.task_role_entry(LIVE_TASK)
        assert entry is not None
        assert entry["roles"]["obj"] == {
            "role": "target",
            "basis": "source_adjudicated",
            "provenance": (
                "robocasa/environments/kitchen/atomic/"
                "kitchen_pick_place.py:73-74 identifies the target; "
                "lines 96-123 identify both distractors"
            ),
        }
        assert entry["roles"]["distr_counter"]["role"] == "distractor"
        assert entry["roles"]["distr_cab"]["role"] == "distractor"
        # No prefix/substring matching exists: a name that merely looks
        # like a distractor is absent, and absence means no role.
        assert "distr_counter_extra" not in entry["roles"]
        assert rc.task_role_entry("NoSuchRoboCasaTask") is None


# ---------------------------------------------------------------------------
# Declared-name reading and reconciliation (sim-free, fake env)
# ---------------------------------------------------------------------------


def _bare_adapter(cfgs):
    adapter = object.__new__(rc.RoboCasaAdapter)

    class _FakeEnv:
        object_cfgs = cfgs

    adapter.env = _FakeEnv()
    # capabilities reads the geometry-pair records __init__ would have set.
    adapter._geom_pair_records = None
    return adapter


class TestDeclaredNames:
    def test_names_read_in_declaration_order(self):
        adapter = _bare_adapter([
            {"name": "obj"}, {"name": "distr_counter"},
            {"name": "distr_cab"},
        ])
        assert adapter._declared_object_names() == [
            "obj", "distr_counter", "distr_cab",
        ]

    def test_no_declarations_reads_empty(self):
        adapter = _bare_adapter(None)
        assert adapter._declared_object_names() == []

    def test_unnamed_cfg_fails_loudly(self):
        adapter = _bare_adapter([{"name": "obj"}, {"info": {}}])
        with pytest.raises(rc.RoboCasaRoleError, match="name"):
            adapter._declared_object_names()

    def test_duplicate_declaration_fails_loudly(self):
        adapter = _bare_adapter([{"name": "obj"}, {"name": "obj"}])
        with pytest.raises(rc.RoboCasaRoleError, match="obj"):
            adapter._declared_object_names()


def test_gripper_joint_positions_use_robot_level_state_api():
    adapter = object.__new__(rc.RoboCasaAdapter)

    class _Robot:
        @staticmethod
        def get_gripper_joint_positions():
            return np.asarray([0.02, -0.02])

    assert adapter._gripper_joint_positions(_Robot()) == pytest.approx(
        [0.02, -0.02]
    )


class TestReconciliation:
    def test_exact_match_passes(self):
        mapping = rc.RoboCasaAdapter._reconcile_object_entities(
            ["obj", "distr_cab"],
            ["obj_main", "distr_cab_main"],
        )
        assert mapping == {
            "obj_main": "obj", "distr_cab_main": "distr_cab",
        }

    def test_missing_body_names_both_sides(self):
        with pytest.raises(rc.RoboCasaRoleError) as excinfo:
            rc.RoboCasaAdapter._reconcile_object_entities(
                ["obj", "ghost_object"],
                ["obj_main"],
            )
        message = str(excinfo.value)
        assert "ghost_object" in message
        assert "ghost_object_main" in message

    def test_undeclared_object_entity_names_both_sides(self):
        with pytest.raises(rc.RoboCasaRoleError) as excinfo:
            rc.RoboCasaAdapter._reconcile_object_entities(
                ["obj"],
                ["obj_main", "stray_main"],
            )
        assert "stray_main" in str(excinfo.value)


class TestDeclarationDrivenBodyRules:
    def test_rules_match_exactly_the_declared_bodies(self):
        adapter = _bare_adapter([
            {"name": "kettle"}, {"name": "distr_counter"},
        ])
        rules = adapter._default_body_rules()
        object_rules = [r for r in rules if r.cls == "object"]
        assert object_rules, "declaration-driven object rule missing"

        def matches(name):
            return any(r.matches(name) for r in object_rules)

        # Declared names match regardless of naming convention...
        assert matches("kettle_main")
        assert matches("distr_counter_main")
        # ...and nothing else does: no compiled-name heuristic remains.
        assert not matches("obj_main")
        assert not matches("kettle")
        assert not matches("distr_counter")

    def test_robot_rule_preserved(self):
        adapter = _bare_adapter([])
        rules = adapter._default_body_rules()
        robot_rules = [r for r in rules if r.cls == "robot"]
        assert any(r.matches("mobilebase0_base") for r in robot_rules)

    def test_fixture_refs_become_exact_subtree_rules(self):
        from types import SimpleNamespace

        adapter = _bare_adapter([])
        cabinet = SimpleNamespace(root_body="cabinet_root")
        adapter.env.fixture_refs = {
            "cab": (cabinet, False),
            "destination": cabinet,
        }
        rules = adapter._default_body_rules()
        fixture_rules = [r for r in rules if r.cls == "fixture"]
        assert len(fixture_rules) == 1
        assert fixture_rules[0].matches("cabinet_root")
        assert not fixture_rules[0].matches("cabinet_root_child")
        assert fixture_rules[0].joint_scope == "subtree"


class TestFixtureDeclarations:
    def test_tuple_and_bare_aliases_deduplicate(self):
        from types import SimpleNamespace

        adapter = _bare_adapter([])
        cabinet = SimpleNamespace(root_body="cabinet_root")
        adapter.env.fixture_refs = {
            "cab": (cabinet, True),
            "destination": cabinet,
        }
        assert adapter._normalized_fixture_refs() == [
            (cabinet, ("cab", "destination"), "cabinet_root")
        ]

    def test_fixture_without_root_fails_loudly(self):
        from types import SimpleNamespace

        adapter = _bare_adapter([])
        adapter.env.fixture_refs = {"cab": SimpleNamespace()}
        with pytest.raises(rc.RoboCasaRoleError, match="root_body"):
            adapter._normalized_fixture_refs()

    def test_distinct_fixture_refs_cannot_share_one_root(self):
        from types import SimpleNamespace

        adapter = _bare_adapter([])
        adapter.env.fixture_refs = {
            "cab": SimpleNamespace(root_body="cabinet_root"),
            "destination": SimpleNamespace(root_body="cabinet_root"),
        }
        with pytest.raises(
            rc.RoboCasaRoleError,
            match="distinct fixture instances",
        ):
            adapter._normalized_fixture_refs()

    def test_scene_fixture_and_ref_cannot_share_root_across_instances(self):
        from types import SimpleNamespace

        adapter = _bare_adapter([])
        adapter.env.fixtures = {
            "cab": SimpleNamespace(root_body="cabinet_root"),
        }
        adapter.env.fixture_refs = {
            "destination": SimpleNamespace(root_body="cabinet_root"),
        }
        with pytest.raises(
            rc.RoboCasaRoleError,
            match="distinct fixture instances",
        ):
            adapter._normalized_fixtures()


# ---------------------------------------------------------------------------
# Contact body ownership (sim-free) — D20 Part 1
# ---------------------------------------------------------------------------


class TestBodyOwnershipMap:
    # parentid is indexed by body ID; worldbody 0 is its own parent.
    # 0 <- 1 (obj main) <- 2 <- 3, and 4 is an unrelated child of 0.

    def test_subtree_bodies_inherit_root_role(self):
        parentid = [0, 0, 1, 2, 0]
        roles = rc._body_roles_by_ancestry(
            parentid, {1: "distractor"}
        )
        assert roles == {1: "distractor", 2: "distractor",
                         3: "distractor"}

    def test_nearest_object_ancestor_wins_when_nested(self):
        # 0 <- 1 (target main) <- 2 (container main) <- 3
        parentid = [0, 0, 1, 2]
        roles = rc._body_roles_by_ancestry(
            parentid, {1: "target", 2: "container"}
        )
        assert roles == {1: "target", 2: "container", 3: "container"}

    def test_unowned_bodies_absent(self):
        parentid = [0, 0, 1]
        assert rc._body_roles_by_ancestry(parentid, {}) == {}


class TestContactPairTags:
    @staticmethod
    def _fake_adapter(robot_ids, role_by_body):
        from types import SimpleNamespace

        return SimpleNamespace(
            _robot_body_ids=frozenset(robot_ids),
            _contact_role_by_body=dict(role_by_body),
        )

    def _tags(self, fake, pair):
        return rc.RoboCasaAdapter._contact_pair_tags(fake, pair)

    def test_robot_distractor_pair(self):
        fake = self._fake_adapter({7}, {3: "distractor"})
        assert self._tags(fake, (3, 7)) == (
            "role:distractor", "role:robot",
        )

    def test_target_fixture_pair_tags_only_the_target(self):
        fake = self._fake_adapter({7}, {2: "target"})
        # Body 9 is fixture/world: no role, no tag.
        assert self._tags(fake, (2, 9)) == ("role:target",)

    def test_untagged_pair_is_empty(self):
        fake = self._fake_adapter({7}, {})
        assert self._tags(fake, (10, 11)) == ()

    def test_body_ids_only_no_label_parsing(self):
        # The mapping is keyed by body ID; a body whose *name* happens
        # to look like a distractor contributes nothing unless its ID
        # is in the ownership map. (Guards the banned compiled-name
        # heuristic.)
        fake = self._fake_adapter(set(), {})
        assert self._tags(fake, (1, 2)) == ()

    def test_declared_tag_vocabulary_covers_every_emittable_tag(self):
        adapter = _bare_adapter([])
        # capabilities() reads the configured rule/class surface, which
        # object.__new__ skipped; a minimal configuration suffices.
        adapter.body_rules = ()
        adapter.regions = ()
        adapter.gripper_class = "gripper"
        vocabulary = adapter._contact_tag_vocabulary()
        assert vocabulary == {
            "role:target", "role:distractor", "role:container",
            "role:auxiliary", "role:robot",
        }
        capabilities = adapter.capabilities
        assert (
            capabilities.component_tags_by_class["contact"]
            == vocabulary
        )
        assert (
            capabilities.component_tags_by_class["robot_contact"]
            == vocabulary
        )


# ---------------------------------------------------------------------------
# Live adapter behavior (robocasa)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live():
    pytest.importorskip("robocasa")
    import robosuite as suite

    env = suite.make(
        env_name=LIVE_TASK,
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
    )
    env.reset()
    adapter = rc.RoboCasaAdapter(env)
    adapter.reset()
    yield env, adapter
    env.close()


@pytest.mark.robocasa
class TestLiveRoles:
    def test_discovery_is_declaration_driven(self, live):
        env, adapter = live
        declared = [cfg["name"] for cfg in env.object_cfgs]
        object_bodies = {
            body.body_name
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        assert object_bodies == {f"{n}_main" for n in declared}

    def test_roles_and_bases_are_structured_metadata(self, live):
        env, adapter = live
        roles_by_body = {
            body.body_name: adapter.entity_roles.get(body.entity_id)
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        assert roles_by_body.get("obj_main") == "target"
        assert roles_by_body.get("distr_counter_main") == "distractor"
        assert roles_by_body.get("distr_cab_main") == "distractor"
        # Two separately bindable declared distractors (D19 requirement).
        distractor_ids = [
            eid for eid, role in adapter.entity_roles.items()
            if role == "distractor"
        ]
        assert len(distractor_ids) >= 2
        assert set(adapter.entity_role_basis) == set(adapter.entity_roles)
        assert set(adapter.entity_role_provenance) == set(
            adapter.entity_roles
        )
        assert all(
            basis in BASIS_VOCABULARY
            for basis in adapter.entity_role_basis.values()
        )

    def test_object_declarations_are_structured_metadata(self, live):
        env, adapter = live
        by_body = {
            body.body_name: body.entity_id
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        for cfg in env.object_cfgs:
            entity_id = by_body["{}_main".format(cfg["name"])]
            assert adapter.entity_declared_names[entity_id] == cfg["name"]
            assert (
                adapter.entity_declaration_sources[entity_id]
                == "env.object_cfgs[name={!r}]".format(cfg["name"])
            )
            category = cfg.get("info", {}).get("cat")
            if category is not None:
                assert (
                    adapter.entity_declared_categories[entity_id]
                    == category
                )

    def test_fixture_refs_are_entities_with_provenance(self, live):
        env, adapter = live
        declared = {
            root: (fixture, names, refs)
            for fixture, names, refs, root
            in adapter._normalized_fixtures()
        }
        by_body = {
            body.body_name: body.entity_id
            for body in adapter._bodies
            if body.entity_class == "fixture"
        }
        assert set(by_body) == set(declared)
        for root, (fixture, names, refs) in declared.items():
            entity_id = by_body[root]
            assert adapter.entity_fixture_names[entity_id] == names
            assert adapter.entity_fixture_refs[entity_id] == refs
            assert (
                adapter.entity_fixture_types[entity_id]
                == type(fixture).__name__
            )
            assert root in adapter.entity_fixture_provenance[entity_id]
            assert adapter.entity_declaration_sources[entity_id].startswith(
                "env.fixtures"
            )

    def test_automatic_regions_are_live_geom_sources(self, live):
        env, adapter = live
        assert adapter.semantic_policy.regions
        compiled_geoms = set(adapter._names("geom"))
        for region in adapter.semantic_policy.regions:
            assert region.source == "geom"
            assert region.source_name in compiled_geoms

    def test_only_object_entities_carry_roles(self, live):
        env, adapter = live
        object_ids = {
            body.entity_id
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        assert set(adapter.entity_roles) <= object_ids

    def test_role_tags_ride_property_keys(self, live):
        env, adapter = live
        adapter.get_observation()
        by_body = {
            body.body_name: body.entity_id for body in adapter._bodies
        }
        for body_name, expected in (
            ("distr_counter_main", "role:distractor"),
            ("obj_main", "role:target"),
        ):
            meta = adapter.property_metadata(by_body[body_name])
            assert meta, body_name
            for key, entry in meta.items():
                assert expected in entry.tags, (body_name, key)

    def test_group_and_tag_namespaces_do_not_collide(self, live):
        env, adapter = live
        adapter.get_observation()
        role_tags = {
            "role:target", "role:distractor",
            "role:container", "role:auxiliary",
        }
        for body in adapter._bodies:
            meta = adapter.property_metadata(body.entity_id)
            for key, entry in meta.items():
                assert entry.group not in role_tags, (body.body_name, key)

    def test_reset_stability(self, live):
        env, adapter = live
        before = {
            body.body_name: adapter.entity_roles.get(body.entity_id)
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        adapter.reset()
        after = {
            body.body_name: adapter.entity_roles.get(body.entity_id)
            for body in adapter._bodies
            if body.entity_class == "object"
        }
        assert before == after

    def test_declaration_mismatch_fails_loudly(self, live):
        env, adapter = live
        original = env.object_cfgs
        env.object_cfgs = list(original) + [{"name": "ghost_object"}]
        try:
            with pytest.raises(rc.RoboCasaRoleError) as excinfo:
                adapter.reset()
            assert "ghost_object" in str(excinfo.value)
        finally:
            env.object_cfgs = original
            adapter.reset()

    def test_no_success_or_reward_api_called(self, live):
        env, adapter = live

        def _forbidden(*args, **kwargs):
            raise AssertionError(
                "adapter touched a success/reward API"
            )

        saved = env._check_success, env.reward
        env._check_success = _forbidden
        env.reward = _forbidden
        try:
            adapter.reset()
            adapter.get_observation()
        finally:
            env._check_success, env.reward = saved
