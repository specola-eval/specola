"""Meta-World v3 task declarations, projected from the source audit.

This module deliberately has no Meta-World import and performs no simulator
construction.  The packaged map contains only the target, ordered object, and
orientation declarations extracted by
``benchmarks/metaworld_success_map.py``.  Criterion and runtime-diagnostic
records remain outside the runtime package.

Accessor expressions are evaluated by a small interpreter rather than
``eval``.  Its grammar is exactly the source-accessor grammar represented by
the committed Meta-World 3.1.1 audit.  A new expression shape therefore fails
loudly until the source extractor, packaged projection, interpreter, and tests
are reviewed together.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pkgutil import get_data
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


SUPPORTED_METAWORLD_VERSION = "3.1.1"
SOURCE_MAP_SCHEMA_VERSION = 1
SOURCE_ARTIFACT = "benchmarks/metaworld_success_criteria.json"
SOURCE_MAP_RESOURCE = "metaworld_v3_source_map.json"

_POSITION_KINDS = frozenset(
    {
        "body_com",
        "body_frame",
        "derived_expression",
        "geom_frame",
        "site",
        "site_helper",
    }
)
_ORIENTATION_REPRESENTATIONS = frozenset(
    {
        "constant_1_0_0_0_unattributed",
        "mujoco_wxyz",
        "scipy_xyzw",
        "zero_placeholder",
    }
)
_SEMANTIC_ORIENTATION_REPRESENTATIONS = frozenset(
    {"mujoco_wxyz", "scipy_xyzw"}
)
_NAMED_ACCESSORS = frozenset(
    {
        "self.get_body_com",
        "self._get_site_pos",
        "self.data.body",
        "self.data.geom",
        "self.data.joint",
        "self.data.site",
    }
)
_DATA_FIELDS = frozenset({"qpos", "xmat", "xpos", "xquat"})
_NUMPY_CALLS = frozenset(
    {"np.array", "np.cos", "np.sin", "np.zeros"}
)
_RECEIVER_CALLS = frozenset({"as_quat", "copy", "item", "reshape"})


class MetaworldDeclarationError(RuntimeError):
    """The packaged declaration or its live source no longer reconciles."""


@dataclass(frozen=True)
class SourceIdentity:
    """One named simulator source referenced by a declared accessor."""

    kind: str
    name: str


@dataclass(frozen=True)
class ObjectDeclaration:
    """One ordered three-vector returned by ``_get_pos_objects()``."""

    index: int
    declared_name: str
    kind: str
    accessor: str
    source_file: str
    source_line: int
    sources: Tuple[SourceIdentity, ...]

    @property
    def provenance(self) -> str:
        return "{}:{}: {}".format(
            self.source_file,
            self.source_line,
            self.accessor,
        )


@dataclass(frozen=True)
class OrientationDeclaration:
    """One ordered four-vector returned by ``_get_quat_objects()``."""

    index: int
    accessor: str
    representation: str
    is_zero_placeholder: bool
    source_file: str
    source_line: int
    sources: Tuple[SourceIdentity, ...]

    @property
    def has_semantic_orientation(self) -> bool:
        """Whether the channel names and represents a physical orientation."""
        return self.representation in _SEMANTIC_ORIENTATION_REPRESENTATIONS


@dataclass(frozen=True)
class TargetDeclaration:
    """The task target returned by ``_get_pos_goal()``."""

    accessor: str
    owner: str
    return_expression: str
    source_file: str
    source_line: int

    @property
    def provenance(self) -> str:
        return "{}:{}: {} owned by {}".format(
            self.source_file,
            self.source_line,
            self.accessor,
            self.owner,
        )


@dataclass(frozen=True)
class TaskDeclaration:
    """All source-declared monitor entities for one Meta-World v3 task."""

    task: str
    class_name: str
    objects: Tuple[ObjectDeclaration, ...]
    orientations: Tuple[OrientationDeclaration, ...]
    target: TargetDeclaration


def project_success_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the Task 2 artifact into runtime declaration-only data.

    This is the sole generation transform.  Tests compare its result with the
    packaged JSON, so the benchmark artifact remains the canonical source and
    a hand-edited runtime map cannot silently diverge.
    """
    tasks = artifact.get("tasks")
    if not isinstance(tasks, list):
        raise MetaworldDeclarationError(
            "Meta-World success artifact has no task list"
        )

    projected_tasks = []
    retained = (
        "class",
        "object_method",
        "objects",
        "orientation_method",
        "orientations",
        "target",
        "task",
    )
    for row in tasks:
        if not isinstance(row, Mapping):
            raise MetaworldDeclarationError(
                "Meta-World success artifact contains a non-mapping task"
            )
        try:
            projected_tasks.append({key: row[key] for key in retained})
        except KeyError as exc:
            raise MetaworldDeclarationError(
                "Meta-World task projection is missing {!r}".format(
                    exc.args[0]
                )
            ) from exc

    return {
        "metaworld_version": artifact.get("metaworld_version"),
        "schema_version": SOURCE_MAP_SCHEMA_VERSION,
        "source_artifact": SOURCE_ARTIFACT,
        "source_artifact_schema_version": artifact.get("schema_version"),
        "source_sha256": artifact.get("source_sha256"),
        "task_count": len(projected_tasks),
        "tasks": projected_tasks,
    }


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetaworldDeclarationError(
            "{} must be a non-empty string".format(context)
        )
    return value


def _require_positive_line(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MetaworldDeclarationError(
            "{} must be a positive integer".format(context)
        )
    return value


def _parse_sources(value: Any, context: str) -> Tuple[SourceIdentity, ...]:
    if not isinstance(value, list):
        raise MetaworldDeclarationError(
            "{} must be a list".format(context)
        )
    parsed = []
    for index, source in enumerate(value):
        item_context = "{}[{}]".format(context, index)
        if not isinstance(source, Mapping):
            raise MetaworldDeclarationError(
                "{} must be a mapping".format(item_context)
            )
        parsed.append(
            SourceIdentity(
                kind=_require_nonempty_string(
                    source.get("kind"), "{}.kind".format(item_context)
                ),
                name=_require_nonempty_string(
                    source.get("name"), "{}.name".format(item_context)
                ),
            )
        )
    return tuple(parsed)


def _parse_task(row: Mapping[str, Any]) -> TaskDeclaration:
    task = _require_nonempty_string(row.get("task"), "task")
    objects_value = row.get("objects")
    orientations_value = row.get("orientations")
    target_value = row.get("target")
    if not isinstance(objects_value, list) or not objects_value:
        raise MetaworldDeclarationError(
            "{}.objects must be a non-empty list".format(task)
        )
    if not isinstance(orientations_value, list):
        raise MetaworldDeclarationError(
            "{}.orientations must be a list".format(task)
        )
    if not isinstance(target_value, Mapping):
        raise MetaworldDeclarationError(
            "{}.target must be a mapping".format(task)
        )

    objects = []
    for index, value in enumerate(objects_value):
        context = "{}.objects[{}]".format(task, index)
        if not isinstance(value, Mapping):
            raise MetaworldDeclarationError(
                "{} must be a mapping".format(context)
            )
        declared_index = value.get("index")
        if declared_index != index:
            raise MetaworldDeclarationError(
                "{} index {!r} is not ordered index {}".format(
                    context, declared_index, index
                )
            )
        kind = _require_nonempty_string(
            value.get("kind"), "{}.kind".format(context)
        )
        if kind not in _POSITION_KINDS:
            raise MetaworldDeclarationError(
                "{} has unsupported position kind {!r}".format(context, kind)
            )
        sources = _parse_sources(
            value.get("sources"), "{}.sources".format(context)
        )
        if not sources:
            raise MetaworldDeclarationError(
                "{} has no named simulator source".format(context)
            )
        objects.append(
            ObjectDeclaration(
                index=index,
                declared_name=_require_nonempty_string(
                    value.get("declared_name"),
                    "{}.declared_name".format(context),
                ),
                kind=kind,
                accessor=_require_nonempty_string(
                    value.get("accessor"), "{}.accessor".format(context)
                ),
                source_file=_require_nonempty_string(
                    value.get("source_file"),
                    "{}.source_file".format(context),
                ),
                source_line=_require_positive_line(
                    value.get("source_line"),
                    "{}.source_line".format(context),
                ),
                sources=sources,
            )
        )

    orientations = []
    for index, value in enumerate(orientations_value):
        context = "{}.orientations[{}]".format(task, index)
        if not isinstance(value, Mapping):
            raise MetaworldDeclarationError(
                "{} must be a mapping".format(context)
            )
        declared_index = value.get("index")
        if declared_index != index:
            raise MetaworldDeclarationError(
                "{} index {!r} is not ordered index {}".format(
                    context, declared_index, index
                )
            )
        representation = _require_nonempty_string(
            value.get("representation"),
            "{}.representation".format(context),
        )
        if representation not in _ORIENTATION_REPRESENTATIONS:
            raise MetaworldDeclarationError(
                "{} has unsupported orientation representation {!r}".format(
                    context, representation
                )
            )
        is_zero = value.get("is_zero_placeholder")
        if not isinstance(is_zero, bool):
            raise MetaworldDeclarationError(
                "{}.is_zero_placeholder must be boolean".format(context)
            )
        if is_zero != (representation == "zero_placeholder"):
            raise MetaworldDeclarationError(
                "{} zero-placeholder flag disagrees with representation".format(
                    context
                )
            )
        orientations.append(
            OrientationDeclaration(
                index=index,
                accessor=_require_nonempty_string(
                    value.get("accessor"), "{}.accessor".format(context)
                ),
                representation=representation,
                is_zero_placeholder=is_zero,
                source_file=_require_nonempty_string(
                    value.get("source_file"),
                    "{}.source_file".format(context),
                ),
                source_line=_require_positive_line(
                    value.get("source_line"),
                    "{}.source_line".format(context),
                ),
                sources=_parse_sources(
                    value.get("sources"), "{}.sources".format(context)
                ),
            )
        )
        if (
            representation in _SEMANTIC_ORIENTATION_REPRESENTATIONS
            and not orientations[-1].sources
        ):
            raise MetaworldDeclarationError(
                "{} names a physical orientation without a simulator "
                "source".format(context)
            )

    if len(orientations) != len(objects):
        raise MetaworldDeclarationError(
            "{} declares {} position rows but {} orientation rows".format(
                task, len(objects), len(orientations)
            )
        )

    target = TargetDeclaration(
        accessor=_require_nonempty_string(
            target_value.get("accessor"), "{}.target.accessor".format(task)
        ),
        owner=_require_nonempty_string(
            target_value.get("owner"), "{}.target.owner".format(task)
        ),
        return_expression=_require_nonempty_string(
            target_value.get("return_expression"),
            "{}.target.return_expression".format(task),
        ),
        source_file=_require_nonempty_string(
            target_value.get("source_file"),
            "{}.target.source_file".format(task),
        ),
        source_line=_require_positive_line(
            target_value.get("source_line"),
            "{}.target.source_line".format(task),
        ),
    )
    if target.accessor != "self._get_pos_goal()":
        raise MetaworldDeclarationError(
            "{} target accessor is {!r}, expected self._get_pos_goal()".format(
                task, target.accessor
            )
        )

    names = [item.declared_name for item in objects]
    duplicates = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicates:
        raise MetaworldDeclarationError(
            "{} has duplicate declared object names {}".format(
                task, duplicates
            )
        )

    return TaskDeclaration(
        task=task,
        class_name=_require_nonempty_string(
            row.get("class"), "{}.class".format(task)
        ),
        objects=tuple(objects),
        orientations=tuple(orientations),
        target=target,
    )


@lru_cache(maxsize=1)
def _load_source_map() -> Tuple[TaskDeclaration, ...]:
    try:
        payload = get_data("specola.data", SOURCE_MAP_RESOURCE)
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map is unavailable"
        ) from exc
    if payload is None:
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map is unavailable"
        )
    text = payload.decode("utf-8")
    try:
        artifact = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map is invalid JSON"
        ) from exc
    if not isinstance(artifact, Mapping):
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map must be a mapping"
        )
    if artifact.get("schema_version") != SOURCE_MAP_SCHEMA_VERSION:
        raise MetaworldDeclarationError(
            "Unsupported Meta-World declaration-map schema {!r}".format(
                artifact.get("schema_version")
            )
        )
    if artifact.get("metaworld_version") != SUPPORTED_METAWORLD_VERSION:
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map targets version {!r}, "
            "expected {!r}".format(
                artifact.get("metaworld_version"),
                SUPPORTED_METAWORLD_VERSION,
            )
        )
    if artifact.get("source_artifact") != SOURCE_ARTIFACT:
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map has unexpected source"
        )
    rows = artifact.get("tasks")
    if not isinstance(rows, list):
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map has no task list"
        )
    tasks = tuple(_parse_task(row) for row in rows if isinstance(row, Mapping))
    if len(tasks) != len(rows):
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map contains a non-mapping task"
        )
    if artifact.get("task_count") != len(tasks):
        raise MetaworldDeclarationError(
            "Packaged Meta-World task count does not match task list"
        )
    if len(tasks) != 50:
        raise MetaworldDeclarationError(
            "Meta-World {} declaration map has {} tasks, expected the exact "
            "50-task v3 registry".format(
                SUPPORTED_METAWORLD_VERSION,
                len(tasks),
            )
        )
    object_count = sum(len(task.objects) for task in tasks)
    orientation_count = sum(len(task.orientations) for task in tasks)
    if (object_count, orientation_count) != (53, 53):
        raise MetaworldDeclarationError(
            "Meta-World declaration map has {} objects and {} orientations, "
            "expected 53 and 53".format(object_count, orientation_count)
        )
    source_digest = artifact.get("source_sha256")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_digest
        )
    ):
        raise MetaworldDeclarationError(
            "Packaged Meta-World declaration map has an invalid source digest"
        )
    names = [item.task for item in tasks]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise MetaworldDeclarationError(
            "Packaged Meta-World map has duplicate tasks {}".format(duplicates)
        )
    return tasks


def task_names() -> Tuple[str, ...]:
    """Return all task names in canonical source-map order."""
    return tuple(item.task for item in _load_source_map())


def task_declaration(task_name: str) -> TaskDeclaration:
    """Return the declaration for one exact Meta-World v3 registry key."""
    matches = [
        item for item in _load_source_map() if item.task == task_name
    ]
    if len(matches) != 1:
        raise MetaworldDeclarationError(
            "No exact Meta-World {} source declaration for task {!r}".format(
                SUPPORTED_METAWORLD_VERSION,
                task_name,
            )
        )
    return matches[0]


def task_name_for_environment(environment: Any) -> str:
    """Resolve a task only from the source-recorded environment class."""
    class_names = {
        "{}.{}".format(base.__module__, base.__name__)
        for base in type(environment).__mro__
    }
    matches = [
        declaration.task
        for declaration in _load_source_map()
        if declaration.class_name in class_names
    ]
    if len(matches) != 1:
        raise MetaworldDeclarationError(
            "Meta-World environment class {} resolves to {} source tasks: "
            "{}".format(
                "{}.{}".format(
                    type(environment).__module__,
                    type(environment).__name__,
                ),
                len(matches),
                matches,
            )
        )
    return matches[0]


def verify_environment_class(
    environment: Any,
    declaration: TaskDeclaration,
) -> None:
    """Reject a caller-supplied task name for a different environment class."""
    class_names = {
        "{}.{}".format(base.__module__, base.__name__)
        for base in type(environment).__mro__
    }
    if declaration.class_name not in class_names:
        raise MetaworldDeclarationError(
            "Meta-World task {!r} expects environment class {!r}, got {!r}".format(
                declaration.task,
                declaration.class_name,
                "{}.{}".format(
                    type(environment).__module__,
                    type(environment).__name__,
                ),
            )
        )


def verify_installed_version() -> None:
    """Require the exact producer version represented by the packaged map."""
    try:
        installed = metadata.version("metaworld")
    except metadata.PackageNotFoundError as exc:
        raise MetaworldDeclarationError(
            "Meta-World distribution metadata is unavailable"
        ) from exc
    if installed != SUPPORTED_METAWORLD_VERSION:
        raise MetaworldDeclarationError(
            "Meta-World source declarations support version {!r}; installed "
            "version is {!r}. Regenerate and review {} before using this "
            "adapter.".format(
                SUPPORTED_METAWORLD_VERSION,
                installed,
                SOURCE_ARTIFACT,
            )
        )


def _call_name(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _named_accessor(call: ast.Call, environment: Any) -> Any:
    name = _call_name(call.func)
    if (
        name not in _NAMED_ACCESSORS
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ):
        raise MetaworldDeclarationError(
            "Unsupported Meta-World named accessor {!r}".format(name)
        )
    source_name = call.args[0].value
    if name == "self.get_body_com":
        return environment.get_body_com(source_name)
    if name == "self._get_site_pos":
        return environment._get_site_pos(source_name)
    if name == "self.data.body":
        return environment.data.body(source_name)
    if name == "self.data.geom":
        return environment.data.geom(source_name)
    if name == "self.data.joint":
        return environment.data.joint(source_name)
    if name == "self.data.site":
        return environment.data.site(source_name)
    raise AssertionError("unreachable named accessor")


def _evaluate_node(
    node: ast.AST,
    environment: Any,
    rotation_type: Optional[Any],
) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [
            _evaluate_node(item, environment, rotation_type)
            for item in node.elts
        ]
    if isinstance(node, ast.Tuple):
        return tuple(
            _evaluate_node(item, environment, rotation_type)
            for item in node.elts
        )
    if isinstance(node, ast.Attribute):
        if (
            node.attr == "data"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return environment.data
        if (
            node.attr in _DATA_FIELDS
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) in _NAMED_ACCESSORS
        ):
            return getattr(_named_accessor(node.value, environment), node.attr)
        raise MetaworldDeclarationError(
            "Unsupported Meta-World accessor attribute {!r}".format(
                ast.dump(node, include_attributes=False)
            )
        )
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, environment, rotation_type)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise MetaworldDeclarationError(
            "Unsupported Meta-World unary operator {}".format(
                type(node.op).__name__
            )
        )
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, environment, rotation_type)
        right = _evaluate_node(node.right, environment, rotation_type)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise MetaworldDeclarationError(
            "Unsupported Meta-World binary operator {}".format(
                type(node.op).__name__
            )
        )
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in _NAMED_ACCESSORS:
            return _named_accessor(node, environment)
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise MetaworldDeclarationError(
                "Starred Meta-World accessor arguments are unsupported"
            )
        if node.keywords:
            raise MetaworldDeclarationError(
                "Keyword Meta-World accessor arguments are unsupported"
            )
        arguments = [
            _evaluate_node(argument, environment, rotation_type)
            for argument in node.args
        ]
        if name in _NUMPY_CALLS:
            return getattr(np, str(name).split(".", 1)[1])(*arguments)
        if name == "Rotation.from_matrix":
            if rotation_type is None:
                from scipy.spatial.transform import Rotation

                rotation_type = Rotation
            return rotation_type.from_matrix(*arguments)
        if isinstance(node.func, ast.Attribute):
            receiver_name = node.func.attr
            if receiver_name not in _RECEIVER_CALLS:
                raise MetaworldDeclarationError(
                    "Unsupported Meta-World receiver call {!r}".format(
                        receiver_name
                    )
                )
            receiver = _evaluate_node(
                node.func.value,
                environment,
                rotation_type,
            )
            return getattr(receiver, receiver_name)(*arguments)
        raise MetaworldDeclarationError(
            "Unsupported Meta-World accessor call {!r}".format(name)
        )
    raise MetaworldDeclarationError(
        "Unsupported Meta-World accessor node {}".format(
            type(node).__name__
        )
    )


def evaluate_accessor(
    expression: str,
    environment: Any,
    *,
    rotation_type: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate one reviewed accessor expression without Python ``eval``."""
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MetaworldDeclarationError(
            "Invalid Meta-World accessor expression {!r}".format(expression)
        ) from exc
    try:
        value = _evaluate_node(parsed.body, environment, rotation_type)
        result = np.asarray(value, dtype=float).reshape(-1)
    except MetaworldDeclarationError:
        raise
    except Exception as exc:
        raise MetaworldDeclarationError(
            "Meta-World accessor {!r} failed ({}: {})".format(
                expression, type(exc).__name__, exc
            )
        ) from exc
    if not np.all(np.isfinite(result)):
        raise MetaworldDeclarationError(
            "Meta-World accessor {!r} returned non-finite values".format(
                expression
            )
        )
    return result


def exact_vector(
    value: Any,
    *,
    size: int,
    context: str,
) -> np.ndarray:
    """Return a finite vector with an exact required component count."""
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise MetaworldDeclarationError(
            "{} is not a numeric vector ({}: {})".format(
                context, type(exc).__name__, exc
            )
        ) from exc
    if result.size != size:
        raise MetaworldDeclarationError(
            "{} has {} components, expected {}".format(
                context, result.size, size
            )
        )
    if not np.all(np.isfinite(result)):
        raise MetaworldDeclarationError(
            "{} contains non-finite values".format(context)
        )
    return result


def assert_exact_agreement(
    declared: np.ndarray,
    independently_evaluated: np.ndarray,
    *,
    context: str,
) -> None:
    """Fail without tolerance or nearest-source recovery on any disagreement."""
    if (
        declared.shape == independently_evaluated.shape
        and np.array_equal(declared, independently_evaluated)
    ):
        return
    max_error = (
        float("inf")
        if declared.shape != independently_evaluated.shape
        else float(np.max(np.abs(declared - independently_evaluated)))
    )
    raise MetaworldDeclarationError(
        "{} disagrees with its source accessor (shape {} versus {}, "
        "max absolute error {!r})".format(
            context,
            declared.shape,
            independently_evaluated.shape,
            max_error,
        )
    )


def normalize_orientation(
    value: np.ndarray,
    representation: str,
) -> Optional[np.ndarray]:
    """Return a physical orientation in MuJoCo ``wxyz`` order, or ``None``.

    Zero placeholders and the unattributed constant are intentionally not
    promoted into orientation state.  They reconcile as authored API output,
    but neither carries a named physical orientation source.
    """
    vector = exact_vector(
        value,
        size=4,
        context="Meta-World object orientation",
    )
    if representation == "mujoco_wxyz":
        return vector.copy()
    if representation == "scipy_xyzw":
        return vector[[3, 0, 1, 2]].copy()
    if representation in {
        "zero_placeholder",
        "constant_1_0_0_0_unattributed",
    }:
        return None
    raise MetaworldDeclarationError(
        "Unsupported orientation representation {!r}".format(representation)
    )
