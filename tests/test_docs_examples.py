"""Guide examples must stay executable.

Extracts every fenced ```yaml spec block from docs/writing-specs.md and
checks it against the real pipeline:

1. Every block must PARSE (syntax level, no simulator needed). This is
   the tier that catches documentation drift like an out-of-bracket
   quantifier spelling or a YAML-escaped backslash — both shipped in the
   guide's first draft and were only caught by manually running the
   examples through specola check.
2. Blocks whose vocabulary the mock simulator provides must also COMPILE
   semantically against its discovered catalog. Blocks using
   sim-specific keys (the robocasa examples) are reported as skipped at
   this tier, not failed — the guide documents real sims, and the mock
   scene cannot carry every simulator's key set.

Tier 2 skips only on vocabulary gaps (unknown class on the mock, a
selector matching no component, or a spec that binds nothing); every
other semantic error — quantifier placement, grammar misuse — fails the
guard. Mutation-verified: reintroducing the out-of-bracket quantifier
form flips the guard red.
"""

import re
from pathlib import Path

import pytest

from specola.adapters.fake import _MockAdapter
from specola.capabilities import reset_adapter
from specola.engine import MonitorEngine
from specola.entities import (
    SpecConfigError,
    UnboundStaticSlotError,
    UnknownClassError,
)

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "writing-specs.md"


def _spec_blocks():
    """Return (label, yaml_text) for every fenced yaml block with specs.

    Guide blocks after the first are written as bare spec-list excerpts
    (``- name: ...``) without the top-level ``specs:`` key; wrap those so
    every block loads through the real MonitorEngine.load path.
    """
    text = GUIDE.read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    labelled = []
    for index, block in enumerate(blocks):
        if "formula:" not in block:
            continue
        if "specs:" not in block:
            block = "specs:\n" + block
        match = re.search(r"name:\s*(\S+)", block)
        label = f"block{index}-{match.group(1) if match else 'unnamed'}"
        labelled.append((label, block))
    return labelled


_BLOCKS = _spec_blocks()


def test_guide_has_spec_blocks():
    """The extraction itself must not silently rot."""
    assert len(_BLOCKS) >= 3, [label for label, _ in _BLOCKS]


@pytest.mark.parametrize(
    "label,block", _BLOCKS, ids=[label for label, _ in _BLOCKS]
)
def test_guide_examples_parse(label, block, tmp_path):
    spec_file = tmp_path / "guide_block.yaml"
    spec_file.write_text(block, encoding="utf-8")
    MonitorEngine.load(str(spec_file))  # raises ValueError on any parse error


@pytest.mark.parametrize(
    "label,block", _BLOCKS, ids=[label for label, _ in _BLOCKS]
)
def test_guide_examples_compile_on_mock_where_possible(label, block, tmp_path):
    spec_file = tmp_path / "guide_block.yaml"
    spec_file.write_text(block, encoding="utf-8")
    engine = MonitorEngine.load(str(spec_file))
    reset = reset_adapter(_MockAdapter())
    try:
        engine.start_episode(
            dt=0.05,
            capabilities=reset.capabilities,
            catalog=reset.discovered_catalog,
        )
    except UnknownClassError as exc:
        pytest.skip(f"class not on mock: {exc}")
    except UnboundStaticSlotError as exc:
        # Sim-specific keys on a class the mock scene does have: the
        # example is fine, the mock simply carries other entities.
        pytest.skip(f"keys not on mock: {exc}")
    except SpecConfigError as exc:
        # Only vocabulary gaps are excusable on the mock scene. Any other
        # SpecConfigError (quantifier placement, grammar misuse, ...) is a
        # broken example and must fail. The three phrases below are the
        # ways a gap surfaces: a bare component, a pinned reference whose
        # entity lacks the key, and a selector no entity of the class
        # matches. Docs examples are written against a real simulator, so
        # all three are reachable here.
        vocabulary_gap = (
            "matched no component",
            "does not expose required components",
            "matches selector(s)",
        )
        if any(phrase in str(exc) for phrase in vocabulary_gap):
            pytest.skip(f"keys not on mock: {exc}")
        raise
    # A spec whose keys exist nowhere on the mock compiles but binds
    # nothing and evaluates nothing — that proves nothing about the
    # example, so it is a skip, not a pass.
    result = engine.update(reset.initial_observation)
    if not result.evaluations:
        pytest.skip("no eligible bindings on mock (sim-specific keys)")
