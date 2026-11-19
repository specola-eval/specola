"""Discriminating computational states for the retained campaign requirements.

These states exercise the normal YAML/session path with retained producer
polygon coordinates. They do not claim to be native conveyor trajectories.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import yaml

# shapely ships with the RoboDojo environment, not with the core package.
pytest.importorskip("shapely")
from shapely.geometry import box, LineString  # noqa: E402
from shapely import affinity  # noqa: E402

from specola.adapter import MonitorObservation, PropertyMeta
from specola.entities import EntityState, TrackState
from specola.session import Monitor
from specola.spec_functions import FunctionInputs, immutable
from test_spec_functions import InputsAdapter

FIXTURES = Path(__file__).parent / 'fixtures/robodojo_functions'
AFFECTED = ('native_target_on_basket_bottom', 'goal', 'native_target_on_basket_bottom_once')


class FootprintAdapter(InputsAdapter):
    function_input_schemas = {'robodojo.footprints': 1}
    supported_classes = frozenset({'rigid'})

    def __init__(self):
        super().__init__()
        self.inner = box(-.03, -.03, .03, .03)
        self.outer = box(-.18, -.12, .18, .12)
        self.zmin, self.zmax = .70, .76
        self.missing = False
        self.duplicate = False

    def _read(self):
        records = {}
        for entity_id, label, polygon in ((2, 'basket', self.outer), (3, 'target', self.inner)):
            records[label] = {'entity_id': entity_id, 'entity_class': 'rigid', 'label': label,
                              'wkb': None if self.missing and entity_id == 3 else polygon.wkb}
        if self.duplicate:
            records['duplicate'] = dict(records['basket'], entity_id=4)
        return MonitorObservation(self.index, self.index * self.dt, {
            2: EntityState(entity_id=2, entity_class='rigid', position=np.zeros(3), track_age=self.index,
                           track_state=TrackState.CONFIRMED, properties={'footprint_zmin': .69, 'footprint_zmax': .91}),
            3: EntityState(entity_id=3, entity_class='rigid', position=np.zeros(3), track_age=self.index,
                           track_state=TrackState.CONFIRMED, properties={'footprint_zmin': self.zmin, 'footprint_zmax': self.zmax,
                                                                       'native_pos_2': .73, 'reset_pos_2': .60}),
        }, function_inputs=FunctionInputs(self.revision, immutable({'robodojo.footprints': records})))

    def property_metadata(self, entity_id):
        return {key: PropertyMeta(group='native producer operands', unit='m')
                for key in self._read().perceived_entities[entity_id].properties}


def _document(tmp_path):
    document = yaml.safe_load((FIXTURES / 'pick_from_conveyor_by_image.yaml').read_text())
    document['specs'] = [spec for spec in document['specs'] if spec['name'] in AFFECTED]
    for declaration in document['functions'].values():
        declaration['python'] = str(Path(__file__).resolve().parents[1] / 'examples/robodojo_functions.py') + ':footprint_containment'
    path = tmp_path / 'containment.yaml'
    path.write_text(yaml.safe_dump(document))
    return path


def test_historical_formula_text_and_constants_are_retained():
    for name, pin in json.loads((FIXTURES / 'provenance.json').read_text()).items():
        specs = yaml.safe_load((FIXTURES / name).read_text())['specs']
        serialized = json.dumps(specs, sort_keys=True, separators=(',', ':')).encode()
        assert len(specs) == pin['formulas']
        assert hashlib.sha256(serialized).hexdigest() == pin['specs_json_sha256']


@pytest.mark.parametrize('x,expected', [(0.0, .4), (.15, 0.0), (.16, -.2)])
def test_basket_contains_target_inclusive_boundary_and_robustness(tmp_path, x, expected):
    adapter = FootprintAdapter()
    adapter.inner = affinity.translate(adapter.inner, xoff=x)
    session = Monitor(str(_document(tmp_path)), results_dir=None).watch(adapter)
    result = session.step()
    assert set(result._rho_by_spec) == set(AFFECTED)
    for name in AFFECTED:
        assert result._rho_by_spec[name] == pytest.approx(expected, abs=1e-14)


def test_height_conjunct_still_rejects_xy_contained_target(tmp_path):
    adapter = FootprintAdapter()
    adapter.zmin, adapter.zmax = .75, .81
    result = Monitor(str(_document(tmp_path)), results_dir=None).watch(adapter).step()
    assert result._rho_by_spec['native_target_on_basket_bottom'] == pytest.approx(-.4)


def test_changing_polygon_vertices_missing_input_and_reset(tmp_path):
    adapter = FootprintAdapter()
    monitor = Monitor(str(_document(tmp_path)), results_dir=None)
    session = monitor.watch(adapter)
    session.step()
    retained = adapter.last_observation.function_inputs.values['robodojo.footprints']['target']['wkb']
    adapter.inner = affinity.rotate(box(-.10, -.03, .10, .03), 35)
    session.step()
    assert adapter.last_observation.function_inputs.values['robodojo.footprints']['target']['wkb'] != retained
    adapter.inner = LineString([(0, 0), (.1, 0)])
    gap = session.step()
    assert 'native_target_on_basket_bottom' not in gap._rho_by_spec
    assert all(ev.robustness is None for ev in gap.raw.evaluations)
    adapter.inner = box(-.03, -.03, .03, .03)
    assert monitor.watch(adapter).step()._rho_by_spec['native_target_on_basket_bottom'] == pytest.approx(.4)
    adapter.missing = True
    assert 'native_target_on_basket_bottom' not in monitor._session.step()._rho_by_spec


def test_duplicate_outer_label_is_rejected(tmp_path):
    adapter = FootprintAdapter()
    adapter.duplicate = True
    with pytest.raises(ValueError, match='expected one outer label'):
        Monitor(str(_document(tmp_path)), results_dir=None).watch(adapter)
