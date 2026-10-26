"""camera_names(): base default and mock adapter inheritance."""

from specola.adapter import SimulatorAdapter


def test_base_adapter_has_no_named_cameras():
    assert SimulatorAdapter.camera_names(object()) == ()


def test_mock_adapter_inherits_empty_camera_names():
    from specola.adapters.fake import _MockAdapter

    adapter = _MockAdapter()
    assert adapter.camera_names() == ()
