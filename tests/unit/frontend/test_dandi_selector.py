"""Unit tests for the DandiSelector Panel component."""

from unittest.mock import MagicMock

import pytest

from guppy.frontend import dandi_selector as dandi_selector_module
from guppy.frontend.dandi_selector import DandiSelector


class FakeAsset:
    def __init__(self, path):
        self.path = path


class FakeDandiset:
    def __init__(self, identifier, asset_paths):
        self.identifier = identifier
        self._asset_paths = asset_paths

    def get_assets(self):
        return [FakeAsset(path) for path in self._asset_paths]


class FakeDandiAPIClient:
    """In-memory stand-in for dandi.dandiapi.DandiAPIClient."""

    dandisets_by_id = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get_dandisets(self):
        return list(self.dandisets_by_id.values())

    def get_dandiset(self, dandiset_id, version=None):
        return self.dandisets_by_id[dandiset_id]


@pytest.fixture
def patched_client(monkeypatch):
    """Replace the DandiAPIClient symbol in the selector module with the fake."""
    FakeDandiAPIClient.dandisets_by_id = {
        "000971": FakeDandiset(
            "000971",
            ["sub-01/ses-1_behavior.nwb", "sub-02/ses-1_behavior.nwb", "README.md"],
        ),
        "000001": FakeDandiset("000001", ["sub-a/data.nwb"]),
    }
    monkeypatch.setattr(dandi_selector_module, "DandiAPIClient", FakeDandiAPIClient)
    return FakeDandiAPIClient


class TestDandiSelector:
    def test_constructs_with_empty_widgets(self, panel_extension, patched_client):
        selector = DandiSelector()
        assert selector.dandiset_input.options == []
        assert selector.asset_select.options == []
        assert selector.selected_uris == []

    def test_load_dandisets_populates_dropdown(self, panel_extension, patched_client):
        selector = DandiSelector()
        selector._on_load_dandisets(MagicMock())
        assert selector.dandiset_input.options == ["000001", "000971"]
        assert "Loaded 2 public dandisets" in selector.status.object

    def test_dandiset_change_filters_to_nwb_assets(self, panel_extension, patched_client):
        selector = DandiSelector()
        selector._on_load_dandisets(MagicMock())
        selector.dandiset_input.value = "000971"
        assert selector.asset_select.options == [
            "sub-01/ses-1_behavior.nwb",
            "sub-02/ses-1_behavior.nwb",
        ]
        assert "2 NWB asset" in selector.status.object

    def test_clearing_dandiset_clears_assets(self, panel_extension, patched_client):
        selector = DandiSelector()
        selector._on_load_dandisets(MagicMock())
        selector.dandiset_input.value = "000971"
        selector.dandiset_input.value = ""
        assert selector.asset_select.options == []

    def test_selected_uris_formats_correctly(self, panel_extension, patched_client):
        selector = DandiSelector()
        selector._on_load_dandisets(MagicMock())
        selector.dandiset_input.value = "000971"
        selector.asset_select.value = ["sub-01/ses-1_behavior.nwb", "sub-02/ses-1_behavior.nwb"]
        assert selector.selected_uris == [
            "dandi://000971/sub-01/ses-1_behavior.nwb",
            "dandi://000971/sub-02/ses-1_behavior.nwb",
        ]

    def test_selected_uris_empty_when_no_dandiset(self, panel_extension, patched_client):
        selector = DandiSelector()
        selector.asset_select.value = ["anything.nwb"]
        assert selector.selected_uris == []

    def test_output_root_returns_first_selected(self, panel_extension, patched_client, tmp_path):
        selector = DandiSelector()
        assert selector.output_root is None
        selector.output_root_selector.value = [str(tmp_path)]
        assert selector.output_root == str(tmp_path)
