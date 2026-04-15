"""Unit tests for the DandiSelector Panel component."""

import os

import pytest
from dandi.exceptions import NotFoundError

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

    def get_dandiset(self, dandiset_id, version=None):
        return self.dandisets_by_id[dandiset_id]


@pytest.fixture
def patched_client(monkeypatch):
    FakeDandiAPIClient.dandisets_by_id = {
        "000971": FakeDandiset(
            "000971",
            [
                "sub-01/ses-1_behavior.nwb",
                "sub-01/ses-2_behavior.nwb",
                "sub-02/ses-1_behavior.nwb",
                "README.md",
            ],
        ),
        "000001": FakeDandiset("000001", ["sub-a/data.nwb"]),
    }
    monkeypatch.setattr(dandi_selector_module, "DandiAPIClient", FakeDandiAPIClient)
    return FakeDandiAPIClient


@pytest.fixture
def selector(panel_extension, patched_client, tmp_path):
    return DandiSelector(mirror_parent=str(tmp_path / "mirror"))


class TestDandiSelector:
    def test_constructs_empty(self, selector):
        assert selector.dandiset_input.value == ""
        assert selector.selected_uris == []
        assert selector._current_mirror_root is None

    def test_dandiset_change_builds_mirror(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        assert mirror_root is not None
        assert os.path.isdir(os.path.join(mirror_root, "sub-01"))
        assert os.path.isdir(os.path.join(mirror_root, "sub-02"))
        assert os.path.isfile(os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb"))
        assert os.path.isfile(os.path.join(mirror_root, "sub-01", "ses-2_behavior.nwb"))
        assert os.path.isfile(os.path.join(mirror_root, "sub-02", "ses-1_behavior.nwb"))
        # Placeholders are zero bytes.
        assert os.path.getsize(os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb")) == 0
        # README.md (non-NWB) is filtered out.
        assert not os.path.exists(os.path.join(mirror_root, "README.md"))
        assert "3 NWB asset" in selector.status.object

    def test_file_selector_is_scoped_to_dandiset(self, selector):
        selector.dandiset_input.value = "000971"
        assert selector.asset_file_selector.root_directory == selector._current_mirror_root
        assert selector.asset_file_selector.directory == selector._current_mirror_root

    def test_selected_uris_translates_absolute_paths(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [
            os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb"),
            os.path.join(mirror_root, "sub-02", "ses-1_behavior.nwb"),
        ]
        assert selector.selected_uris == [
            "dandi://000971/sub-01/ses-1_behavior.nwb",
            "dandi://000971/sub-02/ses-1_behavior.nwb",
        ]

    def test_non_nwb_selections_filtered(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        # A folder path sneaking into .value should be ignored.
        selector.asset_file_selector.value = [
            os.path.join(mirror_root, "sub-01"),
            os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb"),
        ]
        assert selector.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_switching_dandiset_repoints_selector(self, selector):
        selector.dandiset_input.value = "000971"
        first_root = selector._current_mirror_root
        selector.dandiset_input.value = "000001"
        second_root = selector._current_mirror_root
        assert first_root != second_root
        assert os.path.isfile(os.path.join(second_root, "sub-a", "data.nwb"))
        assert selector.asset_file_selector.root_directory == second_root
        # Prior selection cleared on dandiset change.
        assert selector.asset_file_selector.value == []

    def test_clearing_dandiset_clears_selections(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb")]
        selector.dandiset_input.value = ""
        assert selector._current_mirror_root is None
        assert selector.selected_uris == []

    def test_selected_uris_empty_when_no_dandiset(self, selector):
        assert selector.selected_uris == []

    def test_output_root_returns_first_selected(self, selector, tmp_path):
        assert selector.output_root is None
        selector.output_root_selector.value = [str(tmp_path)]
        assert selector.output_root == str(tmp_path)

    def test_directory_path_input_hidden(self, selector):
        assert selector.asset_file_selector._directory.visible is False

    @pytest.mark.parametrize("malformed", ["abc", "12345", "1234567", "#000971", "000971a"])
    def test_malformed_id_shows_warning_and_skips_api(self, selector, tmp_path, monkeypatch, malformed):
        calls = []

        class TrackingClient(dandi_selector_module.DandiAPIClient):
            def __enter__(self_inner):
                calls.append("called")
                return self_inner

        monkeypatch.setattr(dandi_selector_module, "DandiAPIClient", TrackingClient)

        selector.dandiset_input.value = malformed

        assert "\u26a0\ufe0f" in selector.status.object
        assert "Invalid Dandiset ID" in selector.status.object
        assert calls == []
        assert selector._current_mirror_root is None
        # No <id>/ folder was created.
        assert not os.path.isdir(os.path.join(selector._mirror_parent, malformed))

    def test_not_found_shows_warning_and_no_stale_folder(self, selector, patched_client, monkeypatch):
        def raising_get_dandiset(self_inner, dandiset_id, version=None):
            raise NotFoundError(f"Dandiset {dandiset_id} not found")

        monkeypatch.setattr(patched_client, "get_dandiset", raising_get_dandiset)

        selector.dandiset_input.value = "999999"

        assert "\u26a0\ufe0f" in selector.status.object
        assert "not found" in selector.status.object
        assert selector._current_mirror_root is None
        assert not os.path.isdir(os.path.join(selector._mirror_parent, "999999"))

    def test_recovery_after_error(self, selector, patched_client):
        selector.dandiset_input.value = "abc"
        assert "Invalid" in selector.status.object

        selector.dandiset_input.value = "000971"
        assert "\u2705" in selector.status.object
        assert selector._current_mirror_root is not None
        assert selector.asset_file_selector.root_directory == selector._current_mirror_root

        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [os.path.join(mirror_root, "sub-01", "ses-1_behavior.nwb")]
        assert selector.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_widget_swapped_on_load(self, selector):
        original = selector.asset_file_selector
        selector.dandiset_input.value = "000971"
        assert selector.asset_file_selector is not original
        assert selector.asset_file_selector.root_directory.endswith("/000971")
        # The slot holds exactly the current widget.
        assert list(selector._asset_file_selector_slot) == [selector.asset_file_selector]

    def test_widget_swapped_on_reset(self, selector):
        selector.dandiset_input.value = "000971"
        loaded_widget = selector.asset_file_selector
        selector.dandiset_input.value = ""
        assert selector.asset_file_selector is not loaded_widget
        assert selector.asset_file_selector.root_directory == selector._mirror_parent
        assert list(selector._asset_file_selector_slot) == [selector.asset_file_selector]
