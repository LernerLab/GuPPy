"""Unit tests for the DandiSelector Panel component."""

from pathlib import Path

import pytest
from dandi.exceptions import NotFoundError

from guppy.frontend.dandi_selector import DandiSelector
from guppy.utils.dandi_catalog import AssetSummary

from .test_dandi_browser import RecordingPreview, make_preview, make_probe

ASSETS_BY_ID = {
    "000971": [
        AssetSummary(asset_id="a", path="sub-01/ses-1_behavior.nwb", size_in_bytes=240_000),
        AssetSummary(asset_id="b", path="sub-01/ses-2_behavior.nwb", size_in_bytes=60_000_000),
        AssetSummary(asset_id="c", path="sub-02/ses-1_behavior.nwb", size_in_bytes=240_000),
    ],
    "000001": [AssetSummary(asset_id="d", path="sub-a/data.nwb", size_in_bytes=1_000)],
}


class RecordingAssetListing:
    """Stand-in for ``list_nwb_assets`` over a fixed in-memory archive."""

    def __init__(self, assets_by_id=None):
        self.assets_by_id = dict(ASSETS_BY_ID if assets_by_id is None else assets_by_id)
        self.calls = []

    def __call__(self, *, dandiset_id, **kwargs):
        self.calls.append(dandiset_id)
        if dandiset_id not in self.assets_by_id:
            raise NotFoundError(f"Dandiset {dandiset_id} not found")
        return list(self.assets_by_id[dandiset_id])


@pytest.fixture
def selector(panel_extension, tmp_path):
    return DandiSelector(
        mirror_parent=str(tmp_path / "mirror"),
        list_assets_function=RecordingAssetListing(),
        preview_function=RecordingPreview(),
    )


class TestDandiSelector:
    def test_constructs_empty(self, selector):
        assert selector.dandiset_input.value == ""
        assert selector.selected_uris == []
        assert selector._current_mirror_root is None

    def test_dandiset_change_builds_mirror(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-01").is_dir()
        assert (mirror_root / "sub-02").is_dir()
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").is_file()
        assert (mirror_root / "sub-01" / "ses-2_behavior.nwb").is_file()
        assert (mirror_root / "sub-02" / "ses-1_behavior.nwb").is_file()
        # Placeholders are zero bytes.
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").stat().st_size == 0
        assert "3 NWB asset" in selector.status.object

    def test_status_reports_the_asset_size_range(self, selector):
        selector.dandiset_input.value = "000971"
        assert "234 KB – 57.2 MB" in selector.status.object

    def test_file_selector_is_scoped_to_dandiset(self, selector):
        selector.dandiset_input.value = "000971"
        assert selector.asset_file_selector.root_directory == selector._current_mirror_root
        assert selector.asset_file_selector.directory == selector._current_mirror_root

    def test_selected_uris_translates_absolute_paths(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [
            str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"),
            str(Path(mirror_root) / "sub-02" / "ses-1_behavior.nwb"),
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
            str(Path(mirror_root) / "sub-01"),
            str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"),
        ]
        assert selector.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_switching_dandiset_repoints_selector(self, selector):
        selector.dandiset_input.value = "000971"
        first_root = selector._current_mirror_root
        selector.dandiset_input.value = "000001"
        second_root = selector._current_mirror_root
        assert first_root != second_root
        assert (Path(second_root) / "sub-a" / "data.nwb").is_file()
        assert selector.asset_file_selector.root_directory == second_root
        # Prior selection cleared on dandiset change.
        assert selector.asset_file_selector.value == []

    def test_clearing_dandiset_clears_selections(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]
        selector.dandiset_input.value = ""
        assert selector._current_mirror_root is None
        assert selector.selected_uris == []

    def test_selected_uris_empty_when_no_dandiset(self, selector):
        assert selector.selected_uris == []

    def test_selected_uris_empty_while_the_typed_id_has_no_tree(self, selector):
        # A malformed ID leaves its text in the field with no mirror behind it.
        selector.dandiset_input.value = "abc"
        assert selector.selected_uris == []

    def test_output_root_returns_first_selected(self, selector, tmp_path):
        assert selector.output_root is None
        selector.output_root_selector.value = [str(tmp_path)]
        assert selector.output_root == str(tmp_path)

    def test_directory_path_input_hidden(self, selector):
        assert selector.asset_file_selector._directory.visible is False

    @pytest.mark.parametrize("malformed", ["abc", "12345", "1234567", "#000971", "000971a"])
    def test_malformed_id_shows_warning_and_skips_api(self, selector, malformed):
        selector.dandiset_input.value = malformed

        assert "⚠️" in selector.status.object
        assert "Invalid Dandiset ID" in selector.status.object
        assert selector.list_assets_function.calls == []
        assert selector._current_mirror_root is None
        # No <id>/ folder was created.
        assert not (Path(selector._mirror_parent) / malformed).is_dir()

    def test_not_found_shows_warning_and_no_stale_folder(self, selector):
        selector.dandiset_input.value = "999999"

        assert "⚠️" in selector.status.object
        assert "not found" in selector.status.object
        assert selector._current_mirror_root is None
        assert not (Path(selector._mirror_parent) / "999999").is_dir()

    def test_recovery_after_error(self, selector):
        selector.dandiset_input.value = "abc"
        assert "Invalid" in selector.status.object

        selector.dandiset_input.value = "000971"
        assert "✅" in selector.status.object
        assert selector._current_mirror_root is not None
        assert selector.asset_file_selector.root_directory == selector._current_mirror_root

        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]
        assert selector.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_widget_swapped_on_load(self, selector):
        original = selector.asset_file_selector
        selector.dandiset_input.value = "000971"
        assert selector.asset_file_selector is not original
        assert selector.asset_file_selector.root_directory.endswith("000971")
        # The slot holds exactly the current widget.
        assert list(selector._asset_file_selector_slot) == [selector.asset_file_selector]

    def test_widget_swapped_on_reset(self, selector):
        selector.dandiset_input.value = "000971"
        loaded_widget = selector.asset_file_selector
        selector.dandiset_input.value = ""
        assert selector.asset_file_selector is not loaded_widget
        assert selector.asset_file_selector.root_directory == selector._mirror_parent
        assert list(selector._asset_file_selector_slot) == [selector.asset_file_selector]

    def test_asset_selection_watcher_fires_on_the_rebuilt_widget(self, selector):
        events = []
        selector.attach_asset_selection_watcher(callback=events.append)
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        asset_path = Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"
        events.clear()

        # The widget selected into here is the replacement built for this dandiset,
        # not the one the watcher was originally attached to.
        selector.asset_file_selector.value = [asset_path]

        assert len(events) == 1
        assert events[0].new == [asset_path]

    def test_asset_selection_watcher_is_notified_of_the_widget_swap(self, selector):
        events = []
        selector.attach_asset_selection_watcher(callback=events.append)

        selector.dandiset_input.value = "000971"

        # The swap drops the previous selection without firing a value event of its own.
        assert events == [None]


class TestDandiSelectorAssetFilters:
    def test_no_filters_mirrors_every_asset(self, selector):
        selector.dandiset_input.value = "000971"
        assert "Showing **3** of 3" in selector.asset_status.object

    def test_minimum_size_keeps_only_the_larger_assets(self, selector):
        selector.dandiset_input.value = "000971"
        selector.minimum_size_in_mb.value = 5.0
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-2_behavior.nwb").is_file()
        assert not (mirror_root / "sub-01" / "ses-1_behavior.nwb").exists()
        assert "Showing **1** of 3" in selector.asset_status.object

    def test_name_filter_narrows_the_tree(self, selector):
        selector.dandiset_input.value = "000971"
        selector.asset_name_filter.value = "sub-02"
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-02").is_dir()
        assert not (mirror_root / "sub-01").exists()

    def test_relaxing_a_filter_restores_the_assets_it_hid(self, selector):
        selector.dandiset_input.value = "000971"
        selector.minimum_size_in_mb.value = 5.0
        selector.minimum_size_in_mb.value = 0.0
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").is_file()
        assert "Showing **3** of 3" in selector.asset_status.object

    def test_filters_do_not_relist_the_dandiset(self, selector):
        selector.dandiset_input.value = "000971"
        selector.minimum_size_in_mb.value = 5.0
        selector.asset_name_filter.value = "sub-01"
        assert selector.list_assets_function.calls == ["000971"]

    def test_a_dandiset_with_no_assets_reports_no_size_range(self, selector):
        selector.list_assets_function.assets_by_id["000002"] = []
        selector.dandiset_input.value = "000002"
        assert selector.status.object == "✅ Dandiset 000002: 0 NWB asset(s) found."
        assert "Showing **0** of 0" in selector.asset_status.object

    def test_a_filter_change_without_a_dandiset_does_nothing(self, selector):
        selector.minimum_size_in_mb.value = 5.0
        assert selector._current_mirror_root is None
        assert selector.asset_status.object == ""


class TestDandiSelectorAssetPreview:
    def test_preview_streams_the_first_selected_asset(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-2_behavior.nwb")]

        selector.preview_selected_asset()

        assert selector.preview_function.calls == [
            {"dandiset_id": "000971", "asset_path": "sub-01/ses-2_behavior.nwb", "series_name": None}
        ]
        assert selector.asset_preview_pane.panel.visible is True
        assert "2 channel(s)" in selector.asset_preview_pane.summary.object
        assert "Previewed `sub-01/ses-2_behavior.nwb`" in selector.asset_status.object

    def test_preview_without_a_selection_warns(self, selector):
        selector.dandiset_input.value = "000971"
        selector.preview_selected_asset()
        assert selector.preview_function.calls == []
        assert "Select an NWB file" in selector.asset_status.object

    def test_preview_of_a_file_without_photometry_says_so(self, selector):
        selector.preview_function.preview = make_preview(probe=make_probe(has_photometry=False))
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]

        selector.preview_selected_asset()

        assert "no `FiberPhotometryResponseSeries`" in selector.asset_preview_pane.summary.object

    def test_changing_dandiset_clears_a_stale_preview(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-2_behavior.nwb")]
        selector.preview_selected_asset()

        selector.dandiset_input.value = "000001"

        assert selector.asset_preview_pane.panel.visible is False


class TestDandiSelectorBrowserIntegration:
    def test_browser_starts_collapsed(self, selector):
        assert selector.browser_card.collapsed is True

    def test_choosing_a_dandiset_in_the_browser_loads_its_assets(self, selector):
        selector.browser.on_dandiset_selected("000971")
        assert selector.dandiset_input.value == "000971"
        assert selector._current_mirror_root is not None
        assert "3 NWB asset" in selector.status.object

    def test_load_dandiset_is_what_the_browser_calls(self, selector):
        selector.load_dandiset("000001")
        assert selector.dandiset_input.value == "000001"
