"""Unit tests for the DandiSelector Panel component."""

from pathlib import Path
from threading import Event

import pytest
from dandi.exceptions import NotFoundError

from guppy.frontend.dandi_selector import DandiSelector
from guppy.utils.dandi_catalog import AssetSummary

from .test_dandi_browser import RecordingPreview, make_preview, make_probe

ASSETS_BY_ID = {
    "000971": [
        AssetSummary(
            asset_id="a",
            path="sub-01/ses-1_behavior.nwb",
            size_in_bytes=240_000,
            content_url="url-a",
        ),
        AssetSummary(
            asset_id="b",
            path="sub-01/ses-2_behavior.nwb",
            size_in_bytes=60_000_000,
            content_url="url-b",
        ),
        AssetSummary(
            asset_id="c",
            path="sub-02/ses-1_behavior.nwb",
            size_in_bytes=240_000,
            content_url="url-c",
        ),
    ],
    "000001": [
        AssetSummary(
            asset_id="d",
            path="sub-a/data.nwb",
            size_in_bytes=1_000,
            content_url="url-d",
        )
    ],
}

# Which of the 000971 assets a scan reports as holding fiber photometry.
PHOTOMETRY_BY_PATH = {
    "sub-01/ses-1_behavior.nwb": False,
    "sub-01/ses-2_behavior.nwb": True,
    "sub-02/ses-1_behavior.nwb": False,
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


class RecordingScan:
    """Stand-in for ``scan_assets_for_photometry`` over a fixed set of verdicts.

    The scan blocks on ``gate`` so a test can hold it mid-flight and inspect what the widgets
    show while it runs, rather than only after it finishes.
    """

    def __init__(self, verdicts=None):
        self.verdicts = dict(PHOTOMETRY_BY_PATH if verdicts is None else verdicts)
        self.calls = []
        self.gate = Event()
        self.gate.set()

    def __call__(self, assets, progress_callback=None):
        self.calls.append([asset.path for asset in assets])
        self.gate.wait()
        verdicts = {}
        for asset in assets:
            verdicts[asset.path] = self.verdicts.get(asset.path, False)
            if progress_callback is not None:
                progress_callback(len(verdicts))
        return verdicts


@pytest.fixture
def selector(panel_extension, tmp_path):
    return DandiSelector(
        mirror_parent=str(tmp_path / "mirror"),
        list_assets_function=RecordingAssetListing(),
        preview_function=RecordingPreview(),
        scan_function=RecordingScan(),
    )


def run_scan(selector):
    """Press the scan button and drive the poller until the scan has been applied."""
    selector.scan_assets()
    selector._scan["thread"].join()
    selector._poll_scan()


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


class TestDandiSelectorPhotometryScan:
    def test_before_a_scan_every_asset_is_mirrored(self, selector):
        selector.dandiset_input.value = "000971"
        assert "Showing **3** of 3" in selector.asset_status.object

    def test_the_scan_button_waits_for_a_dandiset(self, selector):
        assert selector.scan_button.disabled
        selector.dandiset_input.value = "000971"
        assert not selector.scan_button.disabled

    def test_the_filter_waits_for_a_scan(self, selector):
        selector.dandiset_input.value = "000971"
        assert selector.photometry_only.disabled
        run_scan(selector)
        assert not selector.photometry_only.disabled

    def test_a_finished_scan_narrows_the_tree_to_the_photometry_assets(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-2_behavior.nwb").is_file()
        assert not (mirror_root / "sub-01" / "ses-1_behavior.nwb").exists()
        assert not (mirror_root / "sub-02").exists()

    def test_a_finished_scan_reports_what_it_found(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        assert selector.asset_status.object == (
            "Showing **1** of 3 NWB asset(s) in the tree below. " "Scanned 3 file(s): **1** hold fiber photometry."
        )

    def test_unchecking_the_filter_restores_the_assets_it_hid(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        selector.photometry_only.value = False
        mirror_root = Path(selector._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").is_file()
        assert "Showing **3** of 3" in selector.asset_status.object

    def test_the_progress_bar_tracks_the_running_scan(self, selector):
        selector.dandiset_input.value = "000971"
        selector.scan_function.gate.clear()
        selector.scan_assets()
        assert selector.scan_progress.visible
        assert selector.scan_progress.max == 3
        assert selector.scan_progress.value == 0
        assert selector.scan_button.loading
        assert selector.scan_button.disabled
        assert selector.asset_status.object == "Scanning 0 of 3 NWB asset(s) for fiber photometry..."

        selector.scan_function.gate.set()
        selector._scan["thread"].join()
        selector._poll_scan()
        assert not selector.scan_progress.visible
        assert not selector.scan_button.loading
        assert not selector.scan_button.disabled

    def test_polling_mid_scan_renders_the_count_without_applying_verdicts(self, selector):
        selector.dandiset_input.value = "000971"
        selector.scan_function.gate.clear()
        selector.scan_assets()
        selector._record_scan_progress(2)
        selector._poll_scan()
        assert selector.scan_progress.value == 2
        assert selector.asset_status.object == "Scanning 2 of 3 NWB asset(s) for fiber photometry..."
        assert selector._photometry_by_path == {}
        selector.scan_function.gate.set()
        selector._scan["thread"].join()

    def test_the_scan_reads_every_listed_asset(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        assert selector.scan_function.calls == [
            [
                "sub-01/ses-1_behavior.nwb",
                "sub-01/ses-2_behavior.nwb",
                "sub-02/ses-1_behavior.nwb",
            ]
        ]

    def test_scanning_does_not_relist_the_dandiset(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        selector.photometry_only.value = False
        assert selector.list_assets_function.calls == ["000971"]

    def test_switching_dandiset_forgets_the_previous_scan(self, selector):
        selector.dandiset_input.value = "000971"
        run_scan(selector)
        selector.dandiset_input.value = "000001"
        assert selector._photometry_by_path == {}
        assert selector.photometry_only.disabled
        assert not selector.photometry_only.value
        assert "Showing **1** of 1" in selector.asset_status.object

    def test_scanning_without_a_dandiset_does_nothing(self, selector):
        selector.scan_assets()
        assert selector.scan_function.calls == []
        assert selector._current_mirror_root is None
        assert selector.asset_status.object == ""

    def test_a_dandiset_with_no_assets_reports_no_size_range(self, selector):
        selector.list_assets_function.assets_by_id["000002"] = []
        selector.dandiset_input.value = "000002"
        assert selector.status.object == "✅ Dandiset 000002: 0 NWB asset(s) found."
        assert "Showing **0** of 0" in selector.asset_status.object
        assert selector.scan_button.disabled


class TestDandiSelectorAssetPreview:
    def test_preview_streams_the_first_selected_asset(self, selector):
        selector.dandiset_input.value = "000971"
        mirror_root = selector._current_mirror_root
        selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-2_behavior.nwb")]

        selector.preview_selected_asset()

        assert selector.preview_function.calls == [
            {
                "dandiset_id": "000971",
                "asset_path": "sub-01/ses-2_behavior.nwb",
                "series_name": None,
            }
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
