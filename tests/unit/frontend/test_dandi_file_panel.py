"""Unit tests for the panel that picks NWB files out of one DANDI dandiset."""

from pathlib import Path
from threading import Event

import pytest
from dandi.exceptions import NotFoundError

from guppy.frontend.dandi_file_panel import DandiFilePanel
from guppy.utils.dandi_search import AssetSummary

from .test_dandi_preview_panel import RecordingPreview, make_preview, make_probe

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
def file_panel(panel_extension, tmp_path):
    return DandiFilePanel(
        mirror_parent=str(tmp_path / "mirror"),
        list_assets_function=RecordingAssetListing(),
        preview_function=RecordingPreview(),
        scan_function=RecordingScan(),
    )


def run_scan(file_panel):
    """Press the scan button and drive the poller until the scan has been applied."""
    file_panel.scan_assets()
    file_panel._scan["thread"].join()
    file_panel._poll_scan()


class TestDandiFilePanel:
    def test_constructs_empty(self, file_panel):
        assert file_panel.selected_uris == []
        assert file_panel._current_mirror_root is None
        assert file_panel.catalog_view.visible is True
        assert file_panel.files_view.visible is False

    def test_dandiset_change_builds_mirror(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        assert (mirror_root / "sub-01").is_dir()
        assert (mirror_root / "sub-02").is_dir()
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").is_file()
        assert (mirror_root / "sub-01" / "ses-2_behavior.nwb").is_file()
        assert (mirror_root / "sub-02" / "ses-1_behavior.nwb").is_file()
        # Placeholders are zero bytes.
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").stat().st_size == 0
        assert "3 NWB asset" in file_panel.dandiset_heading.object

    def test_the_heading_reports_the_asset_size_range(self, file_panel):
        file_panel.load_dandiset("000971")
        assert "### Dandiset 000971" in file_panel.dandiset_heading.object
        assert "234 KB – 57.2 MB" in file_panel.dandiset_heading.object

    def test_file_selector_is_scoped_to_dandiset(self, file_panel):
        file_panel.load_dandiset("000971")
        assert file_panel.asset_file_selector.root_directory == file_panel._current_mirror_root
        assert file_panel.asset_file_selector.directory == file_panel._current_mirror_root

    def test_selected_uris_translates_absolute_paths(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [
            str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"),
            str(Path(mirror_root) / "sub-02" / "ses-1_behavior.nwb"),
        ]
        assert file_panel.selected_uris == [
            "dandi://000971/sub-01/ses-1_behavior.nwb",
            "dandi://000971/sub-02/ses-1_behavior.nwb",
        ]

    def test_non_nwb_selections_filtered(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        # A folder path sneaking into .value should be ignored.
        file_panel.asset_file_selector.value = [
            str(Path(mirror_root) / "sub-01"),
            str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"),
        ]
        assert file_panel.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_switching_dandiset_repoints_selector(self, file_panel):
        file_panel.load_dandiset("000971")
        first_root = file_panel._current_mirror_root
        file_panel.load_dandiset("000001")
        second_root = file_panel._current_mirror_root
        assert first_root != second_root
        assert (Path(second_root) / "sub-a" / "data.nwb").is_file()
        assert file_panel.asset_file_selector.root_directory == second_root
        # Prior selection cleared on dandiset change.
        assert file_panel.asset_file_selector.value == []

    def test_clearing_dandiset_clears_selections(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]
        file_panel._reset_to_empty()
        assert file_panel._current_mirror_root is None
        assert file_panel.selected_uris == []

    def test_selected_uris_empty_when_no_dandiset(self, file_panel):
        assert file_panel.selected_uris == []

    def test_output_root_returns_first_selected(self, file_panel, tmp_path):
        assert file_panel.output_root is None
        file_panel.output_root_selector.value = [str(tmp_path)]
        assert file_panel.output_root == str(tmp_path)

    def test_directory_path_input_hidden(self, file_panel):
        assert file_panel.asset_file_selector._directory.visible is False

    def test_not_found_shows_warning_and_no_stale_folder(self, file_panel):
        file_panel.load_dandiset("999999")

        assert "⚠️" in file_panel.status.object
        assert "not found" in file_panel.status.object
        assert file_panel._current_mirror_root is None
        assert not (Path(file_panel._mirror_parent) / "999999").is_dir()

    def test_recovery_after_error(self, file_panel):
        file_panel.load_dandiset("999999")
        assert "not found" in file_panel.status.object

        file_panel.load_dandiset("000971")
        assert file_panel.status.object == ""
        assert file_panel._current_mirror_root is not None
        assert file_panel.asset_file_selector.root_directory == file_panel._current_mirror_root

        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]
        assert file_panel.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]

    def test_widget_swapped_on_load(self, file_panel):
        original = file_panel.asset_file_selector
        file_panel.load_dandiset("000971")
        assert file_panel.asset_file_selector is not original
        assert file_panel.asset_file_selector.root_directory.endswith("000971")
        # The slot holds exactly the current widget.
        assert list(file_panel._asset_file_selector_slot) == [file_panel.asset_file_selector]

    def test_widget_swapped_on_reset(self, file_panel):
        file_panel.load_dandiset("000971")
        loaded_widget = file_panel.asset_file_selector
        file_panel._reset_to_empty()
        assert file_panel.asset_file_selector is not loaded_widget
        assert file_panel.asset_file_selector.root_directory == file_panel._mirror_parent
        assert list(file_panel._asset_file_selector_slot) == [file_panel.asset_file_selector]

    def test_asset_selection_watcher_fires_on_the_rebuilt_widget(self, file_panel):
        events = []
        file_panel.attach_asset_selection_watcher(callback=events.append)
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        asset_path = Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb"
        events.clear()

        # The widget selected into here is the replacement built for this dandiset,
        # not the one the watcher was originally attached to.
        file_panel.asset_file_selector.value = [asset_path]

        assert len(events) == 1
        assert events[0].new == [asset_path]

    def test_asset_selection_watcher_is_notified_of_the_widget_swap(self, file_panel):
        events = []
        file_panel.attach_asset_selection_watcher(callback=events.append)

        file_panel.load_dandiset("000971")

        # The swap drops the previous selection without firing a value event of its own.
        assert events == [None]


class TestDandiFilePanelPhotometryScan:
    def test_before_a_scan_every_asset_is_mirrored(self, file_panel):
        file_panel.load_dandiset("000971")
        assert "Showing **3** of 3" in file_panel.asset_status.object

    def test_the_scan_button_waits_for_a_dandiset(self, file_panel):
        assert file_panel.scan_button.disabled
        file_panel.load_dandiset("000971")
        assert not file_panel.scan_button.disabled

    def test_the_filter_waits_for_a_scan(self, file_panel):
        file_panel.load_dandiset("000971")
        assert file_panel.photometry_only.disabled
        run_scan(file_panel)
        assert not file_panel.photometry_only.disabled

    def test_a_finished_scan_narrows_the_tree_to_the_photometry_assets(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        mirror_root = Path(file_panel._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-2_behavior.nwb").is_file()
        assert not (mirror_root / "sub-01" / "ses-1_behavior.nwb").exists()
        assert not (mirror_root / "sub-02").exists()

    def test_a_finished_scan_reports_what_it_found(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        assert file_panel.asset_status.object == (
            "Showing **1** of 3 NWB asset(s) in the tree below. " "Scanned 3 file(s): **1** hold fiber photometry."
        )

    def test_unchecking_the_filter_restores_the_assets_it_hid(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        file_panel.photometry_only.value = False
        mirror_root = Path(file_panel._current_mirror_root)
        assert (mirror_root / "sub-01" / "ses-1_behavior.nwb").is_file()
        assert "Showing **3** of 3" in file_panel.asset_status.object

    def test_the_progress_bar_tracks_the_running_scan(self, file_panel):
        file_panel.load_dandiset("000971")
        file_panel.scan_function.gate.clear()
        file_panel.scan_assets()
        assert file_panel.scan_progress.visible
        assert file_panel.scan_progress.max == 3
        assert file_panel.scan_progress.value == 0
        assert file_panel.scan_button.loading
        assert file_panel.scan_button.disabled
        assert file_panel.asset_status.object == "Scanning 0 of 3 NWB asset(s) for fiber photometry..."

        file_panel.scan_function.gate.set()
        file_panel._scan["thread"].join()
        file_panel._poll_scan()
        assert not file_panel.scan_progress.visible
        assert not file_panel.scan_button.loading
        assert not file_panel.scan_button.disabled

    def test_polling_mid_scan_renders_the_count_without_applying_verdicts(self, file_panel):
        file_panel.load_dandiset("000971")
        file_panel.scan_function.gate.clear()
        file_panel.scan_assets()
        file_panel._record_scan_progress(2)
        file_panel._poll_scan()
        assert file_panel.scan_progress.value == 2
        assert file_panel.asset_status.object == "Scanning 2 of 3 NWB asset(s) for fiber photometry..."
        assert file_panel._photometry_by_path == {}
        file_panel.scan_function.gate.set()
        file_panel._scan["thread"].join()

    def test_the_scan_reads_every_listed_asset(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        assert file_panel.scan_function.calls == [
            [
                "sub-01/ses-1_behavior.nwb",
                "sub-01/ses-2_behavior.nwb",
                "sub-02/ses-1_behavior.nwb",
            ]
        ]

    def test_scanning_does_not_relist_the_dandiset(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        file_panel.photometry_only.value = False
        assert file_panel.list_assets_function.calls == ["000971"]

    def test_switching_dandiset_forgets_the_previous_scan(self, file_panel):
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        file_panel.load_dandiset("000001")
        assert file_panel._photometry_by_path == {}
        assert file_panel.photometry_only.disabled
        assert not file_panel.photometry_only.value
        assert "Showing **1** of 1" in file_panel.asset_status.object

    def test_scanning_without_a_dandiset_does_nothing(self, file_panel):
        file_panel.scan_assets()
        assert file_panel.scan_function.calls == []
        assert file_panel._current_mirror_root is None
        assert file_panel.asset_status.object == ""

    def test_a_dandiset_with_no_assets_reports_no_size_range(self, file_panel):
        file_panel.list_assets_function.assets_by_id["000002"] = []
        file_panel.load_dandiset("000002")
        assert file_panel.dandiset_heading.object == "### Dandiset 000002\n0 NWB asset(s)."
        assert "Showing **0** of 0" in file_panel.asset_status.object
        assert file_panel.scan_button.disabled


class TestDandiFilePanelPreviewPicker:
    def test_the_picker_offers_the_selected_files_in_path_order(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [
            str(mirror_root / "sub-02" / "ses-1_behavior.nwb"),
            str(mirror_root / "sub-01" / "ses-1_behavior.nwb"),
        ]

        assert file_panel.preview_select.options == [
            "sub-01/ses-1_behavior.nwb",
            "sub-02/ses-1_behavior.nwb",
        ]
        assert file_panel.preview_select.value == "sub-01/ses-1_behavior.nwb"

    def test_the_picker_decides_which_file_is_streamed(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [
            str(mirror_root / "sub-01" / "ses-1_behavior.nwb"),
            str(mirror_root / "sub-02" / "ses-1_behavior.nwb"),
        ]
        file_panel.preview_select.value = "sub-02/ses-1_behavior.nwb"

        file_panel.preview_selected_asset()

        assert file_panel.preview_function.calls == [
            {
                "dandiset_id": "000971",
                "asset_path": "sub-02/ses-1_behavior.nwb",
                "series_name": None,
            }
        ]

    def test_a_still_selected_file_stays_chosen_when_the_selection_grows(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [str(mirror_root / "sub-02" / "ses-1_behavior.nwb")]
        assert file_panel.preview_select.value == "sub-02/ses-1_behavior.nwb"

        file_panel.asset_file_selector.value = [
            str(mirror_root / "sub-01" / "ses-1_behavior.nwb"),
            str(mirror_root / "sub-02" / "ses-1_behavior.nwb"),
        ]

        assert file_panel.preview_select.value == "sub-02/ses-1_behavior.nwb"

    def test_deselecting_the_chosen_file_falls_back_to_the_first(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [
            str(mirror_root / "sub-01" / "ses-1_behavior.nwb"),
            str(mirror_root / "sub-02" / "ses-1_behavior.nwb"),
        ]
        file_panel.preview_select.value = "sub-02/ses-1_behavior.nwb"

        file_panel.asset_file_selector.value = [str(mirror_root / "sub-01" / "ses-1_behavior.nwb")]

        assert file_panel.preview_select.options == ["sub-01/ses-1_behavior.nwb"]
        assert file_panel.preview_select.value == "sub-01/ses-1_behavior.nwb"

    def test_the_picker_empties_when_the_dandiset_changes(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [str(mirror_root / "sub-01" / "ses-1_behavior.nwb")]

        file_panel.load_dandiset("000001")

        assert file_panel.preview_select.options == []
        assert file_panel.preview_select.value is None

    def test_the_picker_follows_the_tree_rebuilt_by_the_photometry_filter(self, file_panel):
        # The FileSelector is rebuilt whenever the filter is toggled, so a picker watching the
        # widget rather than the file_panel would go stale here without raising anything. The
        # scan leaves the filter on, so unticking it is what rebuilds the tree.
        file_panel.load_dandiset("000971")
        run_scan(file_panel)
        mirror_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [str(mirror_root / "sub-01" / "ses-2_behavior.nwb")]
        assert file_panel.preview_select.options == ["sub-01/ses-2_behavior.nwb"]

        file_panel.photometry_only.value = False

        assert file_panel.preview_select.options == []
        assert file_panel.preview_select.value is None

        new_root = Path(file_panel._current_mirror_root)
        file_panel.asset_file_selector.value = [str(new_root / "sub-02" / "ses-1_behavior.nwb")]
        assert file_panel.preview_select.options == ["sub-02/ses-1_behavior.nwb"]


class TestDandiFilePanelAssetPreview:
    def test_preview_streams_the_selected_asset(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-2_behavior.nwb")]

        file_panel.preview_selected_asset()

        assert file_panel.preview_function.calls == [
            {
                "dandiset_id": "000971",
                "asset_path": "sub-01/ses-2_behavior.nwb",
                "series_name": None,
            }
        ]
        assert file_panel.preview_panel.panel.visible is True
        assert "2 channel(s)" in file_panel.preview_panel.summary.object
        assert "Previewed `sub-01/ses-2_behavior.nwb`" in file_panel.asset_status.object

    def test_preview_without_a_selection_warns(self, file_panel):
        file_panel.load_dandiset("000971")
        file_panel.preview_selected_asset()
        assert file_panel.preview_function.calls == []
        assert "Select an NWB file" in file_panel.asset_status.object

    def test_preview_of_a_file_without_photometry_says_so(self, file_panel):
        file_panel.preview_function.preview = make_preview(probe=make_probe(has_photometry=False))
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]

        file_panel.preview_selected_asset()

        assert "no `FiberPhotometryResponseSeries`" in file_panel.preview_panel.summary.object

    def test_changing_dandiset_clears_a_stale_preview(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-2_behavior.nwb")]
        file_panel.preview_selected_asset()

        file_panel.load_dandiset("000001")

        assert file_panel.preview_panel.panel.visible is False


class TestDandiFilePanelNavigation:
    def test_the_catalog_is_the_opening_screen(self, file_panel):
        assert file_panel.catalog_view.visible is True
        assert file_panel.files_view.visible is False

    def test_choosing_a_dandiset_opens_its_files(self, file_panel):
        file_panel.search_panel.on_dandiset_selected("000971")
        assert file_panel.catalog_view.visible is False
        assert file_panel.files_view.visible is True
        assert file_panel._dandiset_id == "000971"
        assert file_panel._current_mirror_root is not None
        assert "3 NWB asset" in file_panel.dandiset_heading.object

    def test_load_dandiset_is_what_the_browser_calls(self, file_panel):
        file_panel.load_dandiset("000001")
        assert file_panel._dandiset_id == "000001"
        assert file_panel.list_assets_function.calls == ["000001"]

    def test_going_back_returns_to_the_catalog(self, file_panel):
        file_panel.load_dandiset("000971")
        file_panel.open_catalog()
        assert file_panel.catalog_view.visible is True
        assert file_panel.files_view.visible is False

    def test_an_unknown_dandiset_stays_on_the_catalog(self, file_panel):
        file_panel.load_dandiset("999999")
        assert file_panel.catalog_view.visible is True
        assert file_panel.files_view.visible is False
        assert "not found" in file_panel.status.object


class TestDandiFilePanelPreviewVisibility:
    def test_the_hide_button_appears_only_once_a_preview_is_shown(self, file_panel):
        file_panel.load_dandiset("000971")
        assert file_panel.hide_preview_button.visible is False

        mirror_root = file_panel._current_mirror_root
        file_panel.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")]
        file_panel.preview_selected_asset()
        assert file_panel.preview_panel.panel.visible is True
        assert file_panel.hide_preview_button.visible is True

    def test_hiding_puts_the_preview_away_and_keeps_the_selection(self, file_panel):
        file_panel.load_dandiset("000971")
        mirror_root = file_panel._current_mirror_root
        selected = str(Path(mirror_root) / "sub-01" / "ses-1_behavior.nwb")
        file_panel.asset_file_selector.value = [selected]
        file_panel.preview_selected_asset()

        file_panel.hide_preview()
        assert file_panel.preview_panel.panel.visible is False
        assert file_panel.hide_preview_button.visible is False
        assert file_panel.selected_uris == ["dandi://000971/sub-01/ses-1_behavior.nwb"]
