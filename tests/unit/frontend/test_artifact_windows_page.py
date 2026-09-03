import json
import os

import numpy as np
import panel as pn
import pytest

from guppy.analysis.io_utils import fetchCoords
from guppy.frontend.artifact_removal import load_pair_traces
from guppy.frontend.artifact_windows_page import (
    ArtifactWindowSelector,
    build_artifact_window_page,
)
from guppy.utils._hdf5_io import write_hdf5

# The keep-window span is widened by one sample, so it runs from -1.0 to 11.0.
TIMESTAMPS = np.arange(0.0, 11.0, 1.0)


def _write_site(filepath, site):
    """Write the control/signal/fit HDF5 files for one recording site."""
    write_hdf5(TIMESTAMPS, "timeCorrection_" + site, str(filepath), "timestampNew")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]), "control_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 2, "signal_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 3, "cntrl_sig_fit_" + site, str(filepath), "data")


@pytest.fixture
def run_folder(tmp_path):
    for site in ("DMS", "DLS"):
        _write_site(tmp_path, site)
    # Step 3 always leaves a snapshot behind; the page updates its artifact keys on save.
    with open(os.path.join(str(tmp_path), "GuPPyParamtersUsed.json"), "w") as parameters_file:
        json.dump({"removeArtifacts": False, "artifactsRemovalMethod": "replace with NaN"}, parameters_file)
    return tmp_path


@pytest.fixture
def session_folder(tmp_path):
    """A session holding two runs of the same recording, neither one marked yet."""
    session = tmp_path / "mySession"
    for run_name in ("1", "2"):
        run_folder = session / f"mySession_output_{run_name}"
        run_folder.mkdir(parents=True)
        for site in ("DMS", "DLS"):
            _write_site(run_folder, site)
        with open(os.path.join(str(run_folder), "GuPPyParamtersUsed.json"), "w") as parameters_file:
            json.dump({"removeArtifacts": False, "artifactsRemovalMethod": "replace with NaN"}, parameters_file)
    return session


@pytest.fixture
def selector(panel_extension, run_folder):
    selector = ArtifactWindowSelector(str(run_folder), load_pair_traces(str(run_folder)))
    # Adding, removing, and the live preview act on the selected site; pin it so the tests
    # do not depend on which site load_pair_traces happens to yield first.
    selector.site_select.value = "DMS"
    return selector


class TestArtifactWindowSelector:
    def test_sites_discovered_from_control_signal_pairs(self, selector):
        assert sorted(selector.sites) == ["DLS", "DMS"]

    def test_starts_with_no_marked_periods(self, selector):
        assert selector.windows_for("DMS") == []
        assert selector.windows_for("DLS") == []

    def test_save_writes_the_complement_of_the_marked_artifacts(self, selector, run_folder):
        """The user marks the bad stretch; disk stores the keep-windows around it."""
        selector.set_windows("DMS", [(3.0, 5.0)])

        selector.save()

        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[-1.0, 3.0], [5.0, 11.0]]))

    def test_save_merges_overlapping_artifact_windows(self, selector, run_folder):
        selector.set_windows("DMS", [(3.0, 5.0), (4.0, 7.0)])

        selector.save()

        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[-1.0, 3.0], [7.0, 11.0]]))

    def test_save_writes_interleaved_npy(self, selector, run_folder):
        selector.set_windows("DMS", [(3.0, 5.0)])

        selector.save()

        saved = np.load(str(run_folder / "coordsForPreProcessing_DMS.npy"))
        np.testing.assert_array_equal(saved[:, 0], np.array([-1.0, 3.0, 5.0, 11.0]))

    def test_period_reaching_the_start_trims_the_opening(self, selector, run_folder):
        """Marking from the start of the trace is how one session loses more of its opening."""
        selector.set_windows("DMS", [(0.0, 3.0)])

        selector.save()

        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DMS", TIMESTAMPS), np.array([[3.0, 11.0]]))

    def test_period_reaching_the_end_trims_the_tail(self, selector, run_folder):
        selector.set_windows("DMS", [(8.0, 10.0)])

        selector.save()

        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DMS", TIMESTAMPS), np.array([[-1.0, 8.0]]))

    def test_period_marked_into_the_margin_is_accepted(self, selector, run_folder):
        """The bounds run one sample past the recording, so the edge is reachable."""
        selector.set_windows("DMS", [(-1.0, 3.0)])

        selector.save()

        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DMS", TIMESTAMPS), np.array([[3.0, 11.0]]))

    def test_save_with_nothing_marked_writes_no_file(self, selector, run_folder):
        selector.save()

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()
        assert not (run_folder / "coordsForPreProcessing_DLS.npy").exists()

    def test_save_raises_when_window_outside_timespan(self, selector, run_folder):
        selector.set_windows("DMS", [(8.0, 15.0)])

        with pytest.raises(ValueError, match="outside the recording timespan"):
            selector.save()

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_save_raises_when_window_is_inverted(self, selector, run_folder):
        selector.set_windows("DMS", [(5.0, 5.0)])

        with pytest.raises(ValueError, match="must be less than"):
            selector.save()

    def test_save_records_removal_method_in_snapshot(self, selector, run_folder):
        selector.set_windows("DMS", [(3.0, 5.0)])
        selector.method_select.value = "concatenate"

        selector.save()

        with open(os.path.join(str(run_folder), "GuPPyParamtersUsed.json")) as parameters_file:
            saved = json.load(parameters_file)
        assert saved["artifactsRemovalMethod"] == "concatenate"
        # Selecting windows does not itself remove anything.
        assert saved["removeArtifacts"] is False

    def test_seeded_from_disk_inverts_saved_keep_windows(self, panel_extension, run_folder):
        """Re-opening the page shows the artifact periods the user marked, not the keep-windows."""
        np.save(
            str(run_folder / "coordsForPreProcessing_DMS.npy"),
            np.array([[-1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [11.0, 0.0]]),
        )

        reopened = ArtifactWindowSelector(str(run_folder), load_pair_traces(str(run_folder)))

        assert reopened.windows_for("DMS") == [(3.0, 5.0)]
        assert reopened.windows_for("DLS") == []

    def test_empty_state_explains_what_happens_when_nothing_is_marked(self, selector):
        rendered = selector.rows_container.objects
        assert len(rendered) == 1
        assert "entire recording will be kept" in rendered[0].object

    def test_marked_periods_render_with_a_column_header(self, selector):
        selector.set_windows("DMS", [(1.0, 2.0), (6.0, 7.0)])

        rendered = selector.rows_container.objects
        # One header row plus one row per marked period.
        assert len(rendered) == 3
        assert [row.widget for row in selector.site_to_rows["DMS"]] == list(rendered[1:])

    def test_switching_site_shows_that_site_periods(self, selector):
        selector.set_windows("DMS", [(1.0, 2.0)])
        selector.set_windows("DLS", [(3.0, 4.0), (6.0, 7.0)])

        selector.site_select.value = "DLS"

        assert [row.widget for row in selector.site_to_rows["DLS"]] == list(selector.rows_container.objects[1:])
        assert selector.spans_pipe.data == [(3.0, 4.0), (6.0, 7.0)]

    def test_add_period_appends_a_blank_row(self, selector):
        selector.add_window_row()
        selector.add_window_row()

        assert len(selector.site_to_rows["DMS"]) == 2
        # Blank rows are not yet windows, so they do not shade or save.
        assert selector.windows_for("DMS") == []

    def test_drag_marks_a_period(self, selector):
        selector.mark_window_from_drag(3.0, 5.0)

        assert selector.windows_for("DMS") == [(3.0, 5.0)]

    def test_drag_shades_the_period_it_marked(self, selector):
        selector.mark_window_from_drag(3.0, 5.0)

        assert selector.spans_pipe.data == [(3.0, 5.0)]

    def test_drag_bounds_are_ordered_and_rounded(self, selector):
        """Dragging right-to-left marks the same period, to the millisecond."""
        selector.mark_window_from_drag(5.00048, 3.00012)

        assert selector.windows_for("DMS") == [(3.0, 5.0)]

    def test_drag_past_the_edge_marks_up_to_the_edge(self, selector):
        selector.mark_window_from_drag(-40.0, 5.0)

        assert selector.windows_for("DMS") == [(-1.0, 5.0)]

    def test_a_click_marks_nothing(self, selector):
        """A press without a drag covers no time, so it is not a period."""
        selector.mark_window_from_drag(4.0, 4.0)

        assert selector.windows_for("DMS") == []
        assert selector.site_to_rows["DMS"] == []

    def test_drag_marks_only_the_selected_site(self, selector):
        selector.mark_window_from_drag(3.0, 5.0)

        assert selector.windows_for("DLS") == []

    def test_remove_period_drops_that_row(self, selector):
        selector.set_windows("DMS", [(1.0, 2.0), (6.0, 7.0)])

        selector.remove_window_row(0)

        assert selector.windows_for("DMS") == [(6.0, 7.0)]

    def test_row_delete_button_removes_its_own_row(self, selector):
        """Each period owns its delete button, so there is nothing to select first."""
        selector.set_windows("DMS", [(1.0, 2.0), (6.0, 7.0)])

        selector.site_to_rows["DMS"][1].remove_button.clicks += 1

        assert selector.windows_for("DMS") == [(1.0, 2.0)]

    def test_editing_a_bound_updates_the_marked_window(self, selector):
        selector.set_windows("DMS", [(1.0, 2.0)])

        selector.site_to_rows["DMS"][0].end_input.value = 4.5

        assert selector.windows_for("DMS") == [(1.0, 4.5)]

    def test_bounds_are_limited_to_the_recording_span(self, selector):
        """A guardrail in the browser; values set programmatically are still validated on save."""
        selector.add_window_row()
        row = selector.site_to_rows["DMS"][0]

        assert (row.start_input.start, row.start_input.end) == (-1.0, 11.0)
        assert (row.end_input.start, row.end_input.end) == (-1.0, 11.0)

    def test_apply_to_all_sites_copies_periods(self, selector):
        selector.set_windows("DMS", [(3.0, 5.0)])

        selector.apply_windows_to_all_sites()

        assert selector.windows_for("DLS") == [(3.0, 5.0)]

    def test_apply_to_all_sites_then_save_writes_every_site(self, selector, run_folder):
        selector.set_windows("DMS", [(3.0, 5.0)])
        selector.apply_windows_to_all_sites()

        selector.save()

        for site in ("DMS", "DLS"):
            np.testing.assert_array_equal(
                fetchCoords(str(run_folder), site, TIMESTAMPS), np.array([[-1.0, 3.0], [5.0, 11.0]])
            )

    def test_edit_updates_spans_without_rebuilding_the_layout(self, selector):
        """Editing a bound repaints only the shaded spans; the trace layout object is untouched."""
        selector.set_windows("DMS", [(1.0, 2.0)])
        layout_before = selector.marking_pane.object

        selector.site_to_rows["DMS"][0].end_input.value = 4.5

        assert selector.marking_pane.object is layout_before
        assert selector.spans_pipe.data == [(1.0, 4.5)]

    def test_blank_rows_are_skipped_in_the_live_preview(self, selector):
        selector.set_windows("DMS", [(3.0, 5.0)])
        selector.add_window_row()

        selector.refresh_spans()

        assert selector.spans_pipe.data == [(3.0, 5.0)]

    def test_on_save_swallows_validation_error_without_writing(self, selector, run_folder):
        selector.set_windows("DMS", [(5.0, 5.0)])

        selector._on_save(None)

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()


class TestBuildArtifactWindowPage:
    def test_page_has_no_data_grid(self, panel_extension, run_folder):
        """Periods are form controls, not a spreadsheet."""
        page = build_artifact_window_page(run_folders=[str(run_folder)])
        assert not page.select(pn.widgets.Tabulator)

    def test_page_offers_an_add_control(self, panel_extension, run_folder):
        page = build_artifact_window_page(run_folders=[str(run_folder)])
        assert [button for button in page.select(pn.widgets.Button) if button.name == "Add period"]

    def test_single_folder_has_no_folder_selector(self, panel_extension, run_folder):
        page = build_artifact_window_page(run_folders=[str(run_folder)])
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert run_folder_selectors == []

    def test_multiple_folders_get_a_folder_selector(self, panel_extension, tmp_path):
        folder_a = tmp_path / "session_a_output_1"
        folder_b = tmp_path / "session_b_output_1"
        for folder in (folder_a, folder_b):
            folder.mkdir()
            _write_site(folder, "DMS")
        page = build_artifact_window_page(run_folders=[str(folder_a), str(folder_b)])
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert len(run_folder_selectors) == 1


class TestCopyWindowsFromAnotherRun:
    """Artifact windows are a property of the recording, so a second run can reuse them."""

    @pytest.fixture
    def marked_run(self, session_folder):
        """Run 1, with a different period marked for each recording site."""
        run_folder = session_folder / "mySession_output_1"
        for site, coords in (("DMS", [-1.0, 3.0, 5.0, 11.0]), ("DLS", [-1.0, 7.0, 9.0, 11.0])):
            np.save(
                str(run_folder / f"coordsForPreProcessing_{site}.npy"),
                np.array([[bound, 0.0] for bound in coords]),
            )
        return run_folder

    @pytest.fixture
    def unmarked_selector(self, panel_extension, session_folder, marked_run):
        run_folder = str(session_folder / "mySession_output_2")
        selector = ArtifactWindowSelector(run_folder, load_pair_traces(run_folder))
        selector.site_select.value = "DMS"
        return selector

    def test_offers_the_other_run_that_has_windows(self, unmarked_selector):
        assert list(unmarked_selector.runs_with_windows) == ["1"]

    def test_does_not_offer_a_run_with_no_windows_saved(self, panel_extension, session_folder):
        run_folder = str(session_folder / "mySession_output_2")
        selector = ArtifactWindowSelector(run_folder, load_pair_traces(run_folder))

        assert selector.runs_with_windows == {}

    def test_control_is_hidden_when_no_other_run_has_windows(self, panel_extension, session_folder):
        run_folder = str(session_folder / "mySession_output_2")
        selector = ArtifactWindowSelector(run_folder, load_pair_traces(run_folder))

        assert selector.copy_from_select not in selector.widget.select(pn.widgets.Select)

    def test_loading_fills_every_recording_site(self, unmarked_selector):
        unmarked_selector.copy_windows_from_run("1")

        assert unmarked_selector.windows_for("DMS") == [(3.0, 5.0)]
        assert unmarked_selector.windows_for("DLS") == [(7.0, 9.0)]

    def test_loading_shades_the_selected_site(self, unmarked_selector):
        unmarked_selector.copy_windows_from_run("1")

        assert unmarked_selector.spans_pipe.data == [(3.0, 5.0)]

    def test_loaded_windows_are_editable_before_saving(self, unmarked_selector, session_folder):
        """Loading fills the rows; nothing is written until Save."""
        unmarked_selector.copy_windows_from_run("1")
        unmarked_selector.site_to_rows["DMS"][0].end_input.value = 6.0

        assert not (session_folder / "mySession_output_2" / "coordsForPreProcessing_DMS.npy").exists()

        unmarked_selector.save()

        coords = fetchCoords(str(session_folder / "mySession_output_2"), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[-1.0, 3.0], [6.0, 11.0]]))

    def test_loading_replaces_whatever_was_already_marked(self, unmarked_selector):
        unmarked_selector.set_windows("DMS", [(1.0, 2.0)])

        unmarked_selector.copy_windows_from_run("1")

        assert unmarked_selector.windows_for("DMS") == [(3.0, 5.0)]

    def test_the_button_loads_the_selected_run(self, unmarked_selector):
        unmarked_selector.copy_from_button.clicks += 1

        assert unmarked_selector.windows_for("DMS") == [(3.0, 5.0)]
