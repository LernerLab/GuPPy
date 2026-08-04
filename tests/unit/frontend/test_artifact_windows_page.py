import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
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


@dataclass
class FakeTableClick:
    """A Tabulator cell-click event: which column was clicked, and on which row."""

    column: str
    row: int


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
def selector(panel_extension, run_folder):
    selector = ArtifactWindowSelector(str(run_folder), load_pair_traces(str(run_folder)))
    # The row, apply-to-all, and preview methods act on the selected site; pin it so the
    # tests do not depend on which site load_pair_traces happens to yield first.
    selector.site_select.value = "DMS"
    return selector


class TestArtifactWindowSelector:
    def test_sites_discovered_from_control_signal_pairs(self, selector):
        assert sorted(selector.sites) == ["DLS", "DMS"]

    def test_table_starts_empty_with_only_start_and_end_columns(self, selector):
        table = selector.site_to_table["DMS"]
        assert list(table.value.columns) == ["start", "end"]
        assert len(table.value) == 0

    def test_save_writes_the_complement_of_the_marked_artifacts(self, selector, run_folder):
        """The user marks the bad stretch; disk stores the keep-windows around it."""
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})

        selector.save()

        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[-1.0, 3.0], [5.0, 11.0]]))

    def test_save_merges_overlapping_artifact_windows(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0, 4.0], "end": [5.0, 7.0]})

        selector.save()

        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[-1.0, 3.0], [7.0, 11.0]]))

    def test_save_writes_interleaved_npy(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})

        selector.save()

        saved = np.load(str(run_folder / "coordsForPreProcessing_DMS.npy"))
        np.testing.assert_array_equal(saved[:, 0], np.array([-1.0, 3.0, 5.0, 11.0]))

    def test_save_empty_table_writes_no_file(self, selector, run_folder):
        selector.save()

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()
        assert not (run_folder / "coordsForPreProcessing_DLS.npy").exists()

    def test_save_raises_when_window_outside_timespan(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [8.0], "end": [15.0]})

        with pytest.raises(ValueError, match="outside the recording timespan"):
            selector.save()

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_save_records_removal_method_in_snapshot(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})
        selector.method_select.value = "concatenate"

        selector.save()

        with open(os.path.join(str(run_folder), "GuPPyParamtersUsed.json")) as parameters_file:
            saved = json.load(parameters_file)
        assert saved["artifactsRemovalMethod"] == "concatenate"
        # Selecting windows does not itself remove anything.
        assert saved["removeArtifacts"] is False

    def test_seeded_from_disk_inverts_saved_keep_windows(self, panel_extension, run_folder):
        """Re-opening the page shows the artifact windows the user marked, not the keep-windows."""
        np.save(
            str(run_folder / "coordsForPreProcessing_DMS.npy"),
            np.array([[-1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [11.0, 0.0]]),
        )

        reopened = ArtifactWindowSelector(str(run_folder), load_pair_traces(str(run_folder)))

        assert list(reopened.site_to_table["DMS"].value.itertuples(index=False, name=None)) == [(3.0, 5.0)]
        assert len(reopened.site_to_table["DLS"].value) == 0

    def test_add_and_remove_rows(self, selector):
        table = selector.site_to_table["DMS"]

        selector.add_window_row()
        selector.add_window_row()
        assert len(table.value) == 2

        table.value = pd.DataFrame({"start": [1.0, 6.0], "end": [2.0, 7.0]})
        selector.remove_window_row(0)

        assert list(table.value.itertuples(index=False, name=None)) == [(6.0, 7.0)]

    def test_each_row_carries_its_own_delete_button(self, selector):
        """No checkbox column to explain: the ✕ on a row deletes that row."""
        table = selector.site_to_table["DMS"]
        assert table.selectable is False
        assert list(table.buttons) == ["remove"]

    def test_clicking_a_row_delete_button_removes_that_row(self, selector):
        table = selector.site_to_table["DMS"]
        table.value = pd.DataFrame({"start": [1.0, 6.0], "end": [2.0, 7.0]})

        selector._on_table_click(FakeTableClick(column="remove", row=1))

        assert list(table.value.itertuples(index=False, name=None)) == [(1.0, 2.0)]

    def test_clicking_a_data_cell_removes_nothing(self, selector):
        table = selector.site_to_table["DMS"]
        table.value = pd.DataFrame({"start": [1.0, 6.0], "end": [2.0, 7.0]})

        selector._on_table_click(FakeTableClick(column="start", row=1))

        assert len(table.value) == 2

    def test_apply_to_all_sites_copies_table(self, selector):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})

        selector.apply_windows_to_all_sites()

        assert list(selector.site_to_table["DLS"].value.itertuples(index=False, name=None)) == [(3.0, 5.0)]

    def test_apply_to_all_sites_then_save_writes_every_site(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})
        selector.apply_windows_to_all_sites()

        selector.save()

        for site in ("DMS", "DLS"):
            np.testing.assert_array_equal(
                fetchCoords(str(run_folder), site, TIMESTAMPS), np.array([[-1.0, 3.0], [5.0, 11.0]])
            )

    def test_edit_updates_spans_without_rebuilding_the_layout(self, selector):
        """A table edit repaints only the shaded spans; the trace layout object is untouched."""
        layout_before = selector.marking_pane.object
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0], "end": [5.0]})

        selector.refresh_spans()

        assert selector.marking_pane.object is layout_before
        assert selector.spans_pipe.data == [(3.0, 5.0)]

    def test_incomplete_rows_are_skipped_in_the_live_preview(self, selector):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [3.0, 7.0], "end": [5.0, np.nan]})

        selector.refresh_spans()

        assert selector.spans_pipe.data == [(3.0, 5.0)]

    def test_on_save_swallows_validation_error_without_writing(self, selector, run_folder):
        selector.site_to_table["DMS"].value = pd.DataFrame({"start": [5.0], "end": [5.0]})

        selector._on_save(None)

        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()


class TestBuildArtifactWindowPage:
    def test_page_has_an_editable_table(self, panel_extension, run_folder):
        page = build_artifact_window_page(run_folders=[str(run_folder)])
        assert page.select(pn.widgets.Tabulator), "the marking page must contain the editable windows table"

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
