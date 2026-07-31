import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.analysis.io_utils import fetchCoords
from guppy.frontend.artifact_removal import (
    ArtifactRemovalConfig,
    build_folder_page,
    build_preprocess_view_page,
    load_pair_traces,
    load_preprocessed_traces,
    signal_options,
)
from guppy.utils._hdf5_io import write_hdf5
from guppy_test_data import resolve_plot

TIMESTAMPS = np.arange(0.0, 11.0, 1.0)


def _write_site(filepath, site):
    """Write the control/signal/fit + preprocessed HDF5 files for one recording site."""
    write_hdf5(TIMESTAMPS, "timeCorrection_" + site, str(filepath), "timestampNew")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]), "control_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 2, "signal_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 3, "cntrl_sig_fit_" + site, str(filepath), "data")
    write_hdf5(TIMESTAMPS.copy(), "z_score_" + site, str(filepath), "data")
    write_hdf5(TIMESTAMPS / 10.0, "dff_" + site, str(filepath), "data")


@pytest.fixture
def run_folder(tmp_path):
    for site in ("DMS", "DLS"):
        _write_site(tmp_path, site)
    return tmp_path


def test_signal_options():
    assert signal_options("z_score") == ["z_score"]
    assert signal_options("dff") == ["dff"]
    assert signal_options("Both") == ["z_score", "dff"]
    assert signal_options("None") == []


def test_load_pair_traces_raises_for_mismatched_recording_sites(tmp_path):
    """A control recording site with no matching signal recording site raises a pairing error."""
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_vms.hdf5").touch()
    with pytest.raises(ValueError) as exception_info:
        load_pair_traces(str(tmp_path))
    message = str(exception_info.value)
    assert "Mismatched control/signal files" in message
    assert "dms" in message
    assert "vms" in message


class TestArtifactRemovalConfig:
    @pytest.fixture
    def config(self, panel_extension, run_folder):
        return ArtifactRemovalConfig(
            str(run_folder),
            load_pair_traces(str(run_folder)),
            load_preprocessed_traces(str(run_folder)),
            signal_options("z_score"),
        )

    def test_sites_discovered_from_control_signal_pairs(self, config):
        assert sorted(config.sites) == ["DLS", "DMS"]

    def test_save_writes_interleaved_npy_roundtripping_through_fetchcoords(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame(
            {"label": ["keep_a", "keep_b"], "start": [1.0, 6.0], "end": [3.0, 8.0]}
        )
        config.save()

        saved = np.load(str(run_folder / "coordsForPreProcessing_DMS.npy"))
        assert saved.shape == (4, 2)
        np.testing.assert_array_equal(saved[:, 0], np.array([1.0, 3.0, 6.0, 8.0]))
        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[1.0, 3.0], [6.0, 8.0]]))

    def test_save_empty_table_writes_no_file(self, config, run_folder):
        config.save()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()
        assert not (run_folder / "coordsForPreProcessing_DLS.npy").exists()

    def test_save_raises_when_window_outside_timespan(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["late"], "start": [8.0], "end": [15.0]})
        with pytest.raises(ValueError, match="outside the recording timespan"):
            config.save()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_on_save_writes_on_success(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["a"], "start": [1.0], "end": [2.0]})
        config._on_save(None)
        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DMS", TIMESTAMPS), np.array([[1.0, 2.0]]))

    def test_on_save_swallows_validation_error_without_writing(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["bad"], "start": [5.0], "end": [5.0]})
        config._on_save(None)
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_marking_plot_shades_defined_windows(self, config):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["a"], "start": [1.0], "end": [2.0]})
        config.site_select.value = "DMS"
        plot = config._make_marking_plot()
        for element in (resolve_plot(axis) for axis in plot.values()):
            assert isinstance(element, hv.Overlay)
            assert any(isinstance(item, hv.VSpan) for item in element.values())


class TestReviewSectionGating:
    def test_review_section_present_when_requested(self, panel_extension, run_folder):
        config = ArtifactRemovalConfig(
            str(run_folder),
            load_pair_traces(str(run_folder)),
            load_preprocessed_traces(str(run_folder)),
            signal_options("Both"),
        )
        assert config.signal_toggle is not None
        assert config.signal_toggle.options == ["z_score", "dff"]

    def test_review_section_absent_when_none(self, panel_extension, run_folder):
        config = ArtifactRemovalConfig(
            str(run_folder),
            load_pair_traces(str(run_folder)),
            load_preprocessed_traces(str(run_folder)),
            signal_options("None"),
        )
        assert config.signal_toggle is None
        assert config.review_pane is None


class TestBuildFolderPage:
    def test_marking_page_has_editable_table(self, panel_extension, run_folder):
        page = build_folder_page(str(run_folder), remove_artifacts=False, plot_zScore_dff="z_score")
        assert page.select(pn.widgets.Tabulator), "marking page must contain the editable windows table"

    def test_review_page_has_no_editable_table(self, panel_extension, run_folder):
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        np.save(str(run_folder / "coordsForPreProcessing_DLS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        page = build_folder_page(str(run_folder), remove_artifacts=True, plot_zScore_dff="z_score")
        assert not page.select(pn.widgets.Tabulator), "review page is read-only — no editable table"


class TestBuildPreprocessViewPage:
    def test_single_folder_has_no_folder_selector(self, panel_extension, run_folder):
        page = build_preprocess_view_page([str(run_folder)], remove_artifacts=False, plot_zScore_dff="None")
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert run_folder_selectors == []

    def test_multiple_folders_get_a_folder_selector(self, panel_extension, tmp_path):
        folder_a = tmp_path / "session_a_output_1"
        folder_b = tmp_path / "session_b_output_1"
        for folder in (folder_a, folder_b):
            folder.mkdir()
            _write_site(folder, "DMS")
        page = build_preprocess_view_page(
            [str(folder_a), str(folder_b)], remove_artifacts=False, plot_zScore_dff="None"
        )
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert len(run_folder_selectors) == 1
        assert set(run_folder_selectors[0].options.values()) == {str(folder_a), str(folder_b)}


class TestPreprocessingReviewView:
    def test_toggle_switches_between_zscore_and_dff(self, panel_extension, run_folder):
        from guppy.frontend.artifact_removal import PreprocessingReviewView

        view = PreprocessingReviewView(str(run_folder), load_preprocessed_traces(str(run_folder)), ["z_score", "dff"])
        view.site_select.value = "DMS"
        view.signal_toggle.value = "z_score"
        np.testing.assert_array_equal(resolve_plot(view.plot_pane.object).dimension_values(1), TIMESTAMPS)
        view.signal_toggle.value = "dff"
        np.testing.assert_array_equal(resolve_plot(view.plot_pane.object).dimension_values(1), TIMESTAMPS / 10.0)


class TestArtifactReviewView:
    def test_shades_saved_good_chunks(self, panel_extension, run_folder):
        from guppy.frontend.artifact_removal import ArtifactReviewView

        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        view = ArtifactReviewView(str(run_folder), load_pair_traces(str(run_folder)))
        view.site_select.value = "DMS"
        plot = view.plot_pane.object
        for element in (resolve_plot(axis) for axis in plot.values()):
            vspans = [item for item in element.values() if isinstance(item, hv.VSpan)]
            assert len(vspans) == 1
