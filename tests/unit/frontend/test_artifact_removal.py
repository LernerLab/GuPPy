import holoviews as hv
import numpy as np
import panel as pn
import pytest

from guppy.frontend.artifact_removal import (
    ControlSignalFitView,
    PreprocessingReviewView,
    build_artifact_review_page,
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


class TestBuildPreprocessViewPage:
    def test_page_is_read_only(self, panel_extension, run_folder):
        page = build_preprocess_view_page(run_folders=[str(run_folder)], plot_zScore_dff="z_score")
        assert not page.select(pn.widgets.Tabulator), "the Step-3 view is read-only — no editable table"

    def test_review_section_present_when_requested(self, panel_extension, run_folder):
        page = build_preprocess_view_page(run_folders=[str(run_folder)], plot_zScore_dff="Both")
        toggles = page.select(pn.widgets.RadioButtonGroup)
        assert len(toggles) == 1
        assert toggles[0].options == ["z_score", "dff"]

    def test_review_section_absent_when_none(self, panel_extension, run_folder):
        page = build_preprocess_view_page(run_folders=[str(run_folder)], plot_zScore_dff="None")
        assert page.select(pn.widgets.RadioButtonGroup) == []

    def test_single_folder_has_no_folder_selector(self, panel_extension, run_folder):
        page = build_preprocess_view_page(run_folders=[str(run_folder)], plot_zScore_dff="None")
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert run_folder_selectors == []

    def test_multiple_folders_get_a_folder_selector(self, panel_extension, tmp_path):
        folder_a = tmp_path / "session_a_output_1"
        folder_b = tmp_path / "session_b_output_1"
        for folder in (folder_a, folder_b):
            folder.mkdir()
            _write_site(folder, "DMS")
        page = build_preprocess_view_page(run_folders=[str(folder_a), str(folder_b)], plot_zScore_dff="None")
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert len(run_folder_selectors) == 1
        assert set(run_folder_selectors[0].options.values()) == {str(folder_a), str(folder_b)}


class TestBuildArtifactReviewPage:
    def test_page_is_read_only(self, panel_extension, run_folder):
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        np.save(str(run_folder / "coordsForPreProcessing_DLS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        page = build_artifact_review_page(run_folders=[str(run_folder)], plot_zScore_dff="z_score")
        assert not page.select(pn.widgets.Tabulator), "the review page is read-only — no editable table"


class TestPreprocessingReviewView:
    def test_toggle_switches_between_zscore_and_dff(self, panel_extension, run_folder):
        view = PreprocessingReviewView(str(run_folder), load_preprocessed_traces(str(run_folder)), ["z_score", "dff"])
        view.site_select.value = "DMS"
        # The traces are shaded into images, so the y-extent of the shaded region is what
        # identifies which one is on display: z_score spans 0-10, dff spans 0-1.
        view.signal_toggle.value = "z_score"
        assert resolve_plot(view.plot_pane.object).bounds.lbrt() == (0.0, 0.0, 10.0, 10.0)
        view.signal_toggle.value = "dff"
        assert resolve_plot(view.plot_pane.object).bounds.lbrt() == (0.0, 0.0, 10.0, 1.0)


def _shaded_spans(view):
    """The VSpans element on each of the view's three panels."""
    spans = []
    for element in (resolve_plot(axis) for axis in view.plot_pane.object.values()):
        panel_spans = [item for item in element.values() if isinstance(item, hv.VSpans)]
        assert len(panel_spans) == 1
        spans.append(panel_spans[0])
    return spans


class TestControlSignalFitView:
    def test_shades_saved_keep_windows_before_removal(self, panel_extension, run_folder):
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        view = ControlSignalFitView(str(run_folder), load_pair_traces(str(run_folder)), artifacts_removed=False)
        view.site_select.value = "DMS"

        for spans in _shaded_spans(view):
            np.testing.assert_array_equal(spans.dimension_values("x0"), np.array([2.0]))
            np.testing.assert_array_equal(spans.dimension_values("x1"), np.array([5.0]))

    def test_shades_nothing_after_removal(self, panel_extension, run_folder):
        """Once the artifacts are gone from the data, shading the kept region says nothing."""
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        view = ControlSignalFitView(str(run_folder), load_pair_traces(str(run_folder)), artifacts_removed=True)
        view.site_select.value = "DMS"

        for spans in _shaded_spans(view):
            assert spans.dimension_values("x0").size == 0

    def test_shades_nothing_before_windows_are_saved(self, panel_extension, run_folder):
        view = ControlSignalFitView(str(run_folder), load_pair_traces(str(run_folder)), artifacts_removed=False)
        view.site_select.value = "DMS"

        for spans in _shaded_spans(view):
            assert spans.dimension_values("x0").size == 0
