import holoviews as hv
import numpy as np
import panel as pn
import pytest

from guppy.frontend.artifact_removal import (
    PreprocessingReviewView,
    build_artifact_review_page,
    build_preprocess_view_page,
    load_pair_traces,
    load_preprocessed_traces,
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


@pytest.fixture
def preprocessing_review(panel_extension, run_folder):
    return PreprocessingReviewView(
        str(run_folder),
        load_pair_traces(str(run_folder)),
        load_preprocessed_traces(str(run_folder)),
        artifacts_removed=False,
    )


@pytest.fixture
def artifact_review(panel_extension, run_folder):
    return PreprocessingReviewView(
        str(run_folder),
        load_pair_traces(str(run_folder)),
        load_preprocessed_traces(str(run_folder)),
        artifacts_removed=True,
    )


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
        page = build_preprocess_view_page(run_folders=[str(run_folder)])
        assert not page.select(pn.widgets.Tabulator), "the Step-3 view is read-only — no editable table"

    def test_single_folder_has_no_folder_selector(self, panel_extension, run_folder):
        page = build_preprocess_view_page(run_folders=[str(run_folder)])
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert run_folder_selectors == []

    def test_multiple_folders_get_a_folder_selector(self, panel_extension, tmp_path):
        folder_a = tmp_path / "session_a_output_1"
        folder_b = tmp_path / "session_b_output_1"
        for folder in (folder_a, folder_b):
            folder.mkdir()
            _write_site(folder, "DMS")
        page = build_preprocess_view_page(run_folders=[str(folder_a), str(folder_b)])
        run_folder_selectors = [w for w in page.select(pn.widgets.Select) if w.name == "Run folder"]
        assert len(run_folder_selectors) == 1
        assert set(run_folder_selectors[0].options.values()) == {str(folder_a), str(folder_b)}


class TestBuildArtifactReviewPage:
    def test_page_is_read_only(self, panel_extension, run_folder):
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        np.save(str(run_folder / "coordsForPreProcessing_DLS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        page = build_artifact_review_page(run_folders=[str(run_folder)])
        assert not page.select(pn.widgets.Tabulator), "the review page is read-only — no editable table"


def _panel_titles(view):
    """The title of each stacked panel, in display order."""
    return [resolve_plot(axis).opts.get("plot").kwargs["title"] for axis in view.plot_pane.object.values()]


def _panel_elements(view, index):
    """The elements drawn on one panel; a lone trace resolves to a bare element, not an Overlay."""
    plot = resolve_plot(view.plot_pane.object.values()[index])
    return list(plot.values()) if isinstance(plot, hv.Overlay) else [plot]


def _shaded_bounds(view, index):
    """The ``(left, bottom, right, top)`` extent of the shaded trace on one panel."""
    images = [item for item in _panel_elements(view, index) if isinstance(item, hv.RGB)]
    assert len(images) == 1
    return images[0].bounds.lbrt()


def _span_elements(view):
    """Every VSpans element across the view's stacked panels."""
    return [
        item
        for index in range(len(view.plot_pane.object.values()))
        for item in _panel_elements(view, index)
        if isinstance(item, hv.VSpans)
    ]


class TestPreprocessingReviewView:
    def test_stacks_all_five_traces_for_the_selected_site(self, preprocessing_review, run_folder):
        preprocessing_review.site_select.value = "DMS"
        assert _panel_titles(preprocessing_review) == [
            f"{run_folder.name} — control_DMS",
            "signal_DMS",
            "cntrl_sig_fit_DMS",
            "z_score_DMS",
            "dff_DMS",
        ]

    def test_zscore_and_dff_panels_show_their_own_data(self, preprocessing_review):
        preprocessing_review.site_select.value = "DMS"
        # The traces are shaded into images, so the y-extent of the shaded region is what
        # identifies which one is on display: z_score spans 0-10, dff spans 0-1.
        assert _shaded_bounds(preprocessing_review, 3) == (0.0, 0.0, 10.0, 10.0)
        assert _shaded_bounds(preprocessing_review, 4) == (0.0, 0.0, 10.0, 1.0)

    def test_site_selector_switches_every_panel_together(self, preprocessing_review, run_folder):
        preprocessing_review.site_select.value = "DLS"
        assert _panel_titles(preprocessing_review) == [
            f"{run_folder.name} — control_DLS",
            "signal_DLS",
            "cntrl_sig_fit_DLS",
            "z_score_DLS",
            "dff_DLS",
        ]

    def test_shades_nothing_even_when_windows_are_saved(self, preprocessing_review, run_folder):
        """Marking windows is a later step, so a review of preprocessing never shades them."""
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        preprocessing_review.site_select.value = "DMS"

        assert _span_elements(preprocessing_review) == []

    def test_shades_nothing_after_removal(self, artifact_review, run_folder):
        """Once the artifacts are gone from the data, shading the kept region says nothing."""
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        artifact_review.site_select.value = "DMS"

        assert _span_elements(artifact_review) == []
