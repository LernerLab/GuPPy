import holoviews as hv
import numpy as np
import panel as pn
import pytest

from guppy.frontend.transient_peaks import (
    PeaksReviewView,
    build_peaks_view_page,
    load_peaks,
)
from guppy.utils._hdf5_io import write_hdf5
from guppy_test_data import resolve_plot

TIMESTAMPS = np.arange(0.0, 11.0, 1.0)
Z_SCORE = TIMESTAMPS.copy()
PEAKS_INDEX = np.array([2, 7])


def _write_transient_outputs(filepath, title):
    # The z_score_* / dff_* file is what load_peaks globs to discover traces.
    write_hdf5(Z_SCORE, title, str(filepath), "data")
    write_hdf5(Z_SCORE, "transient_outputs_" + title, str(filepath), "z_score")
    write_hdf5(TIMESTAMPS, "transient_outputs_" + title, str(filepath), "timestamps")
    write_hdf5(PEAKS_INDEX, "transient_outputs_" + title, str(filepath), "peaksInd")


@pytest.fixture
def run_folder(tmp_path):
    output_dir = tmp_path / "session" / "output_1"
    output_dir.mkdir(parents=True)
    _write_transient_outputs(output_dir, "z_score_DMS")
    return output_dir


class TestLoadPeaks:
    def test_loads_traces_under_their_run(self, run_folder):
        runs = load_peaks([str(run_folder)], "z_score")
        assert list(runs) == ["output_1"]
        assert list(runs["output_1"]) == ["z_score_DMS"]
        entry = runs["output_1"]["z_score_DMS"]
        np.testing.assert_array_equal(entry["timestamps"], TIMESTAMPS)
        np.testing.assert_array_equal(entry["peaksInd"], PEAKS_INDEX)


class TestPeaksReviewView:
    def test_plot_marks_peaks(self, panel_extension, run_folder):
        view = PeaksReviewView(load_peaks([str(run_folder)], "z_score"))
        overlay = resolve_plot(view.plot_pane.object)
        assert isinstance(overlay, hv.Overlay)
        # Peaks at index 2 and 7 -> t = 2.0, 7.0.
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(0), np.array([2.0, 7.0]))


class TestPeaksReviewViewSelectors:
    @pytest.fixture
    def two_runs(self, tmp_path):
        folders = []
        for session, traces in (("session_a", ["z_score_DMS", "z_score_DLS"]), ("session_b", ["z_score_DLS"])):
            folder = tmp_path / session / "output_1"
            folder.mkdir(parents=True)
            for trace in traces:
                _write_transient_outputs(folder, trace)
            folders.append(str(folder))
        return folders

    def test_toolbar_offers_the_run_then_its_traces(self, panel_extension, two_runs):
        view = PeaksReviewView(load_peaks(two_runs, "z_score"))

        assert view.run_select.options == ["session_a/output_1", "session_b/output_1"]
        assert sorted(view.trace_select.options) == ["z_score_DLS", "z_score_DMS"]
        assert [field[0].object.split(">")[1].split("<")[0] for field in view.widget[1]] == ["Run Folder", "Trace"]

    def test_switching_run_keeps_the_trace_when_the_run_has_it(self, panel_extension, two_runs):
        view = PeaksReviewView(load_peaks(two_runs, "z_score"))
        view.trace_select.value = "z_score_DLS"

        view.run_select.value = "session_b/output_1"

        assert view.trace_select.options == ["z_score_DLS"]
        assert view.trace_select.value == "z_score_DLS"

    def test_switching_run_falls_back_to_its_first_trace(self, panel_extension, two_runs):
        view = PeaksReviewView(load_peaks(two_runs, "z_score"))
        view.run_select.value = "session_b/output_1"
        view.trace_select.value = "z_score_DLS"

        view.run_select.value = "session_a/output_1"

        assert view.trace_select.value == "z_score_DLS"


class TestBuildPeaksViewPage:
    def test_returns_page_with_a_peak_plot(self, panel_extension, run_folder):
        page = build_peaks_view_page([str(run_folder)], "z_score")
        holoviews_panes = page.select(pn.pane.HoloViews)
        assert holoviews_panes, "peaks view page must contain a plot"
        assert isinstance(resolve_plot(holoviews_panes[0].object), hv.Overlay)
