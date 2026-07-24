import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.transient_peaks import build_peaks_review_template, load_peaks
from guppy.utils._hdf5_io import write_hdf5

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
    output_dir = tmp_path / "session_output_1"
    output_dir.mkdir()
    _write_transient_outputs(output_dir, "z_score_DMS")
    return output_dir


class TestLoadPeaks:
    def test_loads_entry_per_trace(self, run_folder):
        entries = load_peaks([str(run_folder)], "z_score")
        assert list(entries.keys()) == ["session_output_1 / z_score_DMS"]
        entry = entries["session_output_1 / z_score_DMS"]
        np.testing.assert_array_equal(entry["timestamps"], TIMESTAMPS)
        np.testing.assert_array_equal(entry["peaksInd"], PEAKS_INDEX)


class TestPeaksReviewView:
    def test_plot_marks_peaks(self, panel_extension, run_folder):
        template = build_peaks_review_template([str(run_folder)], "z_score")
        overlay = template._view.plot_pane.object
        assert isinstance(overlay, hv.Overlay)
        # Peaks at index 2 and 7 -> t = 2.0, 7.0.
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(0), np.array([2.0, 7.0]))

    def test_continue_button_signals_done(self, panel_extension, run_folder):
        calls = []
        template = build_peaks_review_template([str(run_folder)], "z_score", on_done=lambda: calls.append(1))
        template._continue_button.clicks += 1
        assert calls == [1]
