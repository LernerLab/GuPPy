import holoviews as hv
import numpy as np
import pytest

from guppy.visualization.transients import build_peaks_overlay


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def z_score():
    return np.array([0.5, 1.5, 0.2, 2.5, 0.1])


@pytest.fixture
def peaks_index():
    return np.array([1, 3])


class TestBuildPeaksOverlay:
    def test_returns_curve_and_scatter(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = build_peaks_overlay(
            title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
        )
        assert isinstance(overlay, hv.Overlay)
        types = [type(element).__name__ for element in overlay.values()]
        assert types == ["Curve", "Scatter"]

    def test_trace_has_correct_data(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = build_peaks_overlay(
            title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
        )
        np.testing.assert_array_equal(overlay.Curve.I.dimension_values(0), timestamps)
        np.testing.assert_array_equal(overlay.Curve.I.dimension_values(1), z_score)

    def test_markers_at_peak_positions(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = build_peaks_overlay(
            title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
        )
        # Markers sit at t=1.0 and t=3.0 with the corresponding z-score values.
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(0), np.array([1.0, 3.0]))
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(1), np.array([1.5, 2.5]))

    def test_empty_peaks_index(self, panel_extension, timestamps, z_score):
        overlay = build_peaks_overlay(
            title="t", suptitle="s", z_score=z_score, timestamps=timestamps, peaksIndex=np.array([], dtype=int)
        )
        assert overlay.Scatter.I.dimension_values(0).shape == (0,)
