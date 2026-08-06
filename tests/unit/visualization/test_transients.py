import holoviews as hv
import numpy as np
import pytest

from guppy.visualization.transients import build_peaks_overlay
from guppy_test_data import resolve_plot


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
    def test_returns_shaded_trace_and_scatter(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = resolve_plot(
            build_peaks_overlay(
                title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
            )
        )
        assert isinstance(overlay, hv.Overlay)
        # The trace is shaded into an image; the peak markers stay a scatter of real points.
        types = [type(element).__name__ for element in overlay.values()]
        assert types == ["RGB", "Scatter"]

    def test_trace_has_correct_data(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = resolve_plot(
            build_peaks_overlay(
                title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
            )
        )
        # The shaded trace spans the full timestamp range and the z-score min/max.
        extent = tuple(float(value) for value in overlay.RGB.I.bounds.lbrt())
        assert extent == (0.0, 0.1, 4.0, 2.5)

    def test_markers_at_peak_positions(self, panel_extension, timestamps, z_score, peaks_index):
        overlay = resolve_plot(
            build_peaks_overlay(
                title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
            )
        )
        # Markers sit at t=1.0 and t=3.0 with the corresponding z-score values.
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(0), np.array([1.0, 3.0]))
        np.testing.assert_array_equal(overlay.Scatter.I.dimension_values(1), np.array([1.5, 2.5]))

    def test_stretches_to_the_page_width(self, panel_extension, timestamps, z_score, peaks_index):
        from bokeh.models import Plot

        figure = hv.render(
            build_peaks_overlay(
                title="z_score_DMS", suptitle="session", z_score=z_score, timestamps=timestamps, peaksIndex=peaks_index
            )
        )
        plots = [model for model in figure.references() if isinstance(model, Plot)]
        assert len(plots) == 1
        assert (plots[0].sizing_mode, plots[0].width, plots[0].height) == ("stretch_width", None, 300)

    def test_empty_peaks_index(self, panel_extension, timestamps, z_score):
        overlay = resolve_plot(
            build_peaks_overlay(
                title="t", suptitle="s", z_score=z_score, timestamps=timestamps, peaksIndex=np.array([], dtype=int)
            )
        )
        assert overlay.Scatter.I.dimension_values(0).shape == (0,)
