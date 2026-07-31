import holoviews as hv
import numpy as np
import pytest
from holoviews.streams import RangeX

from guppy.visualization.downsampling import downsample_for_display
from guppy_test_data import resolve_plot

# The m4 downsampler rounds the requested width down to a multiple of 4, since it keeps the
# minimum, maximum, first and last sample of each pixel column.
POINTS_AT_FULL_WIDTH = 748

# A flat trace long enough to need downsampling, carrying two one-sample artifacts: a positive
# spike at t = 500.0 and a negative one at t = 500.4.
TRACE_LENGTH = 1_000_000
SPIKE_INDEX = 500_000
SPIKE_VALUE = 50.0
DIP_INDEX = 500_400
DIP_VALUE = -30.0


@pytest.fixture
def long_trace():
    timestamps = np.arange(TRACE_LENGTH) * 0.001
    values = np.zeros(TRACE_LENGTH)
    values[SPIKE_INDEX] = SPIKE_VALUE
    values[DIP_INDEX] = DIP_VALUE
    return timestamps, values


class TestDownsampleForDisplay:
    def test_short_trace_passes_through_unchanged(self, panel_extension):
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        curve = resolve_plot(downsample_for_display(hv.Curve((timestamps, values), "time (s)", "sig")))
        np.testing.assert_array_equal(curve.dimension_values(0), timestamps)
        np.testing.assert_array_equal(curve.dimension_values(1), values)

    def test_long_trace_reduced_to_one_bucket_per_pixel_column(self, panel_extension, long_trace):
        timestamps, values = long_trace
        curve = resolve_plot(downsample_for_display(hv.Curve((timestamps, values), "time (s)", "sig")))
        assert curve.dimension_values(0).shape == (POINTS_AT_FULL_WIDTH,)

    def test_single_sample_artifacts_survive_downsampling(self, panel_extension, long_trace):
        timestamps, values = long_trace
        curve = resolve_plot(downsample_for_display(hv.Curve((timestamps, values), "time (s)", "sig")))
        # Each spike is 1 sample in 1,000,000 — m4 keeps per-column extremes, so both remain.
        drawn = curve.dimension_values(1)
        assert drawn.max() == SPIKE_VALUE
        assert drawn.min() == DIP_VALUE

    def test_zoom_reresolves_against_the_full_trace(self, panel_extension, long_trace):
        timestamps, values = long_trace
        plot = downsample_for_display(hv.Curve((timestamps, values), "time (s)", "sig"))
        range_stream = next(stream for stream in plot.streams if isinstance(stream, RangeX))

        range_stream.event(x_range=(499.0, 501.0))
        curve = resolve_plot(plot)

        # The zoomed view is redrawn from the full array, so it still spends its whole point
        # budget — now on the 2 s window alone, which is what keeps zoomed detail exact.
        drawn_timestamps = curve.dimension_values(0)
        assert drawn_timestamps.shape == (POINTS_AT_FULL_WIDTH,)
        assert drawn_timestamps.min() == pytest.approx(499.0)
        assert drawn_timestamps.max() == pytest.approx(500.999)
        assert curve.dimension_values(1).max() == SPIKE_VALUE
