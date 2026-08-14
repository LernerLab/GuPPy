import holoviews as hv
import numpy as np
import pytest
from holoviews.streams import RangeXY

from guppy.visualization.shading import shade_trace
from guppy_test_data import resolve_plot

# A flat trace long enough to need shading, carrying a single one-sample artifact spike.
# 200,000 samples at 0.01 s spacing span t = 0 to 1999.99; the spike sits at t = 1000.0.
TRACE_LENGTH = 200_000
SAMPLE_SPACING = 0.01
SPIKE_INDEX = 100_000
SPIKE_TIME = 1000.0
SPIKE_VALUE = 50.0

# datashade aggregates onto a 400x400 grid by default when no plot size is attached.
IMAGE_SIZE = 400

# Antialiased drawing strokes a sample across the two columns its line spans.
SPIKE_COLUMN_SPAN = 2


@pytest.fixture
def spiked_trace():
    timestamps = np.arange(TRACE_LENGTH) * SAMPLE_SPACING
    values = np.zeros(TRACE_LENGTH)
    values[SPIKE_INDEX] = SPIKE_VALUE
    return hv.Curve((timestamps, values), "time (s)", "sig")


class TestShadeTrace:
    def test_renders_an_image_spanning_the_data(self, panel_extension, spiked_trace):
        image = resolve_plot(shade_trace(spiked_trace))
        assert isinstance(image, hv.RGB)
        left, bottom, right, top = (float(value) for value in image.bounds.lbrt())
        assert (left, bottom) == (0.0, 0.0)
        assert right == pytest.approx(1999.99)
        # The trace is flat at 0 apart from the spike, so the spike alone sets the top.
        assert top == SPIKE_VALUE

    def test_one_sample_spike_is_drawn_sparsely_above_a_solid_baseline(self, panel_extension, spiked_trace):
        """The density distinction the shading exists to make.

        Rows run bottom-up: the flat baseline at y=0 lands in row 0 and the spike at y=50
        in the last row. The baseline is drawn in every column, the spike in only the two
        its stroke spans — so a transient stays visible without being given the same weight
        as the noise band.
        """
        alpha = resolve_plot(shade_trace(spiked_trace)).dimension_values("A", flat=False)
        inked_columns_per_row = (alpha > 0).sum(axis=1)
        assert inked_columns_per_row[0] == IMAGE_SIZE
        assert inked_columns_per_row[-1] == SPIKE_COLUMN_SPAN

    def test_zoom_reaggregates_against_the_full_trace(self, panel_extension, spiked_trace):
        plot = shade_trace(spiked_trace)
        range_stream = next(stream for stream in plot.streams if isinstance(stream, RangeXY))

        range_stream.event(x_range=(999.0, 1001.0), y_range=None)
        image = resolve_plot(plot)

        # Redrawn from the full array over the 2 s window, so the spike at t=1000 still
        # sets the top of the range rather than having been dropped at load time.
        left, _, right, top = (float(value) for value in image.bounds.lbrt())
        assert (left, right) == (999.0, 1001.0)
        assert top == SPIKE_VALUE
        assert SPIKE_TIME == pytest.approx(SPIKE_INDEX * SAMPLE_SPACING)
