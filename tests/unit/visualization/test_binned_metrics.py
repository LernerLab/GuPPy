import holoviews as hv
import numpy as np
import pytest

from guppy.visualization.binned_metrics import (
    build_bin_bars,
    build_binned_metrics_panel,
)


@pytest.fixture
def bins():
    # Two whole bins and a short third one, as a ragged final bin produces.
    return {
        "bin_starts": np.array([0.0, 4.0, 8.0]),
        "bin_ends": np.array([4.0, 8.0, 10.0]),
    }


class TestBuildBinBars:
    def test_one_bar_per_bin_spanning_its_bounds(self, bins, panel_extension):
        bars = build_bin_bars(**bins, values=np.array([1.5, 5.5, 9.0]), value_label="mean z-score")

        data = bars.array()
        assert data.shape[0] == 3
        # Each bar runs from its bin start to its bin end, and from zero to the value.
        np.testing.assert_allclose(data[:, 0], [0.0, 4.0, 8.0])
        np.testing.assert_allclose(data[:, 1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data[:, 2], [4.0, 8.0, 10.0])
        np.testing.assert_allclose(data[:, 3], [1.5, 5.5, 9.0])

    def test_final_bar_is_narrower_when_the_final_bin_is_ragged(self, bins, panel_extension):
        bars = build_bin_bars(**bins, values=np.array([1.5, 5.5, 9.0]), value_label="mean z-score")

        data = bars.array()
        widths = data[:, 2] - data[:, 0]
        np.testing.assert_allclose(widths, [4.0, 4.0, 2.0])

    def test_nan_bins_are_not_drawn(self, bins, panel_extension):
        bars = build_bin_bars(**bins, values=np.array([1.5, np.nan, 9.0]), value_label="mean z-score")

        data = bars.array()
        assert data.shape[0] == 2
        np.testing.assert_allclose(data[:, 0], [0.0, 8.0])

    def test_negative_values_bar_down_from_zero(self, bins, panel_extension):
        bars = build_bin_bars(**bins, values=np.array([-1.5, 5.5, 9.0]), value_label="mean z-score")

        data = bars.array()
        np.testing.assert_allclose(data[0, 1], 0.0)
        np.testing.assert_allclose(data[0, 3], -1.5)

    def test_time_is_the_first_key_dimension(self, bins, panel_extension):
        # The trace panel plots against "time (s)"; matching that name is what
        # makes bokeh link the two axes in the stacked layout.
        bars = build_bin_bars(**bins, values=np.array([1.5, 5.5, 9.0]), value_label="mean z-score")

        assert [dimension.name for dimension in bars.kdims][:2] == ["time (s)", "mean z-score"]


class TestBuildBinnedMetricsPanel:
    def test_stacks_the_trace_over_the_bars(self, bins, panel_extension):
        timestamps = np.arange(0, 11, 1, dtype=float)
        layout = build_binned_metrics_panel(
            timestamps=timestamps,
            trace=timestamps.copy(),
            trace_label="z-score",
            **bins,
            values=np.array([1.5, 5.5, 9.0]),
            value_label="mean z-score",
            suptitle="dms",
        )

        assert len(layout) == 2

    def test_both_panels_share_one_time_axis(self, bins, panel_extension):
        # Naming both panels' x dimension "time (s)" is what makes bokeh give them
        # a single shared x_range, so zooming the bars zooms the trace with them.
        timestamps = np.arange(0, 11, 1, dtype=float)
        layout = build_binned_metrics_panel(
            timestamps=timestamps,
            trace=timestamps.copy(),
            trace_label="z-score",
            **bins,
            values=np.array([1.5, 5.5, 9.0]),
            value_label="mean z-score",
            suptitle="dms",
        )

        plot = hv.renderer("bokeh").get_plot(layout)
        trace_subplot, bars_subplot = list(plot.subplots.values())

        assert trace_subplot.subplots["main"].handles["x_range"] is bars_subplot.subplots["main"].handles["x_range"]
