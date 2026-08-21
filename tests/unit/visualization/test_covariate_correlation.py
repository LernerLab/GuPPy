import numpy as np
import pytest

from guppy.visualization.covariate_correlation import (
    build_covariate_panel,
    build_covariate_scatter,
)


@pytest.fixture
def scatter(panel_extension):
    """A three-bin scatter with one unusable bin, so the drop is observable."""
    return build_covariate_scatter(
        covariate_values=np.array([0.0, np.nan, 2.0, 4.0]),
        metric_values=np.array([1.0, 5.0, 2.0, 6.0]),
        covariate_label="akinesia",
        metric_label="mean z-score",
        pearson_r=0.9449112,
        n_bins=3,
        suptitle="dms",
    )


@pytest.fixture
def panel(panel_extension):
    """A covariate sampled every two seconds, averaged into two five-second bins."""
    return build_covariate_panel(
        covariate_timestamps=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        covariate_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        bin_starts=np.array([0.0, 5.0]),
        bin_ends=np.array([5.0, 10.0]),
        binned_values=np.array([2.0, 4.5]),
        covariate_label="akinesia",
    )


class TestBuildCovariatePanel:
    def test_stacks_the_raw_series_above_its_bins(self, panel):
        assert len(panel) == 2

    def test_raw_panel_keeps_every_sample(self, panel):
        line = panel[0].Curve.I

        np.testing.assert_allclose(line.dimension_values("time (s)"), [0.0, 2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(line.dimension_values("akinesia"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_raw_panel_overlays_markers_on_the_line(self, panel):
        assert len(panel[0]) == 2

    def test_bins_span_their_true_bounds(self, panel):
        bars = panel[1]

        np.testing.assert_allclose(bars.dimension_values("time (s)"), [0.0, 5.0])
        np.testing.assert_allclose(bars.dimension_values("time (s) end"), [5.0, 10.0])
        np.testing.assert_allclose(bars.dimension_values("mean akinesia"), [0.0, 0.0])
        np.testing.assert_allclose(bars.dimension_values("mean akinesia end"), [2.0, 4.5])

    def test_both_panels_share_the_time_dimension(self, panel):
        assert panel[0].Curve.I.kdims[0].name == "time (s)"
        assert panel[1].kdims[0].name == "time (s)"

    def test_empty_bins_are_not_drawn_at_zero(self, panel_extension):
        panel = build_covariate_panel(
            covariate_timestamps=np.array([0.0, 2.0]),
            covariate_values=np.array([1.0, 3.0]),
            bin_starts=np.array([0.0, 5.0]),
            bin_ends=np.array([5.0, 10.0]),
            binned_values=np.array([2.0, np.nan]),
            covariate_label="akinesia",
        )

        np.testing.assert_allclose(panel[1].dimension_values("time (s)"), [0.0])


class TestBuildCovariateScatter:
    def test_drops_bins_missing_either_value(self, scatter):
        points = scatter.Points.I

        np.testing.assert_allclose(points.dimension_values("akinesia"), np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(points.dimension_values("mean z-score"), np.array([1.0, 2.0, 6.0]))

    def test_axis_labels_come_from_the_arguments(self, scatter):
        assert [dimension.name for dimension in scatter.Points.I.kdims] == ["akinesia", "mean z-score"]

    def test_points_carry_no_extra_dimension_to_color_by(self, scatter):
        assert scatter.Points.I.vdims == []
        assert "colorbar" not in scatter.Points.I.opts.get().kwargs

    def test_fit_line_spans_the_covariate_range(self, scatter):
        # Least squares through (0, 1), (2, 2) and (4, 6): slope 10/8 = 1.25,
        # intercept 3 - 1.25 * 2 = 0.5, so the line runs from (0, 0.5) to (4, 5.5).
        line = scatter.Curve.I

        np.testing.assert_allclose(line.dimension_values("akinesia"), [0.0, 4.0])
        np.testing.assert_allclose(line.dimension_values("mean z-score"), [0.5, 5.5])

    def test_fit_line_is_drawn_under_the_points(self, scatter):
        assert [element.__class__.__name__ for element in scatter] == ["Curve", "Points"]

    def test_title_reports_pearson_r_and_the_bin_count(self, scatter):
        title = scatter.opts.get().kwargs["title"]

        assert title == "dms — r = 0.94, n = 3 bins"

    def test_title_reports_neither_a_p_value_nor_spearman_rho(self, scatter):
        title = scatter.opts.get().kwargs["title"]

        assert "p =" not in title
        assert "p-value" not in title
        assert "rho" not in title

    def test_undefined_coefficient_renders_as_nan_with_no_fit_line(self, panel_extension):
        scatter = build_covariate_scatter(
            covariate_values=np.array([3.0, 3.0, 3.0]),
            metric_values=np.array([1.5, 5.5, 9.0]),
            covariate_label="akinesia",
            metric_label="mean z-score",
            pearson_r=float("nan"),
            n_bins=3,
            suptitle="dms",
        )

        assert scatter.opts.get().kwargs["title"] == "dms — r = nan, n = 3 bins"
        assert len(scatter.Curve.I) == 0

    def test_a_single_usable_bin_draws_no_fit_line(self, panel_extension):
        scatter = build_covariate_scatter(
            covariate_values=np.array([2.0, np.nan]),
            metric_values=np.array([1.5, 5.5]),
            covariate_label="akinesia",
            metric_label="mean z-score",
            pearson_r=float("nan"),
            n_bins=1,
            suptitle="dms",
        )

        assert len(scatter.Curve.I) == 0
