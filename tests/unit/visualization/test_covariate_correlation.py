import numpy as np
import pytest

from guppy.visualization.covariate_correlation import build_covariate_scatter


@pytest.fixture
def scatter(panel_extension):
    """A three-bin scatter with one unusable bin, so the drop is observable."""
    return build_covariate_scatter(
        covariate_values=np.array([2.0, np.nan, 4.0, 10.0]),
        metric_values=np.array([1.5, 5.0, 5.5, 9.0]),
        bin_numbers=np.array([0, 1, 2, 3]),
        covariate_label="akinesia",
        metric_label="mean z-score",
        pearson_r=0.9493907,
        spearman_rho=1.0,
        n_bins=3,
        suptitle="dms",
    )


class TestBuildCovariateScatter:
    def test_drops_bins_missing_either_value(self, scatter):
        np.testing.assert_allclose(scatter.dimension_values("akinesia"), np.array([2.0, 4.0, 10.0]))
        np.testing.assert_allclose(scatter.dimension_values("mean z-score"), np.array([1.5, 5.5, 9.0]))

    def test_keeps_bin_numbers_of_the_surviving_bins(self, scatter):
        np.testing.assert_array_equal(scatter.dimension_values("bin"), np.array([0, 2, 3]))

    def test_axis_labels_come_from_the_arguments(self, scatter):
        assert [dimension.name for dimension in scatter.kdims] == ["akinesia", "mean z-score"]
        assert [dimension.name for dimension in scatter.vdims] == ["bin"]

    def test_title_reports_both_coefficients_and_the_bin_count(self, scatter):
        title = scatter.opts.get().kwargs["title"]

        assert title == "dms — r = 0.95, rho = 1.00, n = 3 bins"

    def test_title_reports_no_p_value(self, scatter):
        title = scatter.opts.get().kwargs["title"]

        assert "p =" not in title
        assert "p-value" not in title

    def test_undefined_coefficients_render_as_nan(self, panel_extension):
        scatter = build_covariate_scatter(
            covariate_values=np.array([3.0, 3.0, 3.0]),
            metric_values=np.array([1.5, 5.5, 9.0]),
            bin_numbers=np.array([0, 1, 2]),
            covariate_label="akinesia",
            metric_label="mean z-score",
            pearson_r=float("nan"),
            spearman_rho=float("nan"),
            n_bins=3,
            suptitle="dms",
        )

        assert scatter.opts.get().kwargs["title"] == "dms — r = nan, rho = nan, n = 3 bins"
