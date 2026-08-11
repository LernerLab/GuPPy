import numpy as np
import pandas as pd
import pytest

from guppy.analysis.covariates import (
    bin_edges_from_binned_metrics,
    compute_binned_covariates,
    compute_covariate_correlations,
)


@pytest.fixture
def bin_edges():
    """Three bins, the last one short: [0, 4), [4, 8), [8, 10]."""
    return np.array([0.0, 4.0, 8.0, 10.0])


@pytest.fixture
def akinesia_samples():
    """Four scores whose bin means are hand-computable against ``bin_edges``."""
    return np.array([0.0, 2.0, 5.0, 9.0]), np.array([1.0, 3.0, 4.0, 10.0])


class TestBinEdgesFromBinnedMetrics:
    def test_recovers_edges_including_short_final_bin(self):
        binned_metrics = pd.DataFrame({"bin_start": [0.0, 4.0, 8.0], "bin_end": [4.0, 8.0, 10.0]})

        result = bin_edges_from_binned_metrics(binned_metrics=binned_metrics)

        np.testing.assert_array_equal(result, np.array([0.0, 4.0, 8.0, 10.0]))


class TestComputeBinnedCovariates:
    def test_bin_means_and_counts(self, bin_edges, akinesia_samples):
        result = compute_binned_covariates(covariate_series={"akinesia": akinesia_samples}, bin_edges=bin_edges)

        # bin 0 holds 1 and 3; bin 1 holds only 4; bin 2 holds only 10.
        np.testing.assert_allclose(result["akinesia"].to_numpy(), np.array([2.0, 4.0, 10.0]))
        np.testing.assert_array_equal(result["n_samples_akinesia"].to_numpy(), np.array([2, 1, 1]))

    def test_bin_bounds_and_index(self, bin_edges, akinesia_samples):
        result = compute_binned_covariates(covariate_series={"akinesia": akinesia_samples}, bin_edges=bin_edges)

        np.testing.assert_allclose(result["bin_start"].to_numpy(), np.array([0.0, 4.0, 8.0]))
        np.testing.assert_allclose(result["bin_end"].to_numpy(), np.array([4.0, 8.0, 10.0]))
        assert result.index.name == "bin"
        np.testing.assert_array_equal(result.index.to_numpy(), np.array([0, 1, 2]))

    def test_column_order(self, bin_edges, akinesia_samples):
        result = compute_binned_covariates(covariate_series={"akinesia": akinesia_samples}, bin_edges=bin_edges)

        assert list(result.columns) == ["bin_start", "bin_end", "akinesia", "n_samples_akinesia"]

    def test_two_covariates_group_values_before_counts(self, bin_edges, akinesia_samples):
        tremor = (np.array([1.0, 6.0, 9.0]), np.array([0.0, 2.0, 4.0]))

        result = compute_binned_covariates(
            covariate_series={"akinesia": akinesia_samples, "tremor": tremor}, bin_edges=bin_edges
        )

        assert list(result.columns) == [
            "bin_start",
            "bin_end",
            "akinesia",
            "tremor",
            "n_samples_akinesia",
            "n_samples_tremor",
        ]
        np.testing.assert_allclose(result["tremor"].to_numpy(), np.array([0.0, 2.0, 4.0]))

    def test_sample_on_interior_edge_opens_the_next_bin(self, bin_edges):
        samples = (np.array([0.0, 4.0, 5.0, 9.0]), np.array([1.0, 2.0, 3.0, 10.0]))

        result = compute_binned_covariates(covariate_series={"akinesia": samples}, bin_edges=bin_edges)

        # The sample at exactly 4.0 belongs to bin 1, averaging with 5.0 to 2.5.
        np.testing.assert_allclose(result["akinesia"].to_numpy(), np.array([1.0, 2.5, 10.0]))
        np.testing.assert_array_equal(result["n_samples_akinesia"].to_numpy(), np.array([1, 2, 1]))

    def test_sample_on_final_edge_joins_the_last_bin(self, bin_edges):
        samples = (np.array([0.0, 5.0, 10.0]), np.array([1.0, 2.0, 3.0]))

        result = compute_binned_covariates(covariate_series={"akinesia": samples}, bin_edges=bin_edges)

        np.testing.assert_allclose(result["akinesia"].to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_samples_outside_the_span_are_dropped(self, bin_edges):
        samples = (
            np.array([-5.0, 0.0, 2.0, 5.0, 9.0, 20.0]),
            np.array([99.0, 1.0, 3.0, 4.0, 10.0, 99.0]),
        )

        result = compute_binned_covariates(covariate_series={"akinesia": samples}, bin_edges=bin_edges)

        # Identical to the in-span case: the strays are not folded into the end bins.
        np.testing.assert_allclose(result["akinesia"].to_numpy(), np.array([2.0, 4.0, 10.0]))
        np.testing.assert_array_equal(result["n_samples_akinesia"].to_numpy(), np.array([2, 1, 1]))

    def test_empty_bin_reads_nan_with_zero_count(self):
        samples = (np.array([0.0, 2.0, 5.0, 9.0]), np.array([1.0, 3.0, 4.0, 10.0]))

        result = compute_binned_covariates(
            covariate_series={"akinesia": samples}, bin_edges=np.array([0.0, 4.0, 8.0, 12.0, 16.0])
        )

        np.testing.assert_allclose(result["akinesia"].to_numpy(), np.array([2.0, 4.0, 10.0, np.nan]))
        np.testing.assert_array_equal(result["n_samples_akinesia"].to_numpy(), np.array([2, 1, 1, 0]))

    def test_too_few_occupied_bins_raises_naming_both_spans(self, bin_edges):
        samples = (np.array([0.0, 1.0]), np.array([1.0, 2.0]))

        with pytest.raises(ValueError) as excinfo:
            compute_binned_covariates(covariate_series={"akinesia": samples}, bin_edges=bin_edges)

        message = str(excinfo.value)
        assert "akinesia" in message
        assert "[0, 1]s" in message
        assert "[0, 10]s" in message


@pytest.fixture
def binned_metrics():
    """Three bins of photometry metrics, matching ``bin_edges``."""
    return pd.DataFrame(
        {
            "bin_start": [0.0, 4.0, 8.0],
            "bin_end": [4.0, 8.0, 10.0],
            "n_samples": [100, 100, 50],
            "mean_zscore": [1.5, 5.5, 9.0],
        },
        index=pd.RangeIndex(3, name="bin"),
    )


@pytest.fixture
def binned_covariates():
    """The akinesia bin means that pair with ``binned_metrics``."""
    return pd.DataFrame(
        {
            "bin_start": [0.0, 4.0, 8.0],
            "bin_end": [4.0, 8.0, 10.0],
            "akinesia": [2.0, 4.0, 10.0],
            "n_samples_akinesia": [2, 1, 1],
        },
        index=pd.RangeIndex(3, name="bin"),
    )


def _correlations_for(*, metric_values, covariate_values):
    """Correlate one metric against one covariate over however many bins are given."""
    n_bins = len(metric_values)
    geometry = {
        "bin_start": np.arange(n_bins, dtype=float),
        "bin_end": np.arange(1, n_bins + 1, dtype=float),
    }
    metrics = pd.DataFrame({**geometry, "n_samples": [10] * n_bins, "mean_zscore": metric_values})
    covariates = pd.DataFrame({**geometry, "akinesia": covariate_values, "n_samples_akinesia": [1] * n_bins})
    return compute_covariate_correlations(binned_metrics=metrics, binned_covariates=covariates)


class TestComputeCovariateCorrelations:
    def test_baseline_coefficients(self, binned_metrics, binned_covariates):
        result = compute_covariate_correlations(binned_metrics=binned_metrics, binned_covariates=binned_covariates)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["metric"] == "mean_zscore"
        assert row["covariate"] == "akinesia"
        assert row["pearson_r"] == pytest.approx(0.9493907, abs=1e-6)
        assert row["spearman_rho"] == pytest.approx(1.0, abs=1e-9)
        assert row["n_bins"] == 3

    def test_reports_no_p_value(self, binned_metrics, binned_covariates):
        result = compute_covariate_correlations(binned_metrics=binned_metrics, binned_covariates=binned_covariates)

        assert list(result.columns) == ["metric", "covariate", "pearson_r", "spearman_rho", "n_bins"]
        assert result.index.name == "pair"

    def test_perfect_linear_relationship(self):
        result = _correlations_for(metric_values=[4.0, 7.0, 10.0], covariate_values=[1.0, 2.0, 3.0])

        assert result.iloc[0]["pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert result.iloc[0]["spearman_rho"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_inverse_relationship(self):
        result = _correlations_for(metric_values=[3.0, 1.0, -1.0], covariate_values=[1.0, 2.0, 3.0])

        assert result.iloc[0]["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
        assert result.iloc[0]["spearman_rho"] == pytest.approx(-1.0, abs=1e-9)

    def test_monotone_nonlinear_separates_pearson_from_spearman(self):
        result = _correlations_for(metric_values=[1.0, 4.0, 9.0, 100.0], covariate_values=[1.0, 2.0, 3.0, 4.0])

        assert result.iloc[0]["pearson_r"] == pytest.approx(0.8159778, abs=1e-6)
        assert result.iloc[0]["spearman_rho"] == pytest.approx(1.0, abs=1e-9)

    def test_nan_metric_bin_is_dropped_pairwise(self):
        result = _correlations_for(metric_values=[1.5, np.nan, 9.0, 4.0], covariate_values=[2.0, 99.0, 10.0, 6.0])

        assert result.iloc[0]["n_bins"] == 3
        assert result.iloc[0]["pearson_r"] == pytest.approx(0.9819805, abs=1e-6)

    def test_too_few_paired_bins_yields_nan_coefficients(self):
        result = _correlations_for(metric_values=[1.5, 9.0, np.nan], covariate_values=[2.0, 10.0, 6.0])

        assert result.iloc[0]["n_bins"] == 2
        assert np.isnan(result.iloc[0]["pearson_r"])
        assert np.isnan(result.iloc[0]["spearman_rho"])

    def test_constant_covariate_yields_nan_coefficients(self):
        result = _correlations_for(metric_values=[1.5, 5.5, 9.0], covariate_values=[3.0, 3.0, 3.0])

        assert result.iloc[0]["n_bins"] == 3
        assert np.isnan(result.iloc[0]["pearson_r"])
        assert np.isnan(result.iloc[0]["spearman_rho"])

    def test_every_metric_covariate_pair_is_reported(self):
        geometry = {"bin_start": [0.0, 1.0, 2.0], "bin_end": [1.0, 2.0, 3.0]}
        metrics = pd.DataFrame(
            {
                **geometry,
                "n_samples": [10, 10, 10],
                "mean_zscore": [1.5, 5.5, 9.0],
                "mean_dff": [0.1, 0.2, 0.3],
                "transient_count_z_score": [1, 4, 7],
            }
        )
        covariates = pd.DataFrame(
            {
                **geometry,
                "akinesia": [2.0, 4.0, 10.0],
                "tremor": [0.0, 2.0, 4.0],
                "n_samples_akinesia": [1, 1, 1],
                "n_samples_tremor": [1, 1, 1],
            }
        )

        result = compute_covariate_correlations(binned_metrics=metrics, binned_covariates=covariates)

        assert len(result) == 6
        assert list(result["metric"]) == [
            "mean_dff",
            "mean_dff",
            "mean_zscore",
            "mean_zscore",
            "transient_count_z_score",
            "transient_count_z_score",
        ]
        assert list(result["covariate"]) == ["akinesia", "tremor"] * 3
