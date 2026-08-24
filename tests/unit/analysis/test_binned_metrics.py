import warnings

import numpy as np
import pytest

from guppy.analysis.binned_metrics import compute_bin_edges, compute_binned_metrics


class TestComputeBinEdges:
    def test_exact_multiple(self):
        # 10 s at 5 s bins divides evenly into two whole bins.
        np.testing.assert_allclose(compute_bin_edges(start=0.0, end=10.0, bin_width=5.0), [0.0, 5.0, 10.0])

    def test_ragged_final_bin_is_kept(self):
        # 10 s at 4 s bins -> bins [0,4), [4,8), and a short [8,10].
        np.testing.assert_allclose(compute_bin_edges(start=0.0, end=10.0, bin_width=4.0), [0.0, 4.0, 8.0, 10.0])

    def test_bin_width_longer_than_recording_gives_one_bin(self):
        np.testing.assert_allclose(compute_bin_edges(start=0.0, end=10.0, bin_width=100.0), [0.0, 10.0])

    def test_edges_start_at_the_first_timestamp(self):
        # Bins are anchored to the recording, not to zero.
        np.testing.assert_allclose(compute_bin_edges(start=2.0, end=8.0, bin_width=3.0), [2.0, 5.0, 8.0])

    def test_floating_point_sliver_is_absorbed(self):
        # 0.1 + 0.2 == 0.30000000000000004, so the duration divides into 3 bins
        # but rounds up to 4, leaving a zero-width final bin that np.histogram
        # would reject.
        end = 0.1 + 0.2
        edges = compute_bin_edges(start=0.0, end=end, bin_width=0.1)

        assert edges.shape[0] == 4
        assert np.all(np.diff(edges) > 0)
        assert edges[-1] == end

    def test_non_positive_bin_width_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            compute_bin_edges(start=0.0, end=10.0, bin_width=0.0)

    def test_recording_without_duration_raises(self):
        with pytest.raises(ValueError, match="no duration to bin"):
            compute_bin_edges(start=4.0, end=4.0, bin_width=1.0)


class TestComputeBinnedMetrics:
    @pytest.fixture
    def traces(self):
        timestamps = np.arange(0, 11, 1, dtype=float)
        return {
            "timestamps": timestamps,
            "z_score": timestamps.copy(),
            "dff": timestamps / 10,
        }

    def test_means_over_whole_bins(self, traces):
        # bin 0 = [0,5) -> samples 0,1,2,3,4 -> z 2.0, dff 0.2
        # bin 1 = [5,10] -> samples 5..10 (final bin includes its end) -> z 7.5, dff 0.75
        binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=5.0)

        np.testing.assert_allclose(binned["bin_start"].to_numpy(), [0.0, 5.0])
        np.testing.assert_allclose(binned["bin_end"].to_numpy(), [5.0, 10.0])
        np.testing.assert_allclose(binned["mean_zscore"].to_numpy(), [2.0, 7.5])
        np.testing.assert_allclose(binned["mean_dff"].to_numpy(), [0.2, 0.75])
        np.testing.assert_array_equal(binned["n_samples"].to_numpy(), [5, 6])
        assert binned.index.name == "bin"
        np.testing.assert_array_equal(binned.index.to_numpy(), [0, 1])

    def test_ragged_final_bin_reports_its_true_bounds(self, traces):
        # bin 0 = [0,4) -> 0,1,2,3 -> 1.5;  bin 1 = [4,8) -> 4,5,6,7 -> 5.5
        # bin 2 = [8,10] -> 8,9,10 -> 9.0, and is only 2 s wide
        binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=4.0)

        np.testing.assert_allclose(binned["bin_start"].to_numpy(), [0.0, 4.0, 8.0])
        np.testing.assert_allclose(binned["bin_end"].to_numpy(), [4.0, 8.0, 10.0])
        np.testing.assert_allclose(binned["mean_zscore"].to_numpy(), [1.5, 5.5, 9.0])
        np.testing.assert_array_equal(binned["n_samples"].to_numpy(), [4, 4, 3])

    def test_transient_counts_follow_the_bin_convention(self, traces):
        # Edges are [0,4,8,10]. 3.9 falls in bin 0 and 4.0 opens bin 1 (half-open),
        # while 10.0 lands in the final bin because that one includes its end.
        occurrences = np.array([0.5, 3.9, 4.0, 9.9, 10.0])
        binned = compute_binned_metrics(**traces, transient_timestamps={"z_score": occurrences}, bin_width=4.0)

        np.testing.assert_array_equal(binned["transient_count_z_score"].to_numpy(), [2, 1, 2])
        assert binned["transient_count_z_score"].sum() == occurrences.shape[0]

    def test_one_count_column_per_metric(self, traces):
        binned = compute_binned_metrics(
            **traces,
            transient_timestamps={"z_score": np.array([1.0]), "dff": np.array([6.0, 7.0])},
            bin_width=5.0,
        )

        np.testing.assert_array_equal(binned["transient_count_z_score"].to_numpy(), [1, 0])
        np.testing.assert_array_equal(binned["transient_count_dff"].to_numpy(), [0, 2])

    def test_no_count_columns_when_no_transients_were_detected_on_a_metric(self, traces):
        binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=5.0)

        assert [column for column in binned.columns if column.startswith("transient_count_")] == []

    def test_zero_detected_transients_counts_zero(self, traces):
        binned = compute_binned_metrics(**traces, transient_timestamps={"z_score": np.array([])}, bin_width=5.0)

        np.testing.assert_array_equal(binned["transient_count_z_score"].to_numpy(), [0, 0])

    def test_bin_lost_to_artifact_removal_is_nan_without_warning(self, traces):
        # Artifact removal NaNs the samples in place, so a bin can be entirely NaN.
        traces["z_score"][0:4] = np.nan
        traces["dff"][0:4] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=4.0)

        assert np.isnan(binned["mean_zscore"].iloc[0])
        assert np.isnan(binned["mean_dff"].iloc[0])
        assert binned["n_samples"].iloc[0] == 0
        # The surviving bins are unaffected: [4,8) -> 4,5,6,7 and [8,10] -> 8,9,10.
        np.testing.assert_allclose(binned["mean_zscore"].to_numpy()[1:], [5.5, 9.0])

    def test_partially_removed_bin_averages_only_the_surviving_samples(self, traces):
        # Drop samples 0 and 1 from bin [0,4); the mean is then over 2 and 3.
        traces["z_score"][0:2] = np.nan
        traces["dff"][0:2] = np.nan

        binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=4.0)

        np.testing.assert_allclose(binned["mean_zscore"].iloc[0], 2.5)
        assert binned["n_samples"].iloc[0] == 2

    def test_bin_width_longer_than_recording_gives_a_single_bin(self, traces):
        binned = compute_binned_metrics(**traces, transient_timestamps={}, bin_width=100.0)

        assert binned.shape[0] == 1
        # Mean of 0..10 inclusive.
        np.testing.assert_allclose(binned["mean_zscore"].iloc[0], 5.0)
        np.testing.assert_allclose(binned["bin_end"].iloc[0], 10.0)

    def test_column_order(self, traces):
        binned = compute_binned_metrics(**traces, transient_timestamps={"z_score": np.array([1.0])}, bin_width=5.0)

        assert list(binned.columns) == [
            "bin_start",
            "bin_end",
            "n_samples",
            "mean_zscore",
            "mean_dff",
            "transient_count_z_score",
        ]

    def test_non_positive_bin_width_raises(self, traces):
        with pytest.raises(ValueError, match="must be positive"):
            compute_binned_metrics(**traces, transient_timestamps={}, bin_width=-1.0)
