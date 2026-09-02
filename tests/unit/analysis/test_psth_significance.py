import numpy as np
import pytest

from guppy.analysis.psth_significance import (
    BOOTSTRAP_SEED,
    bootstrap_difference_confidence_interval,
    bootstrap_mean_confidence_interval,
    minimum_consecutive_samples_for,
    significant_sample_mask,
)


@pytest.fixture
def rng():
    return np.random.default_rng(BOOTSTRAP_SEED)


class TestMinimumConsecutiveSamplesFor:
    def test_is_twice_the_filter_window(self):
        # The sampling rate cancels out of the original derivation, leaving 2 * filter_window.
        assert minimum_consecutive_samples_for(filter_window=100) == 200
        assert minimum_consecutive_samples_for(filter_window=1) == 2


class TestSignificantSampleMask:
    def test_keeps_a_fully_contiguous_run(self):
        # A window that is significant end to end has no gaps to split it on. Reported as
        # entirely non-significant before the run detection was rewritten.
        lower = np.full(50, 0.5)
        upper = np.full(50, 1.5)

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=10)

        assert mask.sum() == 50

    def test_drops_a_run_of_exactly_the_threshold(self):
        # 10 significant samples against a threshold of 10: kept only if strictly longer.
        lower, upper = np.full(50, -1.0), np.full(50, 1.0)
        lower[20:30], upper[20:30] = 0.5, 1.5

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=10)

        assert mask.sum() == 0

    def test_keeps_a_run_one_sample_over_the_threshold(self):
        lower, upper = np.full(50, -1.0), np.full(50, 1.0)
        lower[20:31], upper[20:31] = 0.5, 1.5

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=10)

        assert mask.sum() == 11
        np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(20, 31))

    def test_keeps_only_the_runs_that_are_long_enough(self):
        # One 3-sample run and one 12-sample run, threshold 10: only the second survives.
        lower, upper = np.full(60, -1.0), np.full(60, 1.0)
        lower[5:8], upper[5:8] = 0.5, 1.5
        lower[30:42], upper[30:42] = 0.5, 1.5

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=10)

        np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(30, 42))

    def test_marks_intervals_entirely_below_zero(self):
        # A negative-going response is as significant as a positive-going one.
        lower, upper = np.full(50, -1.0), np.full(50, 1.0)
        lower[10:30], upper[10:30] = -2.0, -0.5

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=5)

        np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(10, 30))

    def test_an_interval_containing_zero_is_not_significant(self):
        lower, upper = np.full(50, -1.0), np.full(50, 1.0)

        mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=1)

        assert mask.sum() == 0


class TestBootstrapMeanConfidenceInterval:
    def test_brackets_the_sample_mean(self, rng):
        samples = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0], [5.0, 14.0]])

        lower, upper = bootstrap_mean_confidence_interval(samples=samples, rng=rng, num_resamples=200)

        # Column means are 3.0 and 12.0; a CI on the mean must contain it.
        assert lower[0] <= 3.0 <= upper[0]
        assert lower[1] <= 12.0 <= upper[1]
        assert lower.shape == (2,)

    def test_every_trial_can_be_resampled(self, rng):
        # Trial 0 was excluded from resampling by the original 1-indexed translation, so two
        # datasets differing only in trial 0 gave identical intervals. They must differ now.
        baseline = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        with_outlier = baseline.copy()
        with_outlier[0, 0] = 100.0

        baseline_lower, baseline_upper = bootstrap_mean_confidence_interval(
            samples=baseline, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=200
        )
        outlier_lower, outlier_upper = bootstrap_mean_confidence_interval(
            samples=with_outlier, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=200
        )

        assert outlier_upper[0] > baseline_upper[0]
        assert (outlier_upper[0] - outlier_lower[0]) > (baseline_upper[0] - baseline_lower[0])

    def test_ignores_nan_entries(self, rng):
        # Artifact removal leaves NaN in real trials; they must not poison the whole column.
        samples = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [np.nan]])

        lower, upper = bootstrap_mean_confidence_interval(samples=samples, rng=rng, num_resamples=200)

        assert np.isfinite(lower[0]) and np.isfinite(upper[0])
        # nanmean of the six finite entries is 3.5.
        assert lower[0] <= 3.5 <= upper[0]

    def test_a_degenerate_timepoint_is_not_reported_significant(self, rng):
        # BCa cannot be computed where every trial carries the same value, and returns NaN.
        # A NaN bound compares False, so such a timepoint falls out as non-significant
        # rather than being reported either way.
        samples = np.full((6, 1), 7.0)

        lower, upper = bootstrap_mean_confidence_interval(samples=samples, rng=rng, num_resamples=200)

        assert not ((lower > 0) | (upper < 0))[0]

    def test_is_reproducible_for_a_fixed_seed(self):
        samples = np.arange(20, dtype=float).reshape(5, 4)

        first = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=200
        )
        second = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=200
        )

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_a_stricter_alpha_widens_the_interval(self):
        samples = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])

        wide = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=400, significance_level=0.2
        )
        narrow = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=400, significance_level=0.01
        )

        assert (narrow[1][0] - narrow[0][0]) > (wide[1][0] - wide[0][0])

    def test_resample_count_changes_the_interval(self):
        # Too few resamples cannot resolve the requested tail, so the interval comes back
        # narrower than the alpha asked for. The parameter has to reach scipy for this.
        samples = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])

        coarse = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=10, significance_level=0.01
        )
        fine = bootstrap_mean_confidence_interval(
            samples=samples, rng=np.random.default_rng(BOOTSTRAP_SEED), num_resamples=1000, significance_level=0.01
        )

        assert (coarse[1][0] - coarse[0][0]) < (fine[1][0] - fine[0][0])

    def test_rejects_fewer_than_three_samples(self, rng):
        with pytest.raises(ValueError, match="at least 3 samples"):
            bootstrap_mean_confidence_interval(samples=np.array([[1.0], [2.0]]), rng=rng)


class TestBootstrapDifferenceConfidenceInterval:
    def test_brackets_the_difference_of_means(self, rng):
        samples_a = np.array([[10.0], [11.0], [12.0], [13.0], [14.0]])
        samples_b = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

        lower, upper = bootstrap_difference_confidence_interval(
            samples_a=samples_a, samples_b=samples_b, rng=rng, num_resamples=200
        )

        # Means are 12.0 and 2.0, so the difference is 10.0.
        assert lower[0] <= 10.0 <= upper[0]

    def test_accepts_unequal_sample_counts(self, rng):
        # Four trials against seventeen: the real case, where one event fires far more often.
        samples_a = np.array([[1.0], [2.0], [3.0], [4.0]])
        samples_b = np.arange(1.0, 18.0).reshape(17, 1)

        lower, upper = bootstrap_difference_confidence_interval(
            samples_a=samples_a, samples_b=samples_b, rng=rng, num_resamples=200
        )

        assert lower.shape == (1,)
        # Means are 2.5 and 9.0, so the difference is -6.5.
        assert lower[0] <= -6.5 <= upper[0]

    def test_rejects_mismatched_time_axes(self, rng):
        with pytest.raises(ValueError, match="share a time axis"):
            bootstrap_difference_confidence_interval(samples_a=np.zeros((5, 3)), samples_b=np.zeros((5, 4)), rng=rng)

    def test_rejects_fewer_than_three_samples_on_either_side(self, rng):
        with pytest.raises(ValueError, match="at least 3 samples"):
            bootstrap_difference_confidence_interval(samples_a=np.zeros((5, 2)), samples_b=np.zeros((2, 2)), rng=rng)
