import numpy as np

from guppy.analysis.compute_psth import baselineCorrection, compute_psth, rowFormation

# rowFormation is called from compute_psth with positive nTsPrev
# (the caller negates a negative nSecPrev back to positive before passing here).
# All tests use positive nTsPrev as the function expects.


def test_row_formation_branch1_normal_case_returns_correct_slice():
    # Branch 1: thisIndex > nTsPrev AND thisIndex + nTsPost < len(z_score)
    z_score = np.arange(200, dtype=float)
    this_index = 100
    n_ts_prev = 50
    n_ts_post = 50
    result = rowFormation(z_score, this_index, n_ts_prev, n_ts_post)
    assert result.shape[0] == n_ts_prev + n_ts_post + 1
    # Verify exact slice: z_score[thisIndex - nTsPrev - 1 : thisIndex + nTsPost]
    expected = z_score[this_index - n_ts_prev - 1 : this_index + n_ts_post]
    np.testing.assert_array_equal(result, expected)


def test_row_formation_branch2_short_pre_leading_nan_padding():
    # Branch 2: thisIndex <= nTsPrev AND thisIndex + nTsPost < len(z_score)
    z_score = np.arange(200, dtype=float)
    this_index = 20
    n_ts_prev = 50
    n_ts_post = 50
    result = rowFormation(z_score, this_index, n_ts_prev, n_ts_post)
    assert result.shape[0] == n_ts_prev + n_ts_post + 1
    # Leading elements should be NaN
    mismatch = n_ts_prev - this_index + 1
    assert np.all(np.isnan(result[:mismatch]))
    # Remaining elements come from the start of z_score
    np.testing.assert_array_equal(result[mismatch:], z_score[: this_index + n_ts_post])


def test_row_formation_branch3_short_both_nan_padding_on_both_sides():
    # Branch 3: thisIndex <= nTsPrev AND thisIndex + nTsPost >= len(z_score)
    z_score = np.arange(80, dtype=float)
    this_index = 20
    n_ts_prev = 50
    n_ts_post = 70
    result = rowFormation(z_score, this_index, n_ts_prev, n_ts_post)
    assert result.shape[0] == n_ts_prev + n_ts_post + 1
    mismatch_front = n_ts_prev - this_index + 1
    mismatch_back = (this_index + n_ts_post) - z_score.shape[0]
    assert np.all(np.isnan(result[:mismatch_front]))
    assert np.all(np.isnan(result[-mismatch_back:]))
    # Middle should be the full z_score
    np.testing.assert_array_equal(result[mismatch_front : result.shape[0] - mismatch_back], z_score)


def test_row_formation_branch4_short_post_trailing_nan_padding():
    # Branch 4: thisIndex > nTsPrev AND thisIndex + nTsPost >= len(z_score)
    z_score = np.arange(100, dtype=float)
    this_index = 60
    n_ts_prev = 50
    n_ts_post = 70
    result = rowFormation(z_score, this_index, n_ts_prev, n_ts_post)
    assert result.shape[0] == n_ts_prev + n_ts_post + 1
    mismatch = (this_index + n_ts_post) - z_score.shape[0]
    assert np.all(np.isnan(result[-mismatch:]))
    # Leading part should come from z_score
    expected_data = z_score[this_index - n_ts_prev - 1 : z_score.shape[0]]
    np.testing.assert_array_equal(result[: result.shape[0] - mismatch], expected_data)


def test_baseline_correction_passthrough_when_both_zero():
    time_axis = np.linspace(-5, 5, 101)
    arr = np.arange(101, dtype=float)
    result = baselineCorrection(arr, time_axis, 0, 0)
    np.testing.assert_array_equal(result, arr)


def test_baseline_correction_constant_array_returns_zeros():
    time_axis = np.linspace(-5, 5, 101)
    arr = np.full(101, 7.0)
    result = baselineCorrection(arr, time_axis, -5.0, 0.0)
    np.testing.assert_allclose(result, np.zeros(101), atol=1e-10)


def test_baseline_correction_baseline_window_mean_is_zero():
    rng = np.random.default_rng(seed=10)
    time_axis = np.linspace(-5, 5, 1001)
    arr = rng.standard_normal(1001) + 3.0
    result = baselineCorrection(arr, time_axis, -5.0, 0.0)
    baseline_start_point = np.where(time_axis >= -5.0)[0][0]
    baseline_end_point = np.where(time_axis >= 0.0)[0][0]
    np.testing.assert_allclose(np.nanmean(result[baseline_start_point:baseline_end_point]), 0.0, atol=1e-10)


# compute_psth tests
# Parameters used throughout: nSecPrev=-1.0, nSecPost=1.0, sampling_rate=10.0
# → nTsPrev=-10, nTsPost=10, totalTs=20, row length=21
# baselineStart=0/baselineEnd=0 → passthrough (no baseline subtraction)


def test_compute_psth_single_timestamp_no_corrections_returns_expected_row():
    # Constant z_score=3.0, single timestamp at t=5s → thisIndex=50
    # rowFormation branch 1: z_score[39:60] = 21 threes
    # baselineCorrection passthrough → psth[0,:] = all 3.0
    z_score = np.ones(100) * 3.0
    ts = np.array([5.0])
    psth, _, columns, returned_ts = compute_psth(
        z_score=z_score,
        event="test",
        filepath="",
        nSecPrev=-1.0,
        nSecPost=1.0,
        timeInterval=0.0,
        bin_psth_trials=0,
        use_time_or_trials="none",
        baselineStart=0,
        baselineEnd=0,
        naming="",
        just_use_signal=False,
        sampling_rate=10.0,
        ts=ts,
        corrected_timestamps=ts,
    )
    np.testing.assert_allclose(psth[0, :], np.full(21, 3.0))
    # Last row is the time axis; first value is nSecPrev = -1.0
    np.testing.assert_allclose(psth[-1, 0], -1.0)
    np.testing.assert_array_equal(returned_ts, np.array([5.0]))
    assert columns == [5.0, "timestamps"]


def test_compute_psth_early_timestamps_filtered_by_baseline_window():
    # ts[0]=1.0 < abs(baselineStart=-2.0)=2.0 → dropped; ts[1]=5.0 >= 2.0 → kept
    z_score = np.ones(200) * 1.0
    ts = np.array([1.0, 5.0])
    _, _, _, returned_ts = compute_psth(
        z_score=z_score,
        event="test",
        filepath="",
        nSecPrev=-2.0,
        nSecPost=2.0,
        timeInterval=0.0,
        bin_psth_trials=0,
        use_time_or_trials="none",
        baselineStart=-2.0,
        baselineEnd=0.0,
        naming="",
        just_use_signal=False,
        sampling_rate=10.0,
        ts=ts,
        corrected_timestamps=ts,
    )
    np.testing.assert_array_equal(returned_ts, np.array([5.0]))


def test_compute_psth_burst_timestamps_within_time_interval_are_dropped():
    # ts[1]=5.5: 5.5-5.0=0.5 < timeInterval=1.0 → dropped
    # ts[2]=8.0: 8.0-5.0=3.0 >= 1.0 → kept
    z_score = np.ones(200) * 1.0
    ts = np.array([5.0, 5.5, 8.0])
    _, _, _, returned_ts = compute_psth(
        z_score=z_score,
        event="test",
        filepath="",
        nSecPrev=-1.0,
        nSecPost=1.0,
        timeInterval=1.0,
        bin_psth_trials=0,
        use_time_or_trials="none",
        baselineStart=0,
        baselineEnd=0,
        naming="",
        just_use_signal=False,
        sampling_rate=10.0,
        ts=ts,
        corrected_timestamps=ts,
    )
    np.testing.assert_array_equal(returned_ts, np.array([5.0, 8.0]))


def test_compute_psth_binning_by_trials_produces_correct_bin_mean_and_sem():
    # 4 timestamps → bin_steps=[0,2,4] → 2 bins of 2 trials each
    # All trials are constant 3.0 → bin mean=3.0, sem=std/sqrt(2)=0.0
    # psth rows: 4 trial + 2 mean + 2 sem + 1 timeAxis = 9 rows
    z_score = np.ones(200) * 3.0
    ts = np.array([5.0, 6.0, 7.0, 8.0])
    psth, _, columns, _ = compute_psth(
        z_score=z_score,
        event="test",
        filepath="",
        nSecPrev=-1.0,
        nSecPost=1.0,
        timeInterval=0.0,
        bin_psth_trials=2,
        use_time_or_trials="# of trials",
        baselineStart=0,
        baselineEnd=0,
        naming="",
        just_use_signal=False,
        sampling_rate=10.0,
        ts=ts,
        corrected_timestamps=ts,
    )
    # Row 4: mean of first bin (trials 0 and 1, all 3.0)
    np.testing.assert_allclose(psth[4, :], np.full(21, 3.0))
    # Row 5: sem of first bin (std of two identical rows = 0)
    np.testing.assert_allclose(psth[5, :], np.zeros(21), atol=1e-10)
    assert "bin_(0-2)" in columns
    assert "bin_err_(0-2)" in columns


def test_compute_psth_just_use_signal_true_z_scores_each_trial():
    # With just_use_signal=True each trial row is z-scored:
    # result = (arr - nanmean(arr)) / nanstd(arr) → mean=0, population std=1
    rng = np.random.default_rng(seed=42)
    z_score = rng.standard_normal(200)
    ts = np.array([10.0])
    psth, _, _, _ = compute_psth(
        z_score=z_score,
        event="test",
        filepath="",
        nSecPrev=-1.0,
        nSecPost=1.0,
        timeInterval=0.0,
        bin_psth_trials=0,
        use_time_or_trials="none",
        baselineStart=0,
        baselineEnd=0,
        naming="",
        just_use_signal=True,
        sampling_rate=10.0,
        ts=ts,
        corrected_timestamps=ts,
    )
    np.testing.assert_allclose(np.nanmean(psth[0, :]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.nanstd(psth[0, :]), 1.0, atol=1e-10)
