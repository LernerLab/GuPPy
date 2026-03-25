import numpy as np
import pytest

from guppy.analysis.compute_psth import baselineCorrection, rowFormation

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


@pytest.mark.parametrize(
    "this_index,n_ts_prev,n_ts_post,z_score_length",
    [
        (100, 50, 50, 200),  # branch 1
        (20, 50, 50, 200),  # branch 2
        (20, 50, 70, 80),  # branch 3
        (60, 50, 70, 100),  # branch 4
    ],
)
def test_row_formation_all_branches_output_length_invariant(this_index, n_ts_prev, n_ts_post, z_score_length):
    z_score = np.arange(z_score_length, dtype=float)
    result = rowFormation(z_score, this_index, n_ts_prev, n_ts_post)
    assert result.shape[0] == n_ts_prev + n_ts_post + 1


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


def test_baseline_correction_output_shape_equals_input_shape():
    time_axis = np.linspace(-3, 3, 61)
    arr = np.ones(61)
    result = baselineCorrection(arr, time_axis, -3.0, 0.0)
    assert result.shape == arr.shape
