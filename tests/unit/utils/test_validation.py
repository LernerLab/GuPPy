import os

import numpy as np
import pytest

from guppy.utils.validation import (
    validate_non_negative,
    validate_peak_windows,
    validate_positive,
    validate_required_folder_selection,
    validate_same_parent_directory,
    validate_window_bounds,
)


class TestValidatePositive:
    def test_positive_value_returns_none(self):
        assert validate_positive(value=3, name="moving_window") is None

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="moving_window=0 must be greater than 0"):
            validate_positive(value=0, name="moving_window")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="highAmpFilt=-2 must be greater than 0"):
            validate_positive(value=-2, name="highAmpFilt")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="transientsThresh='x' is not a valid number"):
            validate_positive(value="x", name="transientsThresh")

    def test_bool_rejected_as_non_numeric(self):
        with pytest.raises(ValueError, match="is not a valid number"):
            validate_positive(value=True, name="numberOfCores")


class TestValidateNonNegative:
    def test_positive_value_returns_none(self):
        assert validate_non_negative(value=100, name="filter_window") is None

    def test_zero_returns_none(self):
        assert validate_non_negative(value=0, name="filter_window") is None

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="filter_window=-1 must be 0 or greater"):
            validate_non_negative(value=-1, name="filter_window")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="timeForLightsTurnOn='' is not a valid number"):
            validate_non_negative(value="", name="timeForLightsTurnOn")


class TestValidateWindowBounds:
    def test_valid_window_returns_none(self):
        assert (
            validate_window_bounds(
                start=0.0,
                end=5.0,
                ts_min=-2.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )
            is None
        )

    def test_non_numeric_start_raises(self):
        with pytest.raises(ValueError, match="windowStart='abc' is not a valid number"):
            validate_window_bounds(
                start="abc",
                end=5.0,
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )

    def test_nan_end_raises(self):
        with pytest.raises(ValueError, match="windowEnd=nan is not a valid number"):
            validate_window_bounds(
                start=0.0,
                end=float("nan"),
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )

    def test_start_equal_to_end_raises(self):
        with pytest.raises(ValueError, match="windowStart=5 must be strictly less than windowEnd=5"):
            validate_window_bounds(
                start=5,
                end=5,
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValueError, match="windowStart=7 must be strictly less than windowEnd=3"):
            validate_window_bounds(
                start=7,
                end=3,
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )

    def test_start_before_ts_min_raises_with_range(self):
        with pytest.raises(ValueError, match=r"windowStart=-1 is before the signal start 0s"):
            validate_window_bounds(
                start=-1,
                end=3,
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )

    def test_end_after_ts_max_includes_range_label(self):
        with pytest.raises(ValueError, match=r"PSTH window is \[-10, 20\]s"):
            validate_window_bounds(
                start=-5,
                end=25,
                ts_min=-10.0,
                ts_max=20.0,
                start_name="baselineCorrectionStart",
                end_name="baselineCorrectionEnd",
                range_label="PSTH window",
            )

    def test_bool_rejected_as_non_numeric(self):
        with pytest.raises(ValueError, match="windowStart=True is not a valid number"):
            validate_window_bounds(
                start=True,
                end=5.0,
                ts_min=0.0,
                ts_max=10.0,
                start_name="windowStart",
                end_name="windowEnd",
            )


class TestValidatePeakWindows:
    def test_valid_pair_returns_cleaned_arrays(self):
        starts, ends = validate_peak_windows(
            peak_starts=[0.0, 1.0, np.nan],
            peak_ends=[1.0, 2.0, np.nan],
        )
        np.testing.assert_array_equal(starts, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(ends, np.array([1.0, 2.0]))

    def test_all_nan_returns_empty_arrays(self):
        starts, ends = validate_peak_windows(
            peak_starts=[np.nan, np.nan],
            peak_ends=[np.nan, np.nan],
        )
        assert starts.shape == (0,)
        assert ends.shape == (0,)

    def test_unequal_counts_raises(self):
        with pytest.raises(ValueError, match=r"unequal \(start: 2, end: 1\)"):
            validate_peak_windows(peak_starts=[0.0, 1.0], peak_ends=[2.0])

    def test_end_equal_to_start_raises_with_offending_pair(self):
        with pytest.raises(ValueError, match=r"\(start=1.0, end=1.0\)"):
            validate_peak_windows(peak_starts=[1.0], peak_ends=[1.0])

    def test_end_less_than_start_raises_with_offending_pair(self):
        with pytest.raises(ValueError, match=r"\(start=2.0, end=0.0\)"):
            validate_peak_windows(peak_starts=[2.0], peak_ends=[0.0])

    def test_one_valid_one_invalid_pair_reports_only_offender(self):
        with pytest.raises(ValueError, match=r"1 window\(s\): \(start=3.0, end=2.0\)"):
            validate_peak_windows(peak_starts=[0.0, 3.0], peak_ends=[1.0, 2.0])


class TestValidateRequiredFolderSelection:
    def test_passes_when_one_selector_has_value(self):
        class FakeSelector:
            def __init__(self, value):
                self.value = value

        selector_a = FakeSelector(value=[])
        selector_b = FakeSelector(value=["/path/to/session"])
        assert validate_required_folder_selection(file_selectors=[selector_a, selector_b]) is None

    def test_raises_when_all_selectors_empty(self):
        class FakeSelector:
            def __init__(self, value):
                self.value = value

        selector_a = FakeSelector(value=[])
        selector_b = FakeSelector(value=[])
        with pytest.raises(ValueError, match="No folder is selected for analysis"):
            validate_required_folder_selection(file_selectors=[selector_a, selector_b])


class TestValidateSameParentDirectory:
    def test_returns_single_parent_when_all_match(self):
        paths = [
            os.path.join("/data", "sessions", "s1"),
            os.path.join("/data", "sessions", "s2"),
        ]
        result = validate_same_parent_directory(paths=paths)
        assert result.shape == (1,)
        assert result[0] == os.path.join("/data", "sessions")

    def test_raises_when_paths_have_different_parents(self):
        paths = [
            os.path.join("/data", "sessions_a", "s1"),
            os.path.join("/data", "sessions_b", "s2"),
        ]
        with pytest.raises(ValueError, match="folders selected should be at the same location"):
            validate_same_parent_directory(paths=paths)
