import numpy as np

from guppy.utils.artifact_windows import (
    complement_windows,
    coords_to_windows,
    merge_windows,
    windows_to_coords,
)


class TestMergeWindows:
    def test_disjoint_windows_are_sorted_and_unchanged(self):
        assert merge_windows(windows=[(5.0, 6.0), (1.0, 2.0)]) == [(1.0, 2.0), (5.0, 6.0)]

    def test_overlapping_windows_fuse(self):
        assert merge_windows(windows=[(1.0, 4.0), (3.0, 6.0)]) == [(1.0, 6.0)]

    def test_touching_windows_fuse(self):
        assert merge_windows(windows=[(1.0, 3.0), (3.0, 5.0)]) == [(1.0, 5.0)]

    def test_contained_window_is_absorbed(self):
        assert merge_windows(windows=[(1.0, 9.0), (3.0, 4.0)]) == [(1.0, 9.0)]

    def test_empty_input(self):
        assert merge_windows(windows=[]) == []


class TestComplementWindows:
    def test_empty_windows_yields_the_full_span(self):
        assert complement_windows(windows=[], span_start=-1.0, span_end=11.0) == [(-1.0, 11.0)]

    def test_single_interior_window_splits_the_span(self):
        result = complement_windows(windows=[(3.0, 5.0)], span_start=-1.0, span_end=11.0)
        assert result == [(-1.0, 3.0), (5.0, 11.0)]

    def test_two_windows_yield_three_keep_segments(self):
        result = complement_windows(windows=[(2.0, 4.0), (7.0, 8.0)], span_start=0.0, span_end=10.0)
        assert result == [(0.0, 2.0), (4.0, 7.0), (8.0, 10.0)]

    def test_window_at_the_start_drops_the_leading_segment(self):
        assert complement_windows(windows=[(0.0, 4.0)], span_start=0.0, span_end=10.0) == [(4.0, 10.0)]

    def test_window_at_the_end_drops_the_trailing_segment(self):
        assert complement_windows(windows=[(6.0, 10.0)], span_start=0.0, span_end=10.0) == [(0.0, 6.0)]

    def test_overlapping_windows_are_merged_before_inverting(self):
        result = complement_windows(windows=[(2.0, 5.0), (4.0, 7.0)], span_start=0.0, span_end=10.0)
        assert result == [(0.0, 2.0), (7.0, 10.0)]

    def test_full_coverage_leaves_nothing(self):
        assert complement_windows(windows=[(0.0, 10.0)], span_start=0.0, span_end=10.0) == []

    def test_inverting_a_full_span_keep_window_yields_no_artifacts(self):
        """Round-tripping the saved coords of an unmarked run must not invent a window."""
        assert complement_windows(windows=[(-1.0, 11.0)], span_start=-1.0, span_end=11.0) == []


class TestWindowsToCoords:
    def test_interleaves_starts_and_ends_with_a_zero_placeholder_column(self):
        coords = windows_to_coords(windows=[(1.0, 2.0), (5.0, 6.0)])
        np.testing.assert_array_equal(coords, np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [6.0, 0.0]]))


class TestCoordsToWindows:
    def test_converts_pairs_to_float_tuples(self):
        coords = np.array([[-1.0, 3.0], [5.0, 11.0]])
        assert coords_to_windows(coords=coords) == [(-1.0, 3.0), (5.0, 11.0)]

    def test_round_trips_through_the_on_disk_layout(self):
        """windows_to_coords writes column 0 interleaved; fetchCoords reshapes it back into pairs."""
        windows = [(-1.0, 3.0), (5.0, 11.0)]
        saved = windows_to_coords(windows=windows)
        reshaped = saved[:, 0].reshape(-1, 2)
        assert coords_to_windows(coords=reshaped) == windows
