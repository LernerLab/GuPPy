import numpy as np
import pytest

from guppy.utils.validation import (
    validate_data_not_combined,
    validate_group_definitions,
    validate_group_folders_selected,
    validate_group_member_run_folders,
    validate_non_negative,
    validate_peak_windows,
    validate_positive,
    validate_required_folder_selection,
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


class TestValidateDataNotCombined:
    def test_raises_when_combining(self):
        with pytest.raises(ValueError) as excinfo:
            validate_data_not_combined(combine_data=True)

        message = str(excinfo.value)
        assert "does not support combine_data=True" in message
        assert "'Combine Data?' set to False" in message

    def test_passes_when_not_combining(self):
        validate_data_not_combined(combine_data=False)


@pytest.fixture
def member_run_folder(tmp_path):
    """A valid group member: an ``_output_`` directory holding storesList.csv."""
    session = tmp_path / "sessionA"
    session.mkdir()
    run_folder = session / "sessionA_output_1"
    run_folder.mkdir()
    (run_folder / "storesList.csv").write_text("Dv1A,Dv2A\ncontrol_dms,signal_dms\n")
    return run_folder


@pytest.fixture
def group_folder(tmp_path):
    """A valid group output directory holding storesList.csv."""
    folder = tmp_path / "saline_group"
    folder.mkdir()
    (folder / "group_members.json").write_text('{"member_run_folders": []}')
    return folder


class TestValidateGroupMemberRunFolders:
    def test_passes_for_valid_member(self, member_run_folder):
        validate_group_member_run_folders(member_run_folders=[str(member_run_folder)])

    def test_raises_when_empty(self):
        with pytest.raises(ValueError, match="No member runs selected"):
            validate_group_member_run_folders(member_run_folders=[])

    def test_raises_for_session_folder_instead_of_run_folder(self, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        with pytest.raises(ValueError, match="must be output directories"):
            validate_group_member_run_folders(member_run_folders=[str(session)])

    def test_raises_for_missing_directory(self, tmp_path):
        with pytest.raises(ValueError, match="do not exist"):
            validate_group_member_run_folders(member_run_folders=[str(tmp_path / "gone_output_1")])

    def test_raises_when_stores_list_missing(self, tmp_path):
        run_folder = tmp_path / "sessionA_output_1"
        run_folder.mkdir()
        with pytest.raises(ValueError, match="missing storesList.csv"):
            validate_group_member_run_folders(member_run_folders=[str(run_folder)])


class TestValidateGroupDefinitions:
    def test_passes_for_valid_group(self, group_folder):
        validate_group_definitions(group_folders=[str(group_folder)])

    def test_passes_for_empty_selection(self):
        validate_group_definitions(group_folders=[])

    def test_raises_for_run_folder(self, member_run_folder):
        with pytest.raises(ValueError, match="not group output directories"):
            validate_group_definitions(group_folders=[str(member_run_folder)])

    def test_raises_for_missing_directory(self, tmp_path):
        with pytest.raises(ValueError, match="do not exist"):
            validate_group_definitions(group_folders=[str(tmp_path / "gone_group")])

    def test_raises_when_manifest_missing(self, tmp_path):
        folder = tmp_path / "saline_group"
        folder.mkdir()
        with pytest.raises(ValueError, match="hold no group_members.json"):
            validate_group_definitions(group_folders=[str(folder)])


class TestValidateGroupFoldersSelected:
    def test_passes_for_a_selected_group(self, group_folder):
        validate_group_folders_selected(group_folders=[str(group_folder)])

    def test_raises_when_nothing_selected(self):
        with pytest.raises(ValueError, match="No groups selected"):
            validate_group_folders_selected(group_folders=[])

    def test_raises_for_an_undefined_group(self, tmp_path):
        folder = tmp_path / "saline_group"
        folder.mkdir()
        with pytest.raises(ValueError, match="hold no group_members.json"):
            validate_group_folders_selected(group_folders=[str(folder)])
