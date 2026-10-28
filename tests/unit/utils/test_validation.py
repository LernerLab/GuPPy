import re

import numpy as np
import pytest

from guppy.utils.validation import (
    validate_data_not_combined,
    validate_group_definitions,
    validate_group_folders_selected,
    validate_group_member_run_folders,
    validate_no_legacy_psth_tables,
    validate_non_negative,
    validate_output_root_folder,
    validate_peak_windows,
    validate_positive,
    validate_required_folder_selection,
    validate_significance_level,
    validate_tonic_means_present,
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


class TestValidateNoLegacyPsthTables:
    def test_passes_for_csv_psth_tables(self, member_run_folder):
        (member_run_folder / "ttl_dms_z_score_dms.csv").write_text("")

        validate_no_legacy_psth_tables(output_directories=[str(member_run_folder)])

    def test_passes_when_each_h5_psth_table_has_its_csv_beside_it(self, member_run_folder):
        # Rerunning Step 4 writes the CSV tables and leaves the earlier .h5 files in place.
        for name in ("ttl_dms_z_score_dms", "peak_AUC_ttl_dms_z_score_dms"):
            (member_run_folder / f"{name}.h5").write_bytes(b"")
            (member_run_folder / f"{name}.csv").write_text("")

        validate_no_legacy_psth_tables(output_directories=[str(member_run_folder)])

    @pytest.mark.parametrize("metric", ["z_score", "dff"])
    def test_raises_for_an_h5_psth_table_without_its_csv(self, member_run_folder, metric):
        (member_run_folder / f"ttl_dms_{metric}_dms.h5").write_bytes(b"")
        # Earlier versions wrote the peak/AUC table as both .h5 and .csv.
        (member_run_folder / f"peak_AUC_ttl_dms_{metric}_dms.h5").write_bytes(b"")
        (member_run_folder / f"peak_AUC_ttl_dms_{metric}_dms.csv").write_text("")

        with pytest.raises(ValueError, match=re.escape(str(member_run_folder))):
            validate_no_legacy_psth_tables(output_directories=[str(member_run_folder)])

    def test_names_only_the_directories_holding_h5_psth_tables(self, member_run_folder, tmp_path):
        current_run_folder = tmp_path / "sessionB" / "sessionB_output_1"
        current_run_folder.mkdir(parents=True)
        (current_run_folder / "ttl_dms_z_score_dms.csv").write_text("")
        (member_run_folder / "ttl_dms_z_score_dms.h5").write_bytes(b"")

        with pytest.raises(ValueError) as error:
            validate_no_legacy_psth_tables(output_directories=[str(current_run_folder), str(member_run_folder)])

        assert f"  - {member_run_folder}\n" in str(error.value)
        assert str(current_run_folder) not in str(error.value)


class TestValidateTonicMeansPresent:
    def test_passes_when_each_site_with_epochs_has_its_means(self, member_run_folder):
        (member_run_folder / "tonic_epochs_dms.csv").write_text("label,start,end\n")
        (member_run_folder / "tonic_means_dms.csv").write_text("")

        validate_tonic_means_present(output_directories=[str(member_run_folder)])

    def test_passes_for_the_leftover_h5_of_a_cleared_site(self, member_run_folder):
        # Clearing a site's epochs and saving removes its CSVs, but not an earlier version's .h5.
        (member_run_folder / "tonic_dms.h5").write_bytes(b"")

        validate_tonic_means_present(output_directories=[str(member_run_folder)])

    def test_raises_for_epochs_whose_means_are_only_an_h5(self, member_run_folder):
        (member_run_folder / "tonic_epochs_dms.csv").write_text("label,start,end\n")
        (member_run_folder / "tonic_dms.h5").write_bytes(b"")

        with pytest.raises(ValueError, match=re.escape(str(member_run_folder))):
            validate_tonic_means_present(output_directories=[str(member_run_folder)])


class TestValidateOutputRootFolder:
    def test_sessions_under_the_input_root_folder_pass(self, tmp_path):
        sessions = [str(tmp_path / "data" / "sessionA"), str(tmp_path / "data" / "sessionB")]
        validate_output_root_folder(
            session_folders=sessions,
            output_root_folder=str(tmp_path / "derivatives"),
            input_root_folder=str(tmp_path / "data"),
        )

    def test_sessions_sharing_a_folder_name_pass(self, tmp_path):
        sessions = [str(tmp_path / "data" / "mouse1" / "day1"), str(tmp_path / "data" / "mouse2" / "day1")]
        validate_output_root_folder(
            session_folders=sessions,
            output_root_folder=str(tmp_path / "derivatives"),
            input_root_folder=str(tmp_path / "data"),
        )

    def test_a_session_outside_the_input_root_folder_is_rejected(self, tmp_path):
        sessions = [str(tmp_path / "data" / "sessionA"), str(tmp_path / "elsewhere" / "sessionB")]
        with pytest.raises(ValueError, match="not inside the input root folder"):
            validate_output_root_folder(
                session_folders=sessions,
                output_root_folder=str(tmp_path / "derivatives"),
                input_root_folder=str(tmp_path / "data"),
            )

    def test_the_two_roots_may_be_the_same_folder(self, tmp_path):
        """Each session then mirrors onto itself, which writes its runs inside it."""
        session = str(tmp_path / "data" / "sessionA")
        validate_output_root_folder(
            session_folders=[session],
            output_root_folder=str(tmp_path / "data"),
            input_root_folder=str(tmp_path / "data"),
        )

    def test_a_session_may_sit_inside_the_output_root_folder(self, tmp_path):
        """Implied by the roots being allowed to match, and true whenever they do."""
        session = str(tmp_path / "data" / "sessionA")
        validate_output_root_folder(
            session_folders=[session],
            output_root_folder=str(tmp_path / "data"),
            input_root_folder=str(tmp_path / "data"),
        )

    def test_the_output_root_folder_may_not_be_inside_a_selected_session(self, tmp_path):
        """That would nest each run inside the previous run's results."""
        session = str(tmp_path / "data" / "sessionA")
        with pytest.raises(ValueError, match="inside the selected session"):
            validate_output_root_folder(
                session_folders=[session],
                output_root_folder=str(tmp_path / "data" / "sessionA" / "outputs"),
                input_root_folder=str(tmp_path / "data"),
            )


class TestValidateSignificanceLevel:
    @pytest.mark.parametrize("value", [0.05, 0.01, 0.1, 0.5])
    def test_accepts_a_level_between_zero_and_one(self, value):
        validate_significance_level(value=value, name="psthSignificanceAlpha")

    @pytest.mark.parametrize("value", [0, 1, -0.05, 1.5, np.nan, "0.05", None, True])
    def test_rejects_anything_outside_the_open_unit_interval(self, value):
        with pytest.raises(ValueError, match="not a valid significance level"):
            validate_significance_level(value=value, name="psthSignificanceAlpha")
