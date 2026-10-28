import numpy as np
import pandas as pd
import pytest

from guppy.analysis.psth_average import average_psth_for_group, psth_shape_check
from guppy.analysis.psth_utils import (
    create_Df_for_cross_correlation,
    create_Df_for_psth,
)
from guppy.analysis.standard_io import (
    read_peak_and_area_from_csv,
    write_peak_and_area_to_csv,
)
from guppy.utils.utils import read_Df


def test_psth_shape_check_all_same_length_returns_unchanged():
    arrays = [np.ones(10), np.ones(10) * 2, np.ones(10) * 3]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(10))
    np.testing.assert_array_equal(result[1], np.ones(10) * 2)
    np.testing.assert_array_equal(result[2], np.ones(10) * 3)


def test_psth_shape_check_shorter_arrays_padded_with_nan():
    # First array (length 8) padded to match last element (length 10);
    # original 8 values are preserved; padding slots are NaN
    arrays = [np.ones(8), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0][:8], np.ones(8))
    assert np.all(np.isnan(result[0][8:]))
    np.testing.assert_array_equal(result[1], np.ones(10))


def test_psth_shape_check_longer_arrays_truncated():
    # First array (length 15) truncated to match last element (length 10)
    arrays = [np.ones(15), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(10))
    np.testing.assert_array_equal(result[1], np.ones(10))


def test_psth_shape_check_uses_last_element_length_not_maximum():
    # Lengths [10, 8, 12] → canonical length is 12 (the last element, not the max)
    # First array padded by 2, second padded by 4, third unchanged
    arrays = [np.ones(10), np.ones(8), np.ones(12)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0][:10], np.ones(10))
    assert np.all(np.isnan(result[0][10:]))
    np.testing.assert_array_equal(result[1][:8], np.ones(8))
    assert np.all(np.isnan(result[1][8:]))
    np.testing.assert_array_equal(result[2], np.ones(12))


def test_psth_shape_check_last_element_shorter_truncates_longer_earlier_arrays():
    # Lengths [12, 10, 8] → canonical length is 8 (the last element)
    arrays = [np.ones(12), np.ones(10), np.ones(8)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(8))
    np.testing.assert_array_equal(result[1], np.ones(8))
    np.testing.assert_array_equal(result[2], np.ones(8))


def test_psth_shape_check_single_element_returned_unchanged():
    arrays = [np.array([1.0, 2.0, 3.0])]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0]))


def test_psth_shape_check_preserves_values_of_unchanged_arrays():
    data = np.arange(10, dtype=float)
    arrays = [data.copy(), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], data)


# ── average_psth_for_group ────────────────────────────────────────────────────


@pytest.fixture
def group_folder(tmp_path):
    folder = tmp_path / "saline_group"
    folder.mkdir()
    return folder


def test_average_psth_for_group_averages_the_members_means(tmp_path, group_folder):
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    session1.mkdir(parents=True)
    session2.mkdir(parents=True)

    # Stub preprocessed traces so the recording-site glob finds them
    (session1 / "z_score_dms.hdf5").touch()
    (session2 / "z_score_dms.hdf5").touch()

    # One trial row + timestamps row, 3 timepoints. create_Df_for_psth derives each
    # session's "mean" column from its single trial, so the members' means are the
    # trial rows themselves: [1, 2, 3] and [3, 4, 5].
    columns = ["trial1", "timestamps"]
    create_Df_for_psth(
        str(session1), "event_lever_dms", "z_score_dms", np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]), columns=columns
    )
    create_Df_for_psth(
        str(session2), "event_lever_dms", "z_score_dms", np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]), columns=columns
    )

    input_parameters = {"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)}
    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters=input_parameters,
    )

    assert wrote_psth is True
    result = pd.read_csv(group_folder / "event_lever_dms_z_score_dms.csv")
    np.testing.assert_allclose(result["session1/output_1"].values, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result["session2/output_1"].values, np.array([3.0, 4.0, 5.0]))
    # Group mean is the across-session mean; err is the SEM over the 2 sessions:
    # mean = [2, 3, 4], std (population, ddof=0) = 1.0 → 1.0 / sqrt(2) ≈ 0.7071068
    np.testing.assert_allclose(result["mean"].values, np.array([2.0, 3.0, 4.0]), atol=1e-6)
    np.testing.assert_allclose(result["err"].values, np.full(3, 0.7071068), atol=1e-6)
    np.testing.assert_allclose(result["timestamps"].values, np.array([0.0, 1.0, 2.0]))


def test_average_psth_for_group_writes_into_the_given_group_folder(tmp_path, group_folder):
    session = tmp_path / "session1" / "output_1"
    session.mkdir(parents=True)
    (session / "z_score_dms.hdf5").touch()
    create_Df_for_psth(
        str(session),
        "event_lever_dms",
        "z_score_dms",
        np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]),
        columns=["trial1", "timestamps"],
    )

    average_psth_for_group(
        member_run_folders=[str(session)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    assert sorted(path.name for path in group_folder.iterdir()) == ["event_lever_dms_z_score_dms.csv"]
    # Nothing is written to an "average" directory beside the members any more.
    assert not (tmp_path / "average").exists()


def test_average_psth_for_group_reports_false_when_no_member_has_the_event(tmp_path, group_folder):
    session = tmp_path / "session1" / "output_1"
    session.mkdir(parents=True)
    (session / "z_score_dms.hdf5").touch()

    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session)],
        event="event_never_recorded",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    assert wrote_psth is False
    assert list(group_folder.iterdir()) == []


def test_average_psth_for_group_dff_mode_averages_the_dff_files(tmp_path, group_folder):
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    session1.mkdir(parents=True)
    session2.mkdir(parents=True)

    # A z-score trace is present but must be ignored in dff mode.
    for session in (session1, session2):
        (session / "z_score_dms.hdf5").touch()
        (session / "dff_dms.hdf5").touch()

    columns = ["trial1", "timestamps"]
    create_Df_for_psth(
        str(session1), "event_lever_dms", "dff_dms", np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]), columns=columns
    )
    create_Df_for_psth(
        str(session2), "event_lever_dms", "dff_dms", np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]), columns=columns
    )

    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "dff", "output_root_folder": str(tmp_path)},
    )

    assert wrote_psth is True
    assert not (group_folder / "event_lever_dms_z_score_dms.csv").exists()
    result = pd.read_csv(group_folder / "event_lever_dms_dff_dms.csv")
    np.testing.assert_allclose(result["mean"].values, np.array([2.0, 3.0, 4.0]), atol=1e-6)


def test_average_psth_for_group_both_modes_averages_z_score_and_dff(tmp_path, group_folder):
    session = tmp_path / "session1" / "output_1"
    session.mkdir(parents=True)
    (session / "z_score_dms.hdf5").touch()
    (session / "dff_dms.hdf5").touch()

    columns = ["trial1", "timestamps"]
    create_Df_for_psth(
        str(session), "event_lever_dms", "z_score_dms", np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]), columns=columns
    )
    create_Df_for_psth(
        str(session), "event_lever_dms", "dff_dms", np.array([[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]), columns=columns
    )

    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "both", "output_root_folder": str(tmp_path)},
    )

    assert wrote_psth is True
    z_score_result = pd.read_csv(group_folder / "event_lever_dms_z_score_dms.csv")
    dff_result = pd.read_csv(group_folder / "event_lever_dms_dff_dms.csv")
    np.testing.assert_allclose(z_score_result["mean"].values, np.array([1.0, 2.0, 3.0]), atol=1e-6)
    np.testing.assert_allclose(dff_result["mean"].values, np.array([7.0, 8.0, 9.0]), atol=1e-6)


def test_average_psth_for_group_averages_the_members_cross_correlations(tmp_path, group_folder):
    """Two recording sites give one correlated pair, whose per-member results are averaged."""
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    session1.mkdir(parents=True)
    session2.mkdir(parents=True)

    for session, first_lag_values in ((session1, [1.0, 2.0, 3.0]), (session2, [3.0, 4.0, 5.0])):
        (session / "z_score_dms.hdf5").touch()
        (session / "z_score_nac.hdf5").touch()
        correlation_folder = session / "cross_correlation_output"
        correlation_folder.mkdir()
        create_Df_for_cross_correlation(
            str(correlation_folder),
            "corr_event_lever",
            "z_score_dms_nac",
            np.array([first_lag_values, [0.0, 1.0, 2.0]]),
            columns=["trial1", "timestamps"],
        )

    average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    averaged = pd.read_csv(group_folder / "cross_correlation_output" / "corr_event_lever_z_score_dms_nac.csv")
    np.testing.assert_allclose(averaged["session1/output_1"].values, np.array([1.0, 2.0, 3.0]), atol=1e-6)
    np.testing.assert_allclose(averaged["session2/output_1"].values, np.array([3.0, 4.0, 5.0]), atol=1e-6)
    # Across-member mean, and the SEM over the 2 members: std (ddof=0) = 1.0 -> 1.0 / sqrt(2)
    np.testing.assert_allclose(averaged["mean"].values, np.array([2.0, 3.0, 4.0]), atol=1e-6)
    np.testing.assert_allclose(averaged["err"].values, np.full(3, 0.7071068), atol=1e-6)
    np.testing.assert_allclose(averaged["timestamps"].values, np.array([0.0, 1.0, 2.0]))


def test_average_psth_for_group_handles_non_overlapping_stores_without_indexerror(tmp_path, group_folder):
    """Sessions with entirely non-overlapping store_ids must not raise IndexError.

    Regression test for issue #274 — previously ``new_path`` was sized by the
    largest per-session file count, which could be smaller than the number of
    unique basenames and caused ``list index out of range`` at
    ``new_path[idx].append(path[i])``.
    """
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    session1.mkdir(parents=True)
    session2.mkdir(parents=True)

    # Non-overlapping recording-site labels across sessions
    (session1 / "z_score_regionA.hdf5").touch()
    (session2 / "z_score_regionB.hdf5").touch()

    psth = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    columns = ["trial1", "timestamps"]
    create_Df_for_psth(str(session1), "event_lever_regionA", "z_score_regionA", psth, columns=columns)
    create_Df_for_psth(str(session2), "event_lever_regionB", "z_score_regionB", psth, columns=columns)

    average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    assert sorted(path.name for path in group_folder.iterdir()) == [
        "event_lever_regionA_z_score_regionA.csv",
        "event_lever_regionB_z_score_regionB.csv",
    ]


def test_average_psth_for_group_averages_the_members_binned_trials(tmp_path, group_folder):
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    session1.mkdir(parents=True)
    session2.mkdir(parents=True)

    # One trial, one bin and the timestamps row over 2 timepoints per member.
    columns = ["1.5", "bin_(0-2)", "timestamps"]
    for session, trial, bin_values in ((session1, [1.0, 2.0], [2.0, 3.0]), (session2, [3.0, 4.0], [4.0, 5.0])):
        (session / "z_score_dms.hdf5").touch()
        create_Df_for_psth(
            str(session), "event_lever_dms", "z_score_dms", np.array([trial, bin_values, [0.0, 1.0]]), columns=columns
        )

    average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    assert sorted(path.name for path in group_folder.iterdir()) == [
        "binned_event_lever_dms_z_score_dms.csv",
        "event_lever_dms_z_score_dms.csv",
    ]
    result = read_Df(str(group_folder), "event_lever_dms", "z_score_dms")
    assert list(result.columns) == [
        "session1/output_1",
        "session2/output_1",
        "bin_(0-2)",
        "bin_err_(0-2)",
        "timestamps",
        "mean",
        "err",
    ]
    # Bin mean over the members: [(2+4)/2, (3+5)/2]; err is std (ddof=1) sqrt(2) over sqrt(2 members) = 1.0
    np.testing.assert_allclose(result["bin_(0-2)"].values, [3.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(result["bin_err_(0-2)"].values, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result["mean"].values, [2.0, 3.0], atol=1e-6)


@pytest.fixture
def members_with_peak_and_area(tmp_path):
    session1 = tmp_path / "session1" / "output_1"
    session2 = tmp_path / "session2" / "output_1"
    for session_number, session in ((1, session1), (2, session2)):
        session.mkdir(parents=True)
        (session / "z_score_dms.hdf5").touch()
        write_peak_and_area_to_csv(
            filepath=session,
            trials=pd.DataFrame(
                {
                    "session": [f"session{session_number}", f"session{session_number}"],
                    "trial": ["1.5", "mean"],
                    "peak_pos_1": [float(session_number), float(session_number)],
                }
            ),
            bins=pd.DataFrame(
                {"session": [f"session{session_number}"], "bin": ["bin_(0-2)"], "peak_pos_1": [10.0 * session_number]}
            ),
            name="event_lever_dms_z_score_dms",
        )
    return [str(session1), str(session2)]


def test_average_psth_for_group_concatenates_the_members_peak_and_area_tables(
    tmp_path, group_folder, members_with_peak_and_area
):
    average_psth_for_group(
        member_run_folders=members_with_peak_and_area,
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score", "output_root_folder": str(tmp_path)},
    )

    trials, bins = read_peak_and_area_from_csv(filepath=group_folder, name="event_lever_dms_z_score_dms")
    expected_trials = pd.DataFrame(
        {
            "session": ["session1", "session1", "session2", "session2"],
            "trial": ["1.5", "mean", "1.5", "mean"],
            "peak_pos_1": [1.0, 1.0, 2.0, 2.0],
        }
    )
    pd.testing.assert_frame_equal(trials, expected_trials)
    expected_bins = pd.DataFrame(
        {"session": ["session1", "session2"], "bin": ["bin_(0-2)", "bin_(0-2)"], "peak_pos_1": [10.0, 20.0]}
    )
    pd.testing.assert_frame_equal(bins, expected_bins)
