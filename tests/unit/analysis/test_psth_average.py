import numpy as np
import pandas as pd
import pytest

from guppy.analysis.psth_average import (
    average_psth_for_group,
    psth_shape_check,
    read_Df_area_peak,
)
from guppy.analysis.psth_utils import create_Df_for_psth


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


# ── read_Df_area_peak ─────────────────────────────────────────────────────────


def test_read_df_area_peak_returns_dataframe_with_expected_values(tmp_path):
    # Write a peak_AUC file and verify read_Df_area_peak returns it correctly
    name = "event_lever_z_score_dms"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    df_written = pd.DataFrame(data, index=["session1", "session2"], columns=["peak", "area"])
    df_written.to_hdf(tmp_path / f"peak_AUC_{name}.h5", key="df", mode="w")

    result = read_Df_area_peak(str(tmp_path), name)

    np.testing.assert_allclose(result["peak"].values, np.array([1.0, 3.0]))
    np.testing.assert_allclose(result["area"].values, np.array([2.0, 4.0]))


# ── average_psth_for_group ────────────────────────────────────────────────────


@pytest.fixture
def group_folder(tmp_path):
    folder = tmp_path / "saline_group"
    folder.mkdir()
    return folder


def test_average_psth_for_group_averages_the_members_means(tmp_path, group_folder):
    session1 = tmp_path / "session1"
    session2 = tmp_path / "session2"
    session1.mkdir()
    session2.mkdir()

    # Stub HDF5 files so the recording-site glob finds them
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

    input_parameters = {"selectForComputePsth": "z_score"}
    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session1), str(session2)],
        event="event_lever",
        group_folder=str(group_folder),
        inputParameters=input_parameters,
    )

    assert wrote_psth is True
    result = pd.read_hdf(group_folder / "event_lever_dms_z_score_dms.h5", key="df", mode="r")
    np.testing.assert_allclose(result["session1"].values, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result["session2"].values, np.array([3.0, 4.0, 5.0]))
    # Group mean is the across-session mean; err is the SEM over the 2 sessions:
    # mean = [2, 3, 4], std (population, ddof=0) = 1.0 → 1.0 / sqrt(2) ≈ 0.7071068
    np.testing.assert_allclose(result["mean"].values, np.array([2.0, 3.0, 4.0]), atol=1e-6)
    np.testing.assert_allclose(result["err"].values, np.full(3, 0.7071068), atol=1e-6)
    np.testing.assert_allclose(result["timestamps"].values, np.array([0.0, 1.0, 2.0]))


def test_average_psth_for_group_writes_into_the_given_group_folder(tmp_path, group_folder):
    session = tmp_path / "session1"
    session.mkdir()
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
        inputParameters={"selectForComputePsth": "z_score"},
    )

    assert (group_folder / "event_lever_dms_z_score_dms.h5").exists()
    # Nothing is written to an "average" directory beside the members any more.
    assert not (tmp_path / "average").exists()


def test_average_psth_for_group_reports_false_when_no_member_has_the_event(tmp_path, group_folder):
    session = tmp_path / "session1"
    session.mkdir()
    (session / "z_score_dms.hdf5").touch()

    wrote_psth = average_psth_for_group(
        member_run_folders=[str(session)],
        event="event_never_recorded",
        group_folder=str(group_folder),
        inputParameters={"selectForComputePsth": "z_score"},
    )

    assert wrote_psth is False
    assert not (group_folder / "event_never_recorded_dms_z_score_dms.h5").exists()


def test_average_psth_for_group_handles_non_overlapping_stores_without_indexerror(tmp_path, group_folder):
    """Sessions with entirely non-overlapping store_ids must not raise IndexError.

    Regression test for issue #274 — previously ``new_path`` was sized by the
    largest per-session file count, which could be smaller than the number of
    unique basenames and caused ``list index out of range`` at
    ``new_path[idx].append(path[i])``.
    """
    session1 = tmp_path / "session1"
    session2 = tmp_path / "session2"
    session1.mkdir()
    session2.mkdir()

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
        inputParameters={"selectForComputePsth": "z_score"},
    )

    assert (group_folder / "event_lever_regionA_z_score_regionA.h5").exists()
    assert (group_folder / "event_lever_regionB_z_score_regionB.h5").exists()
