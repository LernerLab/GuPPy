import numpy as np
import pandas as pd

from guppy.analysis.psth_average import (
    averageForGroup,
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


# ── averageForGroup ───────────────────────────────────────────────────────────


def test_average_for_group_creates_averaged_psth_file(tmp_path):
    # Two session folders, each with z_score_dms.hdf5 and a PSTH DataFrame
    # Expected output: tmp_path/average/event_lever_dms_z_score_dms.h5
    session1 = tmp_path / "session1"
    session2 = tmp_path / "session2"
    session1.mkdir()
    session2.mkdir()

    # Stub HDF5 files so glob finds them
    (session1 / "z_score_dms.hdf5").touch()
    (session2 / "z_score_dms.hdf5").touch()

    # PSTH arrays: 1 trial row + timestamps row, 3 timepoints
    psth = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])  # [trial, timestamps]
    columns = ["trial1", "timestamps"]
    create_Df_for_psth(str(session1), "event_lever_dms", "z_score_dms", psth, columns=columns)
    create_Df_for_psth(str(session2), "event_lever_dms", "z_score_dms", psth, columns=columns)

    input_parameters = {"abspath": str(tmp_path), "selectForComputePsth": "z_score"}
    averageForGroup([str(session1), str(session2)], "event_lever", input_parameters)

    assert (tmp_path / "average" / "event_lever_dms_z_score_dms.h5").exists()
