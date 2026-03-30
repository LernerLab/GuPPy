import h5py
import numpy as np
import pandas as pd

from guppy.analysis.io_utils import write_hdf5
from guppy.analysis.standard_io import (
    read_control_and_signal,
    read_coords_pairwise,
    read_corrected_data,
    read_corrected_data_dict,
    read_corrected_timestamps_pairwise,
    read_corrected_ttl_timestamps,
    read_data_for_combining_data,
    read_freq_and_amp_from_hdf5,
    read_timestamps_for_combining_data,
    read_transients_from_hdf5,
    read_ttl,
    read_ttl_timestamps_for_combining_data,
    write_artifact_corrected_timestamps,
    write_artifact_removal,
    write_combined_data,
    write_corrected_data,
    write_corrected_timestamps,
    write_corrected_ttl_timestamps,
    write_freq_and_amp_to_csv,
    write_freq_and_amp_to_hdf5,
    write_peak_and_area_to_csv,
    write_peak_and_area_to_hdf5,
    write_transients_to_hdf5,
    write_zscore,
)

# ── write_corrected_data ──────────────────────────────────────────────────────


def test_write_corrected_data_writes_each_channel_to_hdf5(tmp_path):
    name_to_data = {"control_dms": np.array([1.0, 2.0, 3.0]), "signal_dms": np.array([4.0, 5.0, 6.0])}
    write_corrected_data(str(tmp_path), name_to_data)

    result = write_hdf5.__module__  # just ensure import worked
    control = h5py.File(tmp_path / "control_dms.hdf5", "r")["data"][:]
    signal = h5py.File(tmp_path / "signal_dms.hdf5", "r")["data"][:]
    np.testing.assert_array_equal(control, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(signal, np.array([4.0, 5.0, 6.0]))


# ── write_corrected_ttl_timestamps ────────────────────────────────────────────


def test_write_corrected_ttl_timestamps_writes_ts_key(tmp_path):
    compound_name_to_ttl = {"TTL1_dms": np.array([1.5, 2.5, 3.5])}
    write_corrected_ttl_timestamps(str(tmp_path), compound_name_to_ttl)

    result = h5py.File(tmp_path / "TTL1_dms.hdf5", "r")["ts"][:]
    np.testing.assert_array_equal(result, np.array([1.5, 2.5, 3.5]))


# ── write_zscore ──────────────────────────────────────────────────────────────


def test_write_zscore_creates_four_hdf5_files(tmp_path):
    z_score = np.array([0.5, 1.0, -0.5])
    dff = np.array([0.1, 0.2, 0.3])
    control_fit = np.array([2.0, 2.1, 2.2])

    write_zscore(str(tmp_path), "dms", z_score, dff, control_fit, temp_control_arr=None)

    assert (tmp_path / "z_score_dms.hdf5").exists()
    assert (tmp_path / "dff_dms.hdf5").exists()
    assert (tmp_path / "cntrl_sig_fit_dms.hdf5").exists()
    result_z = h5py.File(tmp_path / "z_score_dms.hdf5", "r")["data"][:]
    np.testing.assert_array_equal(result_z, z_score)


def test_write_zscore_with_temp_control_writes_control_file(tmp_path):
    z_score = np.array([0.5])
    temp_control = np.array([1.0, np.nan, 2.0])
    write_zscore(str(tmp_path), "dms", z_score, z_score, z_score, temp_control_arr=temp_control)

    result = h5py.File(tmp_path / "control_dms.hdf5", "r")["data"][:]
    np.testing.assert_array_equal(result[0], 1.0)
    assert np.isnan(result[1])


# ── write_artifact_corrected_timestamps ──────────────────────────────────────


def test_write_artifact_corrected_timestamps_writes_timestampnew(tmp_path):
    pair_name_to_ts = {"dms": np.array([0.5, 1.5, 2.5])}
    write_artifact_corrected_timestamps(str(tmp_path), pair_name_to_ts)

    result = h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r")["timestampNew"][:]
    np.testing.assert_array_equal(result, np.array([0.5, 1.5, 2.5]))


# ── write_combined_data ───────────────────────────────────────────────────────


def test_write_combined_data_writes_all_three_dicts(tmp_path):
    pair_name_to_tsNew = {"dms": np.array([0.0, 1.0, 2.0])}
    display_name_to_data = {"control_dms": np.ones(3), "signal_dms": np.ones(3) * 2}
    compound_name_to_ttl = {"TTL1_dms": np.array([0.5, 1.5])}

    write_combined_data(str(tmp_path), pair_name_to_tsNew, display_name_to_data, compound_name_to_ttl)

    ts = h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r")["timestampNew"][:]
    np.testing.assert_array_equal(ts, np.array([0.0, 1.0, 2.0]))
    ttl = h5py.File(tmp_path / "TTL1_dms.hdf5", "r")["ts"][:]
    np.testing.assert_array_equal(ttl, np.array([0.5, 1.5]))


# ── write_peak_and_area_to_hdf5 / csv ────────────────────────────────────────


def test_write_peak_and_area_to_hdf5_creates_file(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_peak_and_area_to_hdf5(str(tmp_path), arr, "event_lever_z_score_dms", index=["s1", "s2"])

    assert (tmp_path / "peak_AUC_event_lever_z_score_dms.h5").exists()
    df = pd.read_hdf(tmp_path / "peak_AUC_event_lever_z_score_dms.h5", key="df")
    np.testing.assert_array_equal(df.values, arr)


def test_write_peak_and_area_to_csv_creates_file(tmp_path):
    arr = np.array([[5.0, 6.0]])
    write_peak_and_area_to_csv(str(tmp_path), arr, "event_lever_z_score_dms", index=["s1"])

    assert (tmp_path / "peak_AUC_event_lever_z_score_dms.csv").exists()


# ── write_freq_and_amp_to_hdf5 / csv and read_freq_and_amp_from_hdf5 ─────────


def test_write_and_read_freq_and_amp_roundtrip(tmp_path):
    arr = np.array([[2.5, 1.8]])
    write_freq_and_amp_to_hdf5(
        str(tmp_path), arr, "z_score_dms", index=["session1"], columns=["freq (events/min)", "amplitude"]
    )
    df = read_freq_and_amp_from_hdf5(str(tmp_path), "z_score_dms")

    np.testing.assert_allclose(df["freq (events/min)"].iloc[0], 2.5)
    np.testing.assert_allclose(df["amplitude"].iloc[0], 1.8)


def test_write_freq_and_amp_to_csv_creates_file(tmp_path):
    arr = np.array([[3.0, 2.0]])
    write_freq_and_amp_to_csv(
        str(tmp_path), arr, "freqAndAmp_z_score_dms.csv", index=["s1"], columns=["freq (events/min)", "amplitude"]
    )

    assert (tmp_path / "freqAndAmp_z_score_dms.csv").exists()


# ── write_transients_to_hdf5 / read_transients_from_hdf5 ─────────────────────


def test_write_and_read_transients_roundtrip(tmp_path):
    z_score = np.array([0.1, 5.0, 0.2, 0.3])
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    peaks_ind = np.array([1])

    write_transients_to_hdf5(str(tmp_path), "dms", z_score, ts, peaks_ind)
    result_z, result_ts, result_peaks = read_transients_from_hdf5(str(tmp_path), "dms")

    np.testing.assert_array_equal(result_z, z_score)
    np.testing.assert_array_equal(result_ts, ts)
    np.testing.assert_array_equal(result_peaks, peaks_ind)


# ── write_artifact_removal ────────────────────────────────────────────────────


def test_write_artifact_removal_creates_data_ttl_and_timestamp_files(tmp_path):
    name_to_data = {"control_dms": np.ones(3)}
    compound_name_to_ttl = {"TTL1_dms": np.array([1.0])}
    pair_name_to_ts = {"dms": np.array([0.5, 1.5, 2.5])}

    write_artifact_removal(str(tmp_path), name_to_data, pair_name_to_ts, compound_name_to_ttl)

    assert (tmp_path / "control_dms.hdf5").exists()
    assert (tmp_path / "TTL1_dms.hdf5").exists()
    assert (tmp_path / "timeCorrection_dms.hdf5").exists()


# ── write_corrected_timestamps ────────────────────────────────────────────────


def test_write_corrected_timestamps_writes_all_keys(tmp_path):
    corrected_ts = {"control_dms": np.array([2.0, 3.0, 4.0]), "signal_dms": np.array([2.0, 3.0, 4.0])}
    original_ts = {"control_dms": np.array([0.0, 1.0, 2.0, 3.0]), "signal_dms": np.array([0.0, 1.0, 2.0, 3.0])}
    sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    correction_index = {"control_dms": np.array([2, 3, 4]), "signal_dms": np.array([2, 3, 4])}

    write_corrected_timestamps(str(tmp_path), corrected_ts, original_ts, sampling_rate, correction_index)

    with h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r") as file:
        assert "timestampNew" in file
        assert "timeRecStart" in file
        assert "correctionIndex" in file
        assert "sampling_rate" in file
        np.testing.assert_array_equal(file["timestampNew"][:], np.array([2.0, 3.0, 4.0]))


# ── read_control_and_signal ───────────────────────────────────────────────────


def test_read_control_and_signal_returns_data_and_timestamp_dicts(tmp_path):
    # Write ctrl0.hdf5 and sig0.hdf5 with data, timestamps, sampling_rate
    storesList = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    for storename in ["ctrl0", "sig0"]:
        write_hdf5(np.array([1.0, 2.0, 3.0]), storename, str(tmp_path), "data")
        write_hdf5(np.array([0.0, 1.0, 2.0]), storename, str(tmp_path), "timestamps")
        write_hdf5(np.array([100.0]), storename, str(tmp_path), "sampling_rate")

    name_to_data, name_to_timestamps, name_to_sampling_rate, name_to_npoints = read_control_and_signal(
        str(tmp_path), storesList
    )

    assert "control_dms" in name_to_data
    assert "signal_dms" in name_to_data
    np.testing.assert_array_equal(name_to_data["control_dms"], np.array([1.0, 2.0, 3.0]))


# ── read_ttl ──────────────────────────────────────────────────────────────────


def test_read_ttl_returns_timestamps_for_non_channel_events(tmp_path):
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "ttl0", str(tmp_path), "timestamps")

    result = read_ttl(str(tmp_path), storesList)

    assert "TTL1" in result
    np.testing.assert_array_equal(result["TTL1"], np.array([1.5, 2.5]))


# ── read_corrected_data ───────────────────────────────────────────────────────


def test_read_corrected_data_returns_control_signal_and_timestamps(tmp_path):
    control_path = tmp_path / "control.h5"
    signal_path = tmp_path / "signal.h5"

    with h5py.File(control_path, "w") as f:
        f.create_dataset("data", data=np.array([1.0, 2.0]))
    with h5py.File(signal_path, "w") as f:
        f.create_dataset("data", data=np.array([3.0, 4.0]))
    write_hdf5(np.array([0.0, 1.0]), "timeCorrection_dms", str(tmp_path), "timestampNew")

    control, signal, tsNew = read_corrected_data(str(control_path), str(signal_path), str(tmp_path), "dms")

    np.testing.assert_array_equal(control, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(signal, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(tsNew, np.array([0.0, 1.0]))


# ── read_corrected_timestamps_pairwise ────────────────────────────────────────


def test_read_corrected_timestamps_pairwise_returns_pair_name_dicts(tmp_path):
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_dms.hdf5").touch()
    write_hdf5(np.array([1.0, 2.0, 3.0]), "timeCorrection_dms", str(tmp_path), "timestampNew")
    write_hdf5(np.array([100.0]), "timeCorrection_dms", str(tmp_path), "sampling_rate")

    pair_name_to_tsNew, pair_name_to_sampling_rate = read_corrected_timestamps_pairwise(str(tmp_path))

    assert "dms" in pair_name_to_tsNew
    np.testing.assert_array_equal(pair_name_to_tsNew["dms"], np.array([1.0, 2.0, 3.0]))
    assert pair_name_to_sampling_rate["dms"] == 100.0


# ── read_coords_pairwise ──────────────────────────────────────────────────────


def test_read_coords_pairwise_returns_default_range_when_no_coords_file(tmp_path):
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_dms.hdf5").touch()
    tsNew = np.array([0.0, 1.0, 2.0, 5.0])
    pair_name_to_tsNew = {"dms": tsNew}

    result = read_coords_pairwise(str(tmp_path), pair_name_to_tsNew)

    assert "dms" in result
    # No coords file → default = [[0, tsNew[-1]]] = [[0, 5.0]]
    np.testing.assert_array_equal(result["dms"], np.array([[0, 5.0]]))


# ── read_corrected_data_dict ──────────────────────────────────────────────────


def test_read_corrected_data_dict_returns_control_and_signal(tmp_path):
    storesList = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    write_hdf5(np.array([1.0, 2.0]), "control_dms", str(tmp_path), "data")
    write_hdf5(np.array([3.0, 4.0]), "signal_dms", str(tmp_path), "data")

    result = read_corrected_data_dict(str(tmp_path), storesList)

    assert "control_dms" in result
    assert "signal_dms" in result
    np.testing.assert_array_equal(result["control_dms"], np.array([1.0, 2.0]))


# ── read_corrected_ttl_timestamps ─────────────────────────────────────────────


def test_read_corrected_ttl_timestamps_returns_compound_name_dict(tmp_path):
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "TTL1_dms", str(tmp_path), "ts")

    result = read_corrected_ttl_timestamps(str(tmp_path), storesList)

    assert "TTL1_dms" in result
    np.testing.assert_array_equal(result["TTL1_dms"], np.array([1.5, 2.5]))


# ── read_timestamps_for_combining_data ────────────────────────────────────────


def test_read_timestamps_for_combining_data_returns_nested_dict(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_dms.hdf5").touch()
    write_hdf5(np.array([0.0, 1.0, 2.0]), "timeCorrection_dms", str(session), "timestampNew")

    result = read_timestamps_for_combining_data([str(session)])

    assert "dms" in result
    assert str(session) in result["dms"]
    np.testing.assert_array_equal(result["dms"][str(session)], np.array([0.0, 1.0, 2.0]))


# ── read_data_for_combining_data ──────────────────────────────────────────────


def test_read_data_for_combining_data_returns_nested_dict(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_dms.hdf5").touch()
    storesList = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    write_hdf5(np.array([1.0, 2.0, 3.0]), "control_dms", str(session), "data")
    write_hdf5(np.array([4.0, 5.0, 6.0]), "signal_dms", str(session), "data")

    result = read_data_for_combining_data([str(session)], storesList)

    assert "control_dms" in result
    assert str(session) in result["control_dms"]
    np.testing.assert_array_equal(result["control_dms"][str(session)], np.array([1.0, 2.0, 3.0]))


# ── read_ttl_timestamps_for_combining_data ────────────────────────────────────


def test_read_ttl_timestamps_for_combining_data_returns_nested_dict(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_dms.hdf5").touch()
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "TTL1_dms", str(session), "ts")

    result = read_ttl_timestamps_for_combining_data([str(session)], storesList)

    assert "TTL1_dms" in result
    assert str(session) in result["TTL1_dms"]
    np.testing.assert_array_equal(result["TTL1_dms"][str(session)], np.array([1.5, 2.5]))
