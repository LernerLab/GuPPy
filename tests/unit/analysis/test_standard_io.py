import h5py
import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import write_hdf5
from guppy.analysis.standard_io import (
    read_binned_covariates_from_csv,
    read_binned_metrics_from_csv,
    read_control_and_signal,
    read_coords_pairwise,
    read_corrected_data,
    read_corrected_data_dict,
    read_corrected_timestamps_pairwise,
    read_corrected_ttl_timestamps,
    read_covariate_correlations_from_csv,
    read_data_for_combining_data,
    read_freq_and_amp_from_csv,
    read_peak_and_area_from_csv,
    read_psth_significance_from_csv,
    read_timestamps_for_combining_data,
    read_transients_from_hdf5,
    read_ttl,
    read_ttl_timestamps_for_combining_data,
    write_artifact_corrected_timestamps,
    write_artifact_removal,
    write_binned_covariates_to_csv,
    write_binned_metrics_to_csv,
    write_combined_data,
    write_corrected_data,
    write_corrected_timestamps,
    write_corrected_ttl_timestamps,
    write_covariate_correlations_to_csv,
    write_freq_and_amp_to_csv,
    write_group_table,
    write_peak_and_area_to_csv,
    write_psth_significance_to_csv,
    write_transients_to_hdf5,
    write_zscore,
)

# ── write_corrected_data ──────────────────────────────────────────────────────


def test_write_corrected_data_writes_each_channel_to_hdf5(tmp_path):
    store_label_to_data = {"control_dms": np.array([1.0, 2.0, 3.0]), "signal_dms": np.array([4.0, 5.0, 6.0])}
    write_corrected_data(str(tmp_path), store_label_to_data)

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

    write_zscore(str(tmp_path), "dms", z_score, dff, control_fit, synthetic_control=None)

    assert (tmp_path / "z_score_dms.hdf5").exists()
    assert (tmp_path / "dff_dms.hdf5").exists()
    assert (tmp_path / "cntrl_sig_fit_dms.hdf5").exists()
    result_z = h5py.File(tmp_path / "z_score_dms.hdf5", "r")["data"][:]
    np.testing.assert_array_equal(result_z, z_score)


def test_write_zscore_with_temp_control_writes_control_file(tmp_path):
    z_score = np.array([0.5])
    temp_control = np.array([1.0, np.nan, 2.0])
    write_zscore(str(tmp_path), "dms", z_score, z_score, z_score, synthetic_control=temp_control)

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
    store_label_to_data = {"control_dms": np.ones(3), "signal_dms": np.ones(3) * 2}
    compound_name_to_ttl = {"TTL1_dms": np.array([0.5, 1.5])}

    write_combined_data(str(tmp_path), pair_name_to_tsNew, store_label_to_data, compound_name_to_ttl)

    ts = h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r")["timestampNew"][:]
    np.testing.assert_array_equal(ts, np.array([0.0, 1.0, 2.0]))
    ttl = h5py.File(tmp_path / "TTL1_dms.hdf5", "r")["ts"][:]
    np.testing.assert_array_equal(ttl, np.array([0.5, 1.5]))


# ── write_peak_and_area_to_csv / read_peak_and_area_from_csv ─────────────────


@pytest.fixture
def peak_and_area_trials():
    return pd.DataFrame(
        {
            "session": ["s1", "s1", "s1"],
            "trial": ["1.5", "4.5", "mean"],
            "peak_pos_1": [2.0, 4.0, 3.0],
            "peak_neg_1": [-1.0, -3.0, -2.0],
            "area_1": [0.5, 1.5, 1.0],
        }
    )


@pytest.fixture
def peak_and_area_bins():
    return pd.DataFrame(
        {
            "session": ["s1"],
            "bin": ["bin_(0-2)"],
            "peak_pos_1": [3.0],
            "peak_neg_1": [-2.0],
            "area_1": [1.0],
        }
    )


def test_write_and_read_peak_and_area_without_bins_roundtrip(tmp_path, peak_and_area_trials):
    write_peak_and_area_to_csv(
        filepath=tmp_path, trials=peak_and_area_trials, bins=None, name="event_lever_z_score_dms"
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["peak_AUC_event_lever_z_score_dms.csv"]
    trials, bins = read_peak_and_area_from_csv(filepath=tmp_path, name="event_lever_z_score_dms")
    pd.testing.assert_frame_equal(trials, peak_and_area_trials)
    assert bins is None


def test_write_and_read_peak_and_area_with_bins_roundtrip(tmp_path, peak_and_area_trials, peak_and_area_bins):
    write_peak_and_area_to_csv(
        filepath=tmp_path, trials=peak_and_area_trials, bins=peak_and_area_bins, name="event_lever_z_score_dms"
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "binned_peak_AUC_event_lever_z_score_dms.csv",
        "peak_AUC_event_lever_z_score_dms.csv",
    ]
    trials, bins = read_peak_and_area_from_csv(filepath=tmp_path, name="event_lever_z_score_dms")
    pd.testing.assert_frame_equal(trials, peak_and_area_trials)
    pd.testing.assert_frame_equal(bins, peak_and_area_bins)


def test_write_peak_and_area_without_bins_removes_stale_binned_file(tmp_path, peak_and_area_trials, peak_and_area_bins):
    write_peak_and_area_to_csv(
        filepath=tmp_path, trials=peak_and_area_trials, bins=peak_and_area_bins, name="event_lever_z_score_dms"
    )
    write_peak_and_area_to_csv(
        filepath=tmp_path, trials=peak_and_area_trials, bins=None, name="event_lever_z_score_dms"
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["peak_AUC_event_lever_z_score_dms.csv"]


def test_read_peak_and_area_keeps_numeric_session_names_as_strings(tmp_path):
    trials_with_numeric_session = pd.DataFrame(
        {"session": ["1", "1"], "trial": ["10", "mean"], "peak_pos_1": [2.0, 2.0]}
    )
    write_peak_and_area_to_csv(filepath=tmp_path, trials=trials_with_numeric_session, bins=None, name="lever")

    trials, _ = read_peak_and_area_from_csv(filepath=tmp_path, name="lever")

    assert list(trials["session"]) == ["1", "1"]
    assert list(trials["trial"]) == ["10", "mean"]


# ── write_freq_and_amp_to_csv / read_freq_and_amp_from_csv ───────────────────


def test_write_and_read_freq_and_amp_roundtrip(tmp_path):
    arr = np.array([[2.5, 1.8]])
    write_freq_and_amp_to_csv(
        str(tmp_path),
        arr,
        "freqAndAmp_z_score_dms.csv",
        index=["session1"],
        columns=["freq (events/min)", "amplitude"],
    )
    df = read_freq_and_amp_from_csv(str(tmp_path), "z_score_dms")

    assert list(df.index) == ["session1"]
    np.testing.assert_allclose(df["freq (events/min)"].to_numpy(), [2.5])
    np.testing.assert_allclose(df["amplitude"].to_numpy(), [1.8])


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
    store_label_to_data = {"control_dms": np.ones(3)}
    compound_name_to_ttl = {"TTL1_dms": np.array([1.0])}
    pair_name_to_ts = {"dms": np.array([0.5, 1.5, 2.5])}

    write_artifact_removal(str(tmp_path), store_label_to_data, pair_name_to_ts, compound_name_to_ttl)

    assert (tmp_path / "control_dms.hdf5").exists()
    assert (tmp_path / "TTL1_dms.hdf5").exists()
    assert (tmp_path / "timeCorrection_dms.hdf5").exists()


# ── write_corrected_timestamps ────────────────────────────────────────────────


TIMELINE_ON_SIGNAL = {"control_dms": "signal_dms", "signal_dms": "signal_dms"}


def test_write_corrected_timestamps_writes_all_keys(tmp_path):
    corrected_ts = {"control_dms": np.array([2.0, 3.0, 4.0]), "signal_dms": np.array([2.0, 3.0, 4.0])}
    original_ts = {"control_dms": np.array([0.0, 1.0, 2.0, 3.0]), "signal_dms": np.array([0.0, 1.0, 2.0, 3.0])}
    sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    correction_index = {"control_dms": np.array([2, 3, 4]), "signal_dms": np.array([2, 3, 4])}

    write_corrected_timestamps(
        str(tmp_path), corrected_ts, original_ts, sampling_rate, correction_index, TIMELINE_ON_SIGNAL, "csv"
    )

    with h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r") as file:
        assert "timestampNew" in file
        assert "timeRecStart" in file
        assert "correctionIndex" in file
        assert "sampling_rate" in file
        np.testing.assert_array_equal(file["timestampNew"][:], np.array([2.0, 3.0, 4.0]))


def test_write_corrected_timestamps_csv_mode_records_the_acquisition_clock_start(tmp_path):
    # csv mode keeps the acquisition clock, so the recording start is the first raw timestamp.
    corrected_ts = {"control_dms": np.array([102.0, 103.0]), "signal_dms": np.array([102.0, 103.0])}
    original_ts = {
        "control_dms": np.array([100.0, 101.0, 102.0, 103.0]),
        "signal_dms": np.array([100.0, 101.0, 102.0, 103.0]),
    }
    sampling_rate = {"control_dms": np.array([1.0]), "signal_dms": np.array([1.0])}
    correction_index = {"control_dms": np.array([2, 3]), "signal_dms": np.array([2, 3])}

    write_corrected_timestamps(
        str(tmp_path), corrected_ts, original_ts, sampling_rate, correction_index, TIMELINE_ON_SIGNAL, "csv"
    )

    with h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r") as file:
        np.testing.assert_array_equal(file["recordingStart"][:], np.array([100.0]))
        np.testing.assert_array_equal(file["timeRecStart"][:], np.array([100.0]))


def test_write_corrected_timestamps_tdt_mode_records_a_zero_start(tmp_path):
    # tdt timestamps are rebased to the recording start during correction, so on the corrected
    # basis the recording begins at 0 even though timeRecStart holds the raw epoch value.
    corrected_ts = {"control_dms": np.array([1.0, 2.0]), "signal_dms": np.array([1.0, 2.0])}
    original_ts = {
        "control_dms": np.array([1595956345.0, 1595956346.0]),
        "signal_dms": np.array([1595956345.0, 1595956346.0]),
    }
    sampling_rate = {"control_dms": np.array([1.0]), "signal_dms": np.array([1.0])}
    correction_index = {"control_dms": np.array([0, 1]), "signal_dms": np.array([0, 1])}

    write_corrected_timestamps(
        str(tmp_path), corrected_ts, original_ts, sampling_rate, correction_index, TIMELINE_ON_SIGNAL, "tdt"
    )

    with h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r") as file:
        np.testing.assert_array_equal(file["recordingStart"][:], np.array([0.0]))
        np.testing.assert_array_equal(file["timeRecStart"][:], np.array([1595956345.0]))


@pytest.mark.parametrize(
    "timeline_label, expected_start, expected_sampling_rate",
    [("signal_dms", 100.5, 40.0), ("control_dms", 100.0, 20.0)],
)
def test_write_corrected_timestamps_describes_the_pairs_timeline_channel(
    tmp_path, timeline_label, expected_start, expected_sampling_rate
):
    # The two channels start and run at different rates; the one the pair is timed by
    # supplies its recording start and sampling rate, whichever channel is written last.
    corrected_ts = {"control_dms": np.array([101.0]), "signal_dms": np.array([101.0])}
    original_ts = {"control_dms": np.array([100.0, 101.0]), "signal_dms": np.array([100.5, 101.5])}
    sampling_rate = {"control_dms": np.array([20.0]), "signal_dms": np.array([40.0])}
    correction_index = {"control_dms": np.array([1]), "signal_dms": np.array([1])}
    timeline_labels = {"control_dms": timeline_label, "signal_dms": timeline_label}

    write_corrected_timestamps(
        str(tmp_path), corrected_ts, original_ts, sampling_rate, correction_index, timeline_labels, "csv"
    )

    with h5py.File(tmp_path / "timeCorrection_dms.hdf5", "r") as file:
        np.testing.assert_array_equal(file["recordingStart"][:], [expected_start])
        np.testing.assert_array_equal(file["timeRecStart"][:], [expected_start])
        np.testing.assert_array_equal(file["sampling_rate"][:], [expected_sampling_rate])


# ── read_control_and_signal ───────────────────────────────────────────────────


def test_read_control_and_signal_returns_data_and_timestamp_dicts(tmp_path):
    # Write ctrl0.hdf5 and sig0.hdf5 with data, timestamps, sampling_rate
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    for store_id in ["ctrl0", "sig0"]:
        write_hdf5(np.array([1.0, 2.0, 3.0]), store_id, str(tmp_path), "data")
        write_hdf5(np.array([0.0, 1.0, 2.0]), store_id, str(tmp_path), "timestamps")
        write_hdf5(np.array([100.0]), store_id, str(tmp_path), "sampling_rate")

    store_label_to_data, store_label_to_timestamps, store_label_to_sampling_rate, store_label_to_npoints = (
        read_control_and_signal(str(tmp_path), store_array)
    )

    assert "control_dms" in store_label_to_data
    assert "signal_dms" in store_label_to_data
    np.testing.assert_array_equal(store_label_to_data["control_dms"], np.array([1.0, 2.0, 3.0]))


# ── read_ttl ──────────────────────────────────────────────────────────────────


def test_read_ttl_returns_timestamps_for_non_channel_events(tmp_path):
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "ttl0", str(tmp_path), "timestamps")

    result = read_ttl(str(tmp_path), store_array)

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


def test_read_corrected_timestamps_pairwise_mismatched_pair_names_raises_actionable_error(tmp_path):
    # control suffix is 'dms', signal suffix is 'vms' → mismatch
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_vms.hdf5").touch()

    with pytest.raises(ValueError) as exc_info:
        read_corrected_timestamps_pairwise(str(tmp_path))

    msg = str(exc_info.value)
    assert "dms" in msg
    assert "vms" in msg
    assert str(tmp_path) in msg
    assert "re-run step 1" in msg


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


def test_read_coords_pairwise_mismatched_pair_names_raises_actionable_error(tmp_path):
    # control suffix is 'dms', signal suffix is 'vms' → mismatch
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_vms.hdf5").touch()
    pair_name_to_tsNew = {"dms": np.array([0.0, 1.0, 2.0, 5.0])}

    with pytest.raises(ValueError) as exc_info:
        read_coords_pairwise(str(tmp_path), pair_name_to_tsNew)

    msg = str(exc_info.value)
    assert "dms" in msg
    assert "vms" in msg
    assert str(tmp_path) in msg
    assert "re-run step 1" in msg


# ── read_corrected_data_dict ──────────────────────────────────────────────────


def test_read_corrected_data_dict_returns_control_and_signal(tmp_path):
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    write_hdf5(np.array([1.0, 2.0]), "control_dms", str(tmp_path), "data")
    write_hdf5(np.array([3.0, 4.0]), "signal_dms", str(tmp_path), "data")

    result = read_corrected_data_dict(str(tmp_path), store_array)

    assert "control_dms" in result
    assert "signal_dms" in result
    np.testing.assert_array_equal(result["control_dms"], np.array([1.0, 2.0]))


# ── read_corrected_ttl_timestamps ─────────────────────────────────────────────


def test_read_corrected_ttl_timestamps_returns_compound_name_dict(tmp_path):
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "TTL1_dms", str(tmp_path), "ts")

    result = read_corrected_ttl_timestamps(str(tmp_path), store_array)

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
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    write_hdf5(np.array([1.0, 2.0, 3.0]), "control_dms", str(session), "data")
    write_hdf5(np.array([4.0, 5.0, 6.0]), "signal_dms", str(session), "data")

    result = read_data_for_combining_data([str(session)], store_array)

    assert "control_dms" in result
    assert str(session) in result["control_dms"]
    np.testing.assert_array_equal(result["control_dms"][str(session)], np.array([1.0, 2.0, 3.0]))


# ── read_ttl_timestamps_for_combining_data ────────────────────────────────────


def test_read_ttl_timestamps_for_combining_data_returns_nested_dict(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_dms.hdf5").touch()
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    write_hdf5(np.array([1.5, 2.5]), "TTL1_dms", str(session), "ts")

    result = read_ttl_timestamps_for_combining_data([str(session)], store_array)

    assert "TTL1_dms" in result
    assert str(session) in result["TTL1_dms"]
    np.testing.assert_array_equal(result["TTL1_dms"][str(session)], np.array([1.5, 2.5]))


# ── Combining-data suffix mismatch error paths ────────────────────────────────


def test_read_timestamps_for_combining_data_mismatched_recording_sites_raises(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_vms.hdf5").touch()
    with pytest.raises(ValueError) as exception_info:
        read_timestamps_for_combining_data([str(session)])
    message = str(exception_info.value)
    assert "Mismatched control/signal files" in message
    assert "dms" in message
    assert "vms" in message


def test_read_data_for_combining_data_mismatched_recording_sites_raises(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_vms.hdf5").touch()
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_vms"]])
    with pytest.raises(ValueError, match="Mismatched control/signal files"):
        read_data_for_combining_data([str(session)], store_array)


def test_read_ttl_timestamps_for_combining_data_mismatched_recording_sites_raises(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "control_dms.hdf5").touch()
    (session / "signal_vms.hdf5").touch()
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_vms", "TTL1"]])
    with pytest.raises(ValueError, match="Mismatched control/signal files"):
        read_ttl_timestamps_for_combining_data([str(session)], store_array)


# ── write_binned_metrics_to_csv / read_binned_metrics_from_csv ───────────────


@pytest.fixture
def binned_metrics():
    return pd.DataFrame(
        {
            "bin_start": [0.0, 5.0],
            "bin_end": [5.0, 10.0],
            "n_samples": [5, 6],
            "mean_zscore": [2.0, 7.5],
            "mean_dff": [0.2, 0.75],
            "transient_count_z_score": [1, 3],
        },
        index=pd.RangeIndex(2, name="bin"),
    )


def test_write_and_read_binned_metrics_roundtrip(tmp_path, binned_metrics):
    write_binned_metrics_to_csv(str(tmp_path), binned_metrics, "dms")
    result = read_binned_metrics_from_csv(str(tmp_path), "dms")

    pd.testing.assert_frame_equal(result, binned_metrics, check_index_type=False)


def test_write_binned_metrics_to_csv_preserves_the_bin_index(tmp_path, binned_metrics):
    write_binned_metrics_to_csv(str(tmp_path), binned_metrics, "dms")
    result = pd.read_csv(tmp_path / "binned_metrics_dms.csv", index_col="bin")

    np.testing.assert_allclose(result["mean_zscore"].to_numpy(), [2.0, 7.5])
    np.testing.assert_allclose(result["bin_end"].to_numpy(), [5.0, 10.0])
    np.testing.assert_array_equal(result["transient_count_z_score"].to_numpy(), [1, 3])


# ── PSTH significance, binned covariates, covariate correlations ─────────────


def test_write_and_read_psth_significance_roundtrip(tmp_path):
    significance = pd.DataFrame(
        {
            "timestamps": [0.0, 0.5],
            "estimate": [1.0, -0.5],
            "ci_lower": [0.2, -1.0],
            "ci_upper": [1.8, 0.1],
            "significant": [1, 0],
            "alpha": [0.05, 0.05],
            "n": [4, 4],
        }
    )
    write_psth_significance_to_csv(filepath=tmp_path, significance=significance, name="lever_vs_press")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["significance_lever_vs_press.csv"]
    result = read_psth_significance_from_csv(filepath=tmp_path, name="lever_vs_press")
    pd.testing.assert_frame_equal(result, significance)


def test_write_and_read_binned_covariates_roundtrip(tmp_path):
    binned_covariates = pd.DataFrame(
        {"bin_start": [0.0, 5.0], "bin_end": [5.0, 10.0], "speed": [1.5, 2.5], "n_samples_speed": [3, 4]},
        index=pd.RangeIndex(2, name="bin"),
    )
    write_binned_covariates_to_csv(filepath=tmp_path, binned_covariates=binned_covariates, recording_site="dms")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["binned_covariates_dms.csv"]
    result = read_binned_covariates_from_csv(filepath=tmp_path, recording_site="dms")
    pd.testing.assert_frame_equal(result, binned_covariates, check_index_type=False)


def test_write_and_read_covariate_correlations_roundtrip(tmp_path):
    correlations = pd.DataFrame(
        {"metric": ["mean_zscore"], "covariate": ["speed"], "pearson_r": [0.5], "spearman_rho": [0.4], "n_bins": [6]},
        index=pd.RangeIndex(1, name="pair"),
    )
    write_covariate_correlations_to_csv(filepath=tmp_path, correlations=correlations, recording_site="dms")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["covariate_correlations_dms.csv"]
    result = read_covariate_correlations_from_csv(filepath=tmp_path, recording_site="dms")
    pd.testing.assert_frame_equal(result, correlations, check_index_type=False)


# ── write_group_table ─────────────────────────────────────────────────────────


def test_write_group_table_writes_only_a_csv(tmp_path):
    table = pd.DataFrame({"mean_zscore": [1.0, 3.0]}, index=pd.Index(["run_a", "run_b"], name="run"))

    write_group_table(filepath=tmp_path, table=table, name="tonic_means_dms")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["tonic_means_dms.csv"]
    result = pd.read_csv(tmp_path / "tonic_means_dms.csv", index_col="run")
    pd.testing.assert_frame_equal(result, table)
