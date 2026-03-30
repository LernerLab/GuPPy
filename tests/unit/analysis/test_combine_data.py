import numpy as np

from guppy.analysis.combine_data import combine_data, eliminateData, eliminateTs

# ── eliminateData ─────────────────────────────────────────────────────────────


def test_eliminate_data_two_filepaths_concatenates_and_starts_at_lights_turn_on():
    # filepath1: ts=[0,1,2], data=[10,20,30], timeForLightsTurnOn=0.5
    # sub = ts[0] - 0.5 = -0.5; new_ts = [0.5, 1.5, 2.5]
    # filepath2: ts=[5,6,7], data=[40,50,60]
    # new_ts = [5,6,7] - (5 - 2.5) = [2.5, 3.5, 4.5]; stored as [2.5+0.01, 3.5+0.01, 4.5+0.01]
    filepath1, filepath2 = "path1", "path2"
    filepath_to_timestamps = {filepath1: np.array([0.0, 1.0, 2.0]), filepath2: np.array([5.0, 6.0, 7.0])}
    filepath_to_data = {filepath1: np.array([10.0, 20.0, 30.0]), filepath2: np.array([40.0, 50.0, 60.0])}

    arr, ts_arr = eliminateData(filepath_to_timestamps, filepath_to_data, 0.5, 100.0)

    assert arr.shape[0] == 6
    np.testing.assert_array_equal(arr, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    np.testing.assert_allclose(ts_arr[0], 0.5, atol=1e-6)


def test_eliminate_data_second_segment_follows_first_with_one_sample_gap():
    # Second segment starts right after the last sample of the first + 1/sampling_rate
    filepath_to_timestamps = {"p1": np.array([0.0, 1.0, 2.0]), "p2": np.array([10.0, 11.0, 12.0])}
    filepath_to_data = {"p1": np.ones(3), "p2": np.ones(3)}

    _, ts_arr = eliminateData(filepath_to_timestamps, filepath_to_data, 0.0, 10.0)

    # First segment ends at ts[2] - sub = 2.0; second starts at 2.0 + 1/10 = 2.1
    np.testing.assert_allclose(ts_arr[3], ts_arr[2] + 1 / 10.0, atol=1e-6)


# ── eliminateTs ───────────────────────────────────────────────────────────────


def test_eliminate_ts_two_filepaths_concatenates_ttl_timestamps():
    # filepath1: tsNew=[0,1,2], ttl=[0.5,1.5], timeForLightsTurnOn=0.5
    # sub = 0 - 0.5 = -0.5; tsNew_arr=[0.5,1.5,2.5]; ts_arr=[1.0, 2.0]
    # filepath2: tsNew=[5,6,7], ttl=[5.5,6.5]
    # new_ts = [5.5,6.5] - (5-2.5) = [3.0,4.0]; ts_arr += [3.0+0.01, 4.0+0.01]
    filepath_to_timestamps = {"p1": np.array([0.0, 1.0, 2.0]), "p2": np.array([5.0, 6.0, 7.0])}
    filepath_to_ttl = {"p1": np.array([0.5, 1.5]), "p2": np.array([5.5, 6.5])}

    ts_arr = eliminateTs(filepath_to_timestamps, filepath_to_ttl, 0.5, 100.0)

    assert ts_arr.shape[0] == 4
    np.testing.assert_allclose(ts_arr[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(ts_arr[1], 2.0, atol=1e-6)
    np.testing.assert_allclose(ts_arr[2], 3.01, atol=1e-6)
    np.testing.assert_allclose(ts_arr[3], 4.01, atol=1e-6)


# ── combine_data ──────────────────────────────────────────────────────────────


def test_combine_data_single_filepath_produces_correct_output_keys(tmp_path):
    # Create control and signal HDF5 stub files so decide_naming_convention finds them
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "control_dms.hdf5").touch()
    (session_dir / "signal_dms.hdf5").touch()

    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    filepath = str(session_dir)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    pair_name_to_filepath_to_timestamps = {"dms": {filepath: timestamps.copy()}}
    display_name_to_filepath_to_data = {
        "control_dms": {filepath: np.ones(4)},
        "signal_dms": {filepath: np.ones(4) * 2.0},
    }
    compound_name_to_filepath_to_ttl_timestamps = {"TTL1_dms": {filepath: np.array([1.5, 2.5])}}

    pair_name_to_tsNew, display_name_to_data, compound_name_to_ttl = combine_data(
        filepaths_to_combine=[filepath],
        pair_name_to_filepath_to_timestamps=pair_name_to_filepath_to_timestamps,
        display_name_to_filepath_to_data=display_name_to_filepath_to_data,
        compound_name_to_filepath_to_ttl_timestamps=compound_name_to_filepath_to_ttl_timestamps,
        timeForLightsTurnOn=0.0,
        storesList=storesList,
        sampling_rate=100.0,
    )

    assert "dms" in pair_name_to_tsNew
    assert "control_dms" in display_name_to_data
    assert "signal_dms" in display_name_to_data
    assert "TTL1_dms" in compound_name_to_ttl
    assert display_name_to_data["control_dms"].shape[0] == 4
