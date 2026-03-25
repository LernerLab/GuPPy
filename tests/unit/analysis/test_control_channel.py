import h5py
import numpy as np
import pandas as pd

from guppy.analysis.control_channel import (
    add_control_channel,
    create_control_channel,
    curveFitFn,
    helper_create_control_channel,
)


def test_curve_fit_fn_known_params_at_x_zero():
    # a + b * exp(-(1/c) * 0) = a + b
    result = curveFitFn(np.array([0.0]), a=0.0, b=1.0, c=1.0)
    np.testing.assert_allclose(result, np.array([1.0]))


def test_curve_fit_fn_large_x_approaches_a():
    # At large x, exp term vanishes and result → a
    result = curveFitFn(np.array([1000.0]), a=5.0, b=10.0, c=1.0)
    np.testing.assert_allclose(result, np.array([5.0]), atol=1e-3)


def test_curve_fit_fn_vectorized_produces_same_shape():
    x = np.linspace(0, 10, 100)
    result = curveFitFn(x, a=1.0, b=2.0, c=3.0)
    assert result.shape == x.shape


def test_helper_create_control_channel_output_shape_matches_signal():
    rng = np.random.default_rng(seed=20)
    timestamps = np.linspace(0, 10, 500)
    # Well-conditioned exponential decay that curve_fit can reliably converge on
    signal = 3.0 + 10.0 * np.exp(-timestamps / 5.0) + 0.01 * rng.standard_normal(500)
    result = helper_create_control_channel(signal, timestamps, window=101)
    assert result.shape == signal.shape


def test_helper_create_control_channel_output_is_smooth():
    rng = np.random.default_rng(seed=21)
    timestamps = np.linspace(0, 10, 500)
    signal = 2.0 + 8.0 * np.exp(-timestamps / 4.0) + 0.01 * rng.standard_normal(500)
    result = helper_create_control_channel(signal, timestamps, window=101)
    # The fitted curve should have lower variance than the noisy signal
    assert result.var() < signal.var()


def test_helper_create_control_channel_large_window_adjusted_down():
    rng = np.random.default_rng(seed=22)
    timestamps = np.linspace(0, 10, 200)
    signal = 1.0 + 5.0 * np.exp(-timestamps / 3.0) + 0.01 * rng.standard_normal(200)
    result = helper_create_control_channel(signal, timestamps, window=9999)
    assert result.shape == signal.shape


# ── add_control_channel ───────────────────────────────────────────────────────


def test_add_control_channel_adds_entry_for_unmatched_signal(tmp_path):
    with h5py.File(tmp_path / "sig0.hdf5", "w") as file:
        file.create_dataset("data", data=np.zeros(10))
    arr = np.array([["sig0"], ["signal_dms"]])
    result = add_control_channel(str(tmp_path), arr)
    np.testing.assert_array_equal(result[0, :], ["sig0", "cntrl0"])
    np.testing.assert_array_equal(result[1, :], ["signal_dms", "control_dms"])
    assert (tmp_path / "cntrl0.hdf5").exists()


def test_add_control_channel_no_new_entry_when_control_exists(tmp_path):
    arr = np.array([["sig0", "ctrl0"], ["signal_dms", "control_dms"]])
    result = add_control_channel(str(tmp_path), arr)
    np.testing.assert_array_equal(result, arr)


# ── create_control_channel ────────────────────────────────────────────────────


def test_create_control_channel_writes_control_hdf5_and_csv(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    timestamps = np.linspace(0, 10, 200)
    # Pure exponential so curve_fit converges to known params (a=3, b=10, c=5)
    signal = 3.0 + 10.0 * np.exp(-timestamps / 5.0)
    sampling_rate = np.full(timestamps.shape, np.nan)
    sampling_rate[0] = 100.0
    with h5py.File(session_dir / "signal_dms.hdf5", "w") as file:
        file.create_dataset("data", data=signal)
    with h5py.File(session_dir / "timeCorrection_dms.hdf5", "w") as file:
        file.create_dataset("timestampNew", data=timestamps)
        file.create_dataset("sampling_rate", data=sampling_rate)
    arr = np.array([["cntrl0", "sig0"], ["control_dms", "signal_dms"]])
    create_control_channel(str(session_dir), arr, window=101)
    assert (session_dir / "control_dms.hdf5").exists()
    with h5py.File(session_dir / "control_dms.hdf5", "r") as file:
        control_data = file["data"][:]
    dataframe = pd.read_csv(tmp_path / "cntrl0.csv")
    np.testing.assert_allclose(dataframe["timestamps"].values, timestamps)
    # CSV and HDF5 should contain identical control data
    np.testing.assert_allclose(dataframe["data"].values, control_data)
