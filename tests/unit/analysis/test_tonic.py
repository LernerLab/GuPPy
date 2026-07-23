import numpy as np
import pandas as pd
import pytest

from guppy.analysis.standard_io import read_tonic_epochs, write_tonic_to_hdf5
from guppy.analysis.tonic import compute_tonic_means


class TestComputeTonicMeans:
    @pytest.fixture
    def traces(self):
        # timestamps 0,1,2,...,10; z_score = timestamps; dff = timestamps / 10.
        timestamps = np.arange(0.0, 11.0, 1.0)
        z_score = timestamps.copy()
        dff = timestamps / 10.0
        return z_score, dff, timestamps

    def test_means_over_windows(self, traces):
        z_score, dff, timestamps = traces
        # baseline covers t in [0, 2] -> samples 0,1,2 -> z-mean 1.0, dff-mean 0.1
        # post covers t in [8, 10] -> samples 8,9,10 -> z-mean 9.0, dff-mean 0.9
        epochs = pd.DataFrame({"label": ["baseline", "post"], "start": [0.0, 8.0], "end": [2.0, 10.0]})
        result = compute_tonic_means(z_score, dff, timestamps, epochs)

        assert list(result.index) == ["baseline", "post"]
        assert result.index.name == "epoch"
        np.testing.assert_allclose(result["mean_zscore"].to_numpy(), [1.0, 9.0])
        np.testing.assert_allclose(result["mean_dff"].to_numpy(), [0.1, 0.9])

    def test_nan_samples_are_ignored(self, traces):
        z_score, dff, timestamps = traces
        z_score = z_score.copy()
        z_score[1] = np.nan  # sample at t=1 dropped from the [0,2] mean -> (0+2)/2 = 1.0
        epochs = pd.DataFrame({"label": ["baseline"], "start": [0.0], "end": [2.0]})
        result = compute_tonic_means(z_score, dff, timestamps, epochs)
        np.testing.assert_allclose(result["mean_zscore"].to_numpy(), [1.0])

    def test_start_not_less_than_end_raises(self, traces):
        z_score, dff, timestamps = traces
        epochs = pd.DataFrame({"label": ["bad"], "start": [5.0], "end": [5.0]})
        with pytest.raises(ValueError, match="strictly less"):
            compute_tonic_means(z_score, dff, timestamps, epochs)

    def test_window_running_past_the_end_is_clamped(self, traces):
        # timestamps end at 10; a window [8, 20] clamps to samples 8,9,10 -> z-mean 9.0.
        # This is the "typed the nominal duration" case and must NOT raise.
        z_score, dff, timestamps = traces
        epochs = pd.DataFrame({"label": ["wash-in"], "start": [8.0], "end": [20.0]})
        result = compute_tonic_means(z_score, dff, timestamps, epochs)
        np.testing.assert_allclose(result["mean_zscore"].to_numpy(), [9.0])

    def test_window_starting_before_the_recording_is_clamped(self, traces):
        # timestamps start at 0; a window [-5, 2] clamps to samples 0,1,2 -> z-mean 1.0.
        z_score, dff, timestamps = traces
        epochs = pd.DataFrame({"label": ["baseline"], "start": [-5.0], "end": [2.0]})
        result = compute_tonic_means(z_score, dff, timestamps, epochs)
        np.testing.assert_allclose(result["mean_zscore"].to_numpy(), [1.0])

    def test_window_not_overlapping_recording_raises(self, traces):
        z_score, dff, timestamps = traces
        epochs = pd.DataFrame({"label": ["late"], "start": [15.0], "end": [20.0]})
        with pytest.raises(ValueError, match="does not overlap"):
            compute_tonic_means(z_score, dff, timestamps, epochs)


class TestTonicSerialization:
    def test_write_and_read_epochs_round_trip(self, tmp_path):
        epochs = pd.DataFrame({"label": ["baseline", "post"], "start": [0.0, 8.0], "end": [2.0, 10.0]})
        epochs.to_csv(tmp_path / "tonic_epochs_DMS.csv", index=False)
        loaded = read_tonic_epochs(str(tmp_path), "DMS")
        pd.testing.assert_frame_equal(loaded, epochs)

    def test_read_epochs_missing_file_returns_empty(self, tmp_path):
        loaded = read_tonic_epochs(str(tmp_path), "DMS")
        assert loaded.empty
        assert list(loaded.columns) == ["label", "start", "end"]

    def test_write_tonic_to_hdf5(self, tmp_path):
        tonic_df = pd.DataFrame(
            {"mean_zscore": [1.0, 9.0], "mean_dff": [0.1, 0.9]},
            index=pd.Index(["baseline", "post"], name="epoch"),
        )
        write_tonic_to_hdf5(str(tmp_path), tonic_df, "DMS")
        output_path = tmp_path / "tonic_DMS.h5"
        assert output_path.exists()
        reloaded = pd.read_hdf(output_path, key="df")
        pd.testing.assert_frame_equal(reloaded, tonic_df)
