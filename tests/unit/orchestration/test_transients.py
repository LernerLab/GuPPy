import os

import numpy as np
import pytest

from guppy.analysis.io_utils import write_hdf5
from guppy.analysis.standard_io import read_binned_metrics_from_hdf5
from guppy.orchestration.transients import (
    execute_average_for_group,
    executeFindFreqAndAmp,
    findBinnedMetrics,
)


@pytest.fixture
def transient_params(base_input_parameters):
    base_input_parameters["numberOfCores"] = 1
    base_input_parameters["moving_window"] = 15
    base_input_parameters["group_session_folders"] = ["/group_1"]
    base_input_parameters["session_folders"] = ["/session_1"]
    return base_input_parameters


@pytest.fixture
def capture_dispatch(monkeypatch):
    """Capture which compute routine executeFindFreqAndAmp dispatches to (none of them serve)."""
    calls = {}
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_find_freq_and_amp",
        lambda ip, sf, mw, procs: calls.setdefault("individual", sf),
    )
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_find_freq_and_amp_combined",
        lambda ip, sf, mw, procs: calls.setdefault("combined", sf),
    )
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_average_for_group",
        lambda ip, folders: calls.setdefault("average", folders),
    )
    return calls


class TestExecuteFindFreqAndAmpDispatch:
    def test_individual_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = False
        transient_params["combine_data"] = False
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"individual"}
        assert capture_dispatch["individual"] == ["/session_1"]

    def test_combined_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = False
        transient_params["combine_data"] = True
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"combined"}
        assert capture_dispatch["combined"] == ["/session_1"]

    def test_average_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = True
        transient_params["combine_data"] = False
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"average"}
        assert capture_dispatch["average"] == ["/group_1"]


def test_execute_average_for_group_raises_for_empty_folders(base_input_parameters):
    with pytest.raises(ValueError, match="No folders selected for group averaging"):
        execute_average_for_group(base_input_parameters, [])


class TestFindBinnedMetrics:
    @pytest.fixture
    def run_folder(self, tmp_path):
        # 0..10 s at 1 Hz, with z-score equal to time so bin means are obvious.
        timestamps = np.arange(0, 11, 1, dtype=float)
        write_hdf5(timestamps, "timeCorrection_dms", str(tmp_path), "timestampNew")
        write_hdf5(timestamps.copy(), "z_score_dms", str(tmp_path), "data")
        write_hdf5(timestamps / 10, "dff_dms", str(tmp_path), "data")
        return str(tmp_path)

    def test_writes_one_table_per_recording_site(self, run_folder, base_input_parameters):
        base_input_parameters["binnedMetricsWidth"] = 5
        findBinnedMetrics(run_folder, base_input_parameters, {"dms": {"z_score": np.array([1.0, 6.0, 7.0])}})

        binned = read_binned_metrics_from_hdf5(run_folder, "dms")
        # bin 0 = [0,5) -> 0..4 -> 2.0;  bin 1 = [5,10] -> 5..10 -> 7.5
        np.testing.assert_allclose(binned["mean_zscore"].to_numpy(), [2.0, 7.5])
        np.testing.assert_allclose(binned["mean_dff"].to_numpy(), [0.2, 0.75])
        np.testing.assert_array_equal(binned["transient_count_z_score"].to_numpy(), [1, 2])
        assert os.path.exists(os.path.join(run_folder, "binned_metrics_dms.csv"))

    def test_bins_span_the_uncompressed_time_axis(self, run_folder, base_input_parameters):
        # Regression guard: binning must use timestampNew, not the NaN-stripped
        # time axis the transient detector works on.
        base_input_parameters["binnedMetricsWidth"] = 5
        findBinnedMetrics(run_folder, base_input_parameters, {"dms": {"z_score": np.array([])}})

        binned = read_binned_metrics_from_hdf5(run_folder, "dms")
        np.testing.assert_allclose(binned["bin_start"].iloc[0], 0.0)
        np.testing.assert_allclose(binned["bin_end"].iloc[-1], 10.0)

    def test_both_metrics_produce_two_count_columns(self, run_folder, base_input_parameters):
        base_input_parameters["binnedMetricsWidth"] = 5
        findBinnedMetrics(
            run_folder,
            base_input_parameters,
            {"dms": {"z_score": np.array([1.0]), "dff": np.array([6.0, 7.0])}},
        )

        binned = read_binned_metrics_from_hdf5(run_folder, "dms")
        np.testing.assert_array_equal(binned["transient_count_z_score"].to_numpy(), [1, 0])
        np.testing.assert_array_equal(binned["transient_count_dff"].to_numpy(), [0, 2])
