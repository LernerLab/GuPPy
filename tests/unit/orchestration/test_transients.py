import json
import os

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import write_hdf5
from guppy.analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
    write_binned_metrics_to_hdf5,
)
from guppy.orchestration.transients import (
    executeFindFreqAndAmp,
    findBinnedMetrics,
    findCovariateCorrelations,
)


@pytest.fixture
def transient_params(base_input_parameters):
    base_input_parameters["numberOfCores"] = 1
    base_input_parameters["moving_window"] = 15
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
    return calls


class TestExecuteFindFreqAndAmpDispatch:
    def test_individual_path(self, transient_params, capture_dispatch):
        transient_params["combine_data"] = False
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"individual"}
        assert capture_dispatch["individual"] == ["/session_1"]

    def test_combined_path(self, transient_params, capture_dispatch):
        transient_params["combine_data"] = True
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"combined"}
        assert capture_dispatch["combined"] == ["/session_1"]


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


class TestFindCovariateCorrelations:
    @pytest.fixture
    def run_folder(self, tmp_path):
        """A run folder with one binned-metrics table and one covariate store."""
        np.savetxt(
            os.path.join(str(tmp_path), "storesList.csv"),
            np.array([["akinesia", "Dv2A"], ["covariate_akinesia", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        # Scores at 0, 2, 5 and 9 s bin to means of 2.0, 4.0 and 10.0 on the edges below.
        write_hdf5(np.array([0.0, 2.0, 5.0, 9.0]), "akinesia", str(tmp_path), "timestamps")
        write_hdf5(np.array([1.0, 3.0, 4.0, 10.0]), "akinesia", str(tmp_path), "data")

        binned_metrics = pd.DataFrame(
            {
                "bin_start": [0.0, 4.0, 8.0],
                "bin_end": [4.0, 8.0, 10.0],
                "n_samples": [4, 4, 2],
                "mean_zscore": [1.5, 5.5, 9.0],
            },
            index=pd.RangeIndex(3, name="bin"),
        )
        write_binned_metrics_to_hdf5(str(tmp_path), binned_metrics, "dms")
        return str(tmp_path)

    def test_writes_binned_covariates(self, run_folder, base_input_parameters):
        findCovariateCorrelations(run_folder, base_input_parameters, ["dms"])

        binned = read_binned_covariates_from_hdf5(filepath=run_folder, recording_site="dms")
        np.testing.assert_allclose(binned["akinesia"].to_numpy(), [2.0, 4.0, 10.0])
        np.testing.assert_array_equal(binned["n_samples_akinesia"].to_numpy(), [2, 1, 1])

    def test_bins_match_the_metrics_table_exactly(self, run_folder, base_input_parameters):
        findCovariateCorrelations(run_folder, base_input_parameters, ["dms"])

        binned = read_binned_covariates_from_hdf5(filepath=run_folder, recording_site="dms")
        metrics = read_binned_metrics_from_hdf5(run_folder, "dms")
        np.testing.assert_allclose(binned["bin_start"].to_numpy(), metrics["bin_start"].to_numpy())
        np.testing.assert_allclose(binned["bin_end"].to_numpy(), metrics["bin_end"].to_numpy())

    def test_writes_correlations(self, run_folder, base_input_parameters):
        findCovariateCorrelations(run_folder, base_input_parameters, ["dms"])

        correlations = read_covariate_correlations_from_hdf5(filepath=run_folder, recording_site="dms")
        assert len(correlations) == 1
        row = correlations.iloc[0]
        assert row["metric"] == "mean_zscore"
        assert row["covariate"] == "akinesia"
        assert row["pearson_r"] == pytest.approx(0.9493907, abs=1e-6)
        assert row["n_bins"] == 3

    def test_writes_csv_twins(self, run_folder, base_input_parameters):
        findCovariateCorrelations(run_folder, base_input_parameters, ["dms"])

        assert os.path.exists(os.path.join(run_folder, "binned_covariates_dms.csv"))
        assert os.path.exists(os.path.join(run_folder, "covariate_correlations_dms.csv"))

    def test_no_covariate_store_writes_nothing(self, tmp_path, base_input_parameters):
        np.savetxt(
            os.path.join(str(tmp_path), "storesList.csv"),
            np.array([["Dv2A", "PrtN"], ["signal_dms", "port_entries"]]),
            delimiter=",",
            fmt="%s",
        )

        findCovariateCorrelations(str(tmp_path), base_input_parameters, ["dms"])

        assert not os.path.exists(os.path.join(str(tmp_path), "covariate_correlations_dms.h5"))

    def test_concatenated_outputs_raise(self, run_folder, base_input_parameters):
        with open(os.path.join(run_folder, "GuPPyParamtersUsed.json"), "w") as parameters_file:
            json.dump({"removeArtifacts": True, "artifactsRemovalMethod": "concatenate"}, parameters_file)

        with pytest.raises(ValueError, match="concatenate"):
            findCovariateCorrelations(run_folder, base_input_parameters, ["dms"])
