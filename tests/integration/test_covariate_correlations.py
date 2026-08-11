import glob
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
)
from guppy.testing.api import step1, step2, step3, step4
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import REPRESENTATIVE_SESSIONS, _locate_output_directory

# The stubbed CSV session runs ~411 s, so 100 s bins give four whole bins and a
# short fifth one. Scoring every 50 s puts at least one sample in every bin.
BIN_WIDTH = 100
COVARIATE_CADENCE = 50.0
RECORDING_SITE = "region"
COVARIATE_NAME = "akinesia"


@pytest.fixture(scope="module")
def covariate_session(tmp_path_factory):
    """A stubbed CSV session carrying a hand-scored behavioral covariate, run through step 4."""
    representative_config = REPRESENTATIVE_SESSIONS["csv"]
    source_session = os.path.join(str(STUBBED_TESTING_DATA), representative_config["session_subdir"])
    base_directory = tmp_path_factory.mktemp("integration_covariates")
    session = base_directory / os.path.basename(source_session)
    shutil.copytree(source_session, session, ignore=shutil.ignore_patterns("*_output_*"))

    signal = pd.read_csv(session / "Sample_Signal_Channel.csv")
    span_end = float(signal["timestamps"].iloc[-1])
    timestamps = np.arange(0.0, span_end, COVARIATE_CADENCE)
    pd.DataFrame(
        {
            "timestamps": timestamps,
            "data": np.arange(timestamps.size, dtype=float),
            "sampling_rate": np.full(timestamps.size, 1.0 / COVARIATE_CADENCE),
        }
    ).to_csv(session / (COVARIATE_NAME + ".csv"), index=False)

    store_id_to_store_label = dict(representative_config["store_id_to_store_label"])
    store_id_to_store_label[COVARIATE_NAME] = "covariate_" + COVARIATE_NAME

    step1(
        base_dir=str(base_directory),
        selected_folders=[str(session)],
        store_id_to_store_label=store_id_to_store_label,
    )
    output_directory = _locate_output_directory(session_copy=str(session))
    selected_runs = {str(session): [parse_run_name(output_directory)]}

    step2(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
    step3(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
    step4(
        base_dir=str(base_directory),
        selected_folders=[str(session)],
        selected_runs=selected_runs,
        compute_binned_metrics=True,
        binned_metrics_width=BIN_WIDTH,
    )
    return _locate_output_directory(session_copy=str(session))


class TestCovariateIngestion:
    def test_step2_preserves_values_and_timestamps(self, covariate_session):
        values = np.asarray(read_hdf5(COVARIATE_NAME, covariate_session, "data")).ravel()
        timestamps = np.asarray(read_hdf5(COVARIATE_NAME, covariate_session, "timestamps")).ravel()

        np.testing.assert_allclose(values, np.arange(values.size, dtype=float))
        np.testing.assert_allclose(timestamps, np.arange(0.0, timestamps.size * COVARIATE_CADENCE, COVARIATE_CADENCE))

    def test_step3_does_not_treat_the_covariate_as_an_event(self, covariate_session):
        # Regression guard: a covariate must not go down the event-correction path,
        # which keeps only timestamps and drops the scored values.
        assert glob.glob(os.path.join(covariate_session, "covariate_*_" + RECORDING_SITE + ".hdf5")) == []

    def test_step4_computes_no_psth_for_the_covariate(self, covariate_session):
        psth_files = glob.glob(os.path.join(covariate_session, "*covariate*psth*"))
        psth_files += glob.glob(os.path.join(covariate_session, "*" + COVARIATE_NAME + "*_z_score*"))

        assert psth_files == []


class TestCovariateOutputs:
    def test_all_four_files_are_written(self, covariate_session):
        for name in [
            "binned_covariates_" + RECORDING_SITE + ".h5",
            "binned_covariates_" + RECORDING_SITE + ".csv",
            "covariate_correlations_" + RECORDING_SITE + ".h5",
            "covariate_correlations_" + RECORDING_SITE + ".csv",
        ]:
            assert os.path.exists(os.path.join(covariate_session, name)), name

    def test_binned_covariates_schema(self, covariate_session):
        binned = read_binned_covariates_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)

        assert list(binned.columns) == [
            "bin_start",
            "bin_end",
            COVARIATE_NAME,
            "n_samples_" + COVARIATE_NAME,
        ]
        assert binned.index.name == "bin"

    def test_bins_are_identical_to_the_metrics_table(self, covariate_session):
        binned = read_binned_covariates_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        metrics = read_binned_metrics_from_hdf5(covariate_session, RECORDING_SITE)

        assert len(binned) == len(metrics)
        np.testing.assert_allclose(binned["bin_start"].to_numpy(), metrics["bin_start"].to_numpy())
        np.testing.assert_allclose(binned["bin_end"].to_numpy(), metrics["bin_end"].to_numpy())

    def test_correlations_schema_has_no_p_value(self, covariate_session):
        correlations = read_covariate_correlations_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)

        assert list(correlations.columns) == ["metric", "covariate", "pearson_r", "spearman_rho", "n_bins"]
        assert correlations.index.name == "pair"

    def test_every_metric_is_correlated_against_the_covariate(self, covariate_session):
        correlations = read_covariate_correlations_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        metrics = read_binned_metrics_from_hdf5(covariate_session, RECORDING_SITE)
        expected_metrics = [name for name in metrics.columns if name not in ("bin_start", "bin_end", "n_samples")]

        assert sorted(correlations["metric"]) == sorted(expected_metrics)
        assert set(correlations["covariate"]) == {COVARIATE_NAME}

    def test_reported_bin_count_matches_the_tables_on_disk(self, covariate_session):
        correlations = read_covariate_correlations_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        binned = read_binned_covariates_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        metrics = read_binned_metrics_from_hdf5(covariate_session, RECORDING_SITE)

        for _, row in correlations.iterrows():
            both_finite = ~np.isnan(binned[row["covariate"]].to_numpy()) & ~np.isnan(
                metrics[row["metric"]].to_numpy(dtype=float)
            )
            assert row["n_bins"] == int(np.count_nonzero(both_finite)), row["metric"]

    def test_csv_matches_hdf5(self, covariate_session):
        from_hdf5 = read_binned_covariates_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        from_csv = pd.read_csv(
            os.path.join(covariate_session, "binned_covariates_" + RECORDING_SITE + ".csv"), index_col="bin"
        )

        pd.testing.assert_frame_equal(from_hdf5, from_csv, check_dtype=False)
