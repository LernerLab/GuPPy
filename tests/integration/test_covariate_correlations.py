import glob
import os

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
)
from guppy.testing.covariate_session import (
    BIN_WIDTH,
    COVARIATE_NAMES,
    RECORDING_SITE,
    run_covariate_session,
)
from guppy.testing.scripts.create_stubbed_testing_data import (
    COVARIATE_CSV_DURATION,
    COVARIATE_SCORING_CADENCE,
)

# The sample session runs 600 s, so the 50 s bins give twelve of them. The first
# second is trimmed with the lights-on window, which shortens the first and last bins
# but leaves every bin holding at least one score.
EXPECTED_BIN_COUNT = 12
DRIVING_COVARIATE, NULL_COVARIATE = COVARIATE_NAMES

# The session is synthesized so that akinesia drives the signal and grooming does not.
# Both numbers are what the deterministic generator produces; the contrast between them
# is the assertion with teeth, since crossed covariate columns or a misaligned binning
# would not preserve it.
DRIVING_PEARSON_R = 0.8436
NULL_PEARSON_R = 0.3288


@pytest.fixture(scope="module")
def covariate_session(tmp_path_factory):
    """The behavioral-covariate sample session, run through step 4."""
    base_directory = tmp_path_factory.mktemp("integration_covariates")
    return run_covariate_session(base_directory=base_directory)


class TestCovariateIngestion:
    def test_step2_preserves_values_and_timestamps(self, covariate_session):
        values = np.asarray(read_hdf5(DRIVING_COVARIATE, covariate_session, "data")).ravel()
        timestamps = np.asarray(read_hdf5(DRIVING_COVARIATE, covariate_session, "timestamps")).ravel()
        scored = pd.read_csv(
            os.path.join(
                os.path.dirname(covariate_session),
                DRIVING_COVARIATE + ".csv",
            )
        )

        np.testing.assert_allclose(values, scored["data"].to_numpy())
        np.testing.assert_allclose(timestamps, np.arange(0.0, COVARIATE_CSV_DURATION, COVARIATE_SCORING_CADENCE))

    def test_step3_does_not_treat_the_covariate_as_an_event(self, covariate_session):
        # Regression guard: a covariate must not go down the event-correction path,
        # which keeps only timestamps and drops the scored values.
        assert glob.glob(os.path.join(covariate_session, "covariate_*_" + RECORDING_SITE + ".hdf5")) == []

    def test_step4_computes_no_psth_for_the_covariate(self, covariate_session):
        psth_files = glob.glob(os.path.join(covariate_session, "*covariate*psth*"))
        for name in COVARIATE_NAMES:
            psth_files += glob.glob(os.path.join(covariate_session, "*" + name + "*_z_score*"))

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
            "akinesia",
            "grooming",
            "n_samples_akinesia",
            "n_samples_grooming",
        ]
        assert binned.index.name == "bin"

    def test_bins_cover_the_session_at_the_requested_width(self, covariate_session):
        binned = read_binned_covariates_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)

        assert len(binned) == EXPECTED_BIN_COUNT
        np.testing.assert_allclose(binned["bin_start"].to_numpy()[1:5], [51.0, 101.0, 151.0, 201.0])
        np.testing.assert_allclose(np.diff(binned["bin_start"].to_numpy()), BIN_WIDTH)

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

    def test_every_metric_is_correlated_against_every_covariate(self, covariate_session):
        correlations = read_covariate_correlations_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        metrics = read_binned_metrics_from_hdf5(covariate_session, RECORDING_SITE)
        expected_metrics = [name for name in metrics.columns if name not in ("bin_start", "bin_end", "n_samples")]

        assert set(correlations["covariate"]) == set(COVARIATE_NAMES)
        assert sorted(correlations["metric"]) == sorted(name for name in expected_metrics for _ in COVARIATE_NAMES)

    def test_the_driving_covariate_correlates_and_the_null_one_does_not(self, covariate_session):
        correlations = read_covariate_correlations_from_hdf5(filepath=covariate_session, recording_site=RECORDING_SITE)
        mean_zscore = correlations[correlations["metric"] == "mean_zscore"].set_index("covariate")

        assert mean_zscore.loc[DRIVING_COVARIATE, "pearson_r"] == pytest.approx(DRIVING_PEARSON_R, abs=0.02)
        assert mean_zscore.loc[NULL_COVARIATE, "pearson_r"] == pytest.approx(NULL_PEARSON_R, abs=0.02)

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
