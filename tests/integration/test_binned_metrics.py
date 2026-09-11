import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.analysis.standard_io import read_binned_metrics_from_hdf5
from guppy.testing.api import step1, step2, step3, step4
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import REPRESENTATIVE_SESSIONS, _locate_output_directory

# The stubbed CSV session runs ~411 s, so 100 s bins give four whole bins and a
# short fifth one -- exercising the ragged final bin end to end.
BIN_WIDTH = 100
RECORDING_SITE = "region"


@pytest.fixture(scope="module")
def preprocessed_session(tmp_path_factory):
    representative_config = REPRESENTATIVE_SESSIONS["csv"]
    source_session = Path(str(STUBBED_TESTING_DATA)) / representative_config["session_subdir"]
    base_directory = tmp_path_factory.mktemp("integration_binned_metrics")
    session = base_directory / Path(source_session).name
    # Output directories are gitignored, so a stale one from a previous local run
    # would otherwise seed this session.
    shutil.copytree(source_session, session, ignore=shutil.ignore_patterns("*_output_*"))

    step1(
        base_dir=str(base_directory),
        selected_folders=[str(session)],
        store_id_to_store_label=representative_config["store_id_to_store_label"],
    )
    output_directory = _locate_output_directory(session_copy=str(session))
    selected_runs = {str(session): [parse_run_name(output_directory)]}

    step2(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
    step3(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)

    return {
        "base_directory": str(base_directory),
        "session": str(session),
        "selected_runs": selected_runs,
        "output_directory": _locate_output_directory(session_copy=str(session)),
    }


@pytest.fixture(scope="module")
def binned_output(preprocessed_session):
    step4(
        base_dir=preprocessed_session["base_directory"],
        selected_folders=[preprocessed_session["session"]],
        selected_runs=preprocessed_session["selected_runs"],
        compute_binned_metrics=True,
        binned_metrics_width=BIN_WIDTH,
    )
    return preprocessed_session["output_directory"]


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestBinnedMetrics:
    def test_writes_both_formats(self, binned_output):
        assert (Path(binned_output) / (f"binned_metrics_{RECORDING_SITE}.h5")).exists()
        assert (Path(binned_output) / (f"binned_metrics_{RECORDING_SITE}.csv")).exists()

    def test_schema(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)

        assert binned.index.name == "bin"
        assert list(binned.columns) == [
            "bin_start",
            "bin_end",
            "n_samples",
            "mean_zscore",
            "mean_dff",
            "transient_count_z_score",
        ]

    def test_bins_tile_the_recording_without_gaps(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        timestamps = read_hdf5("timeCorrection_" + RECORDING_SITE, binned_output, "timestampNew").ravel()

        # Each bin picks up exactly where the previous one ended, and the tiling
        # spans the corrected time axis end to end.
        np.testing.assert_allclose(binned["bin_start"].to_numpy()[1:], binned["bin_end"].to_numpy()[:-1])
        np.testing.assert_allclose(binned["bin_start"].iloc[0], timestamps[0])
        np.testing.assert_allclose(binned["bin_end"].iloc[-1], timestamps[-1])

    def test_final_bin_is_kept_and_is_short(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        widths = (binned["bin_end"] - binned["bin_start"]).to_numpy()

        np.testing.assert_allclose(widths[:-1], BIN_WIDTH)
        assert 0 < widths[-1] < BIN_WIDTH

    def test_means_match_the_traces_on_disk(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        timestamps = read_hdf5("timeCorrection_" + RECORDING_SITE, binned_output, "timestampNew").ravel()
        z_score = read_hdf5("z_score_" + RECORDING_SITE, binned_output, "data").ravel()
        dff = read_hdf5("dff_" + RECORDING_SITE, binned_output, "data").ravel()

        for bin_number, row in binned.iterrows():
            # Half-open, except the final bin which owns its end.
            if bin_number == binned.index[-1]:
                mask = (timestamps >= row["bin_start"]) & (timestamps <= row["bin_end"])
            else:
                mask = (timestamps >= row["bin_start"]) & (timestamps < row["bin_end"])

            assert row["n_samples"] == np.count_nonzero(~np.isnan(z_score[mask]))
            assert row["mean_zscore"] == pytest.approx(np.nanmean(z_score[mask]))
            assert row["mean_dff"] == pytest.approx(np.nanmean(dff[mask]))

    def test_every_sample_lands_in_exactly_one_bin(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        z_score = read_hdf5("z_score_" + RECORDING_SITE, binned_output, "data").ravel()

        assert binned["n_samples"].sum() == np.count_nonzero(~np.isnan(z_score))

    def test_transient_counts_account_for_every_detected_transient(self, binned_output):
        binned = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        occurrences = pd.read_csv(
            Path(binned_output) / (f"transientsOccurrences_z_score_{RECORDING_SITE}.csv"), index_col=0
        )

        assert binned["transient_count_z_score"].sum() == occurrences.shape[0]

    def test_csv_matches_the_hdf5(self, binned_output):
        from_hdf5 = read_binned_metrics_from_hdf5(binned_output, RECORDING_SITE)
        from_csv = pd.read_csv(Path(binned_output) / (f"binned_metrics_{RECORDING_SITE}.csv"), index_col="bin")

        pd.testing.assert_frame_equal(from_csv, from_hdf5, check_dtype=False)
