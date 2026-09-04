"""End-to-end test that GuPPy's optional analysis outputs reach the exported NWB file.

The four products added after the first export landed -- tonic epoch means, binned metrics,
behavioral covariates and spontaneous mode -- are discovered by ``GuppyInterface`` from the files
a run happens to hold, so a run that skipped one simply exports without it. That discovery is
what these tests exercise, against two stubbed sessions run through the real pipeline:

* the behavioral-covariate session, run with **Compute Binned Metrics?** on and then through
  **Tonic Analysis**, which produces all four of the whole-session tables at once;
* the plain CSV session run with **Use Transients as Events?** on, whose detected transients
  stand in for external TTLs.

Every assertion compares the NWB objects against the GuPPy files on disk they were built from,
so a version of the export that silently omits one of them fails here.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
from pynwb import NWBHDF5IO

from guppy.analysis.io_utils import read_hdf5
from guppy.analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
)
from guppy.orchestration.metadata import METADATA_FILENAME
from guppy.testing.api import step1, step2, step3, step4, step7, tonic_analysis
from guppy.testing.covariate_session import (
    COVARIATE_NAMES,
    RECORDING_SITE,
    SESSION_NAME,
    run_covariate_session,
)
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import write_metadata_yaml

# Both trace types the interface writes a row per, in the order it emits them.
TRACE_TYPES = ["z_score", "dff"]

# The covariate session runs 600 s in 50 s bins, so twelve bins, each contributing one row per
# trace type to the metrics table and one per covariate to the covariates table. The three
# binned-metric columns correlated against each of the two covariates give six correlation rows.
EXPECTED_BIN_COUNT = 12
EXPECTED_CORRELATION_COUNT = 6

# GuPPy names a correlated quantity with one composite string; the export splits it back into the
# trace it was measured on and the metric taken over that trace.
GUPPY_METRIC_TO_TRACE_TYPE_AND_METRIC = {
    "mean_zscore": ("z_score", "mean"),
    "mean_dff": ("dff", "mean"),
    "transient_count_z_score": ("z_score", "transient_count"),
    "transient_count_dff": ("dff", "transient_count"),
}

# Two windows clear of each other inside the session's 600 s, named as the epoch page would name
# them. Each contributes one row per trace type.
TONIC_EPOCHS = pd.DataFrame(
    {"label": ["baseline", "late"], "start": [5.0, 400.0], "end": [200.0, 590.0]},
)

SPONTANEOUS_SESSION_NAME = "sample_data_csv_1"
SPONTANEOUS_STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
SPONTANEOUS_RECORDING_SITE = "region"
# GuPPy names the event train after the site whose transients it holds; the registry keeps the
# metric and the site structured instead, so its row is named for the metric alone.
TRANSIENT_EVENT = f"transients_z_score_{SPONTANEOUS_RECORDING_SITE}"
TRANSIENT_REGISTRY_NAME = "transients_z_score"


def export_run(*, session: str, output_directory: str, acquisition_format: str) -> str:
    """Write the Step 6 metadata overlay into ``output_directory``, then run Step 7 over the run."""
    write_metadata_yaml(
        session_folder_path=session,
        output_directory=output_directory,
        acquisition_format=acquisition_format,
        path=os.path.join(output_directory, METADATA_FILENAME),
    )
    base_dir = os.path.dirname(session)
    step7(
        base_dir=base_dir,
        selected_folders=[session],
        selected_runs={session: [parse_run_name(output_directory)]},
    )
    return os.path.join(output_directory, f"{os.path.basename(output_directory)}.nwb")


class TestCovariateAndWholeSessionOutputs:
    """The covariate session exported after Step 4 with binned metrics and after Tonic Analysis."""

    @pytest.fixture(scope="class")
    def exported(self, tmp_path_factory) -> dict:
        base_directory = tmp_path_factory.mktemp("export_optional_outputs")
        output_directory = run_covariate_session(
            session_path=STUBBED_TESTING_DATA / "csv" / SESSION_NAME,
            base_directory=base_directory,
        )
        session = os.path.dirname(output_directory)
        tonic_analysis(
            base_dir=str(base_directory),
            selected_folders=[session],
            tonic_epochs={RECORDING_SITE: TONIC_EPOCHS},
            selected_runs={session: [parse_run_name(output_directory)]},
        )
        nwbfile_path = export_run(session=session, output_directory=output_directory, acquisition_format="csv")
        return {"output_directory": output_directory, "nwbfile_path": nwbfile_path}

    @pytest.fixture(scope="class")
    def nwbfile(self, exported):
        """The exported file, read once and held open for the whole class."""
        with NWBHDF5IO(exported["nwbfile_path"], "r") as io:
            yield io.read()

    @pytest.fixture(scope="class")
    def guppy_module(self, nwbfile):
        return nwbfile.processing["guppy"]

    def test_every_optional_product_is_present(self, guppy_module):
        assert {
            "tonic_epochs",
            "binned_metrics",
            "binned_covariates",
            "covariate_correlations",
            *COVARIATE_NAMES,
        } <= set(guppy_module.data_interfaces)

    def test_covariate_series_carry_the_scored_values(self, guppy_module, exported):
        for covariate_name in COVARIATE_NAMES:
            series = guppy_module[covariate_name]
            scored = pd.read_csv(os.path.join(os.path.dirname(exported["output_directory"]), f"{covariate_name}.csv"))

            np.testing.assert_allclose(series.data[:], scored["data"].to_numpy())
            np.testing.assert_allclose(series.timestamps[:], scored["timestamps"].to_numpy())

    def test_binned_metrics_match_the_hdf5(self, guppy_module, exported):
        binned = read_binned_metrics_from_hdf5(
            filepath=exported["output_directory"], recording_site=RECORDING_SITE
        ).reset_index(drop=True)
        table = guppy_module["binned_metrics"].to_dataframe()

        assert len(binned) == EXPECTED_BIN_COUNT
        assert len(table) == EXPECTED_BIN_COUNT * len(TRACE_TYPES)
        # One row per (bin, trace type), the trace types cycling fastest within each bin.
        assert list(table["trace_type"]) == TRACE_TYPES * EXPECTED_BIN_COUNT

        for trace_type, mean_column in zip(TRACE_TYPES, ["mean_zscore", "mean_dff"], strict=True):
            rows = table[table["trace_type"] == trace_type].reset_index(drop=True)
            np.testing.assert_allclose(rows["start_time"], binned["bin_start"])
            np.testing.assert_allclose(rows["stop_time"], binned["bin_end"])
            np.testing.assert_allclose(rows["mean"], binned[mean_column])

        # The session detected transients on the z-score trace only, so the ΔF/F rows have a mean
        # but no count -- the shape a run that binned one trace type and not the other exports as.
        z_score_rows = table[table["trace_type"] == "z_score"].reset_index(drop=True)
        np.testing.assert_allclose(z_score_rows["transient_count"], binned["transient_count_z_score"])
        assert table[table["trace_type"] == "dff"]["transient_count"].isna().all()

    def test_binned_covariates_match_the_hdf5(self, guppy_module, exported):
        binned = read_binned_covariates_from_hdf5(
            filepath=exported["output_directory"], recording_site=RECORDING_SITE
        ).reset_index(drop=True)
        table = guppy_module["binned_covariates"].to_dataframe()

        assert len(table) == EXPECTED_BIN_COUNT * len(COVARIATE_NAMES)

        for index, covariate_name in enumerate(COVARIATE_NAMES):
            # The covariates cycle fastest within each bin, in the order Step 1 labeled them.
            rows = table.iloc[index :: len(COVARIATE_NAMES)].reset_index(drop=True)
            # Each row points at the TimeSeries holding that covariate's own scored trace.
            assert {series.name for series in rows["covariate"]} == {covariate_name}
            np.testing.assert_allclose(rows["start_time"], binned["bin_start"])
            np.testing.assert_allclose(rows["mean"], binned[covariate_name])

    def test_covariate_correlations_carry_the_coefficients(self, guppy_module, exported):
        correlations = read_covariate_correlations_from_hdf5(
            filepath=exported["output_directory"], recording_site=RECORDING_SITE
        )
        table = guppy_module["covariate_correlations"].to_dataframe()

        assert len(table) == EXPECTED_CORRELATION_COUNT
        # GuPPy names the correlated quantity with one composite string; the interface splits it
        # back into the trace it was measured on and the metric taken over that trace.
        assert set(zip(table["trace_type"], table["metric"], strict=True)) == {
            ("z_score", "mean"),
            ("dff", "mean"),
            ("z_score", "transient_count"),
        }

        exported_coefficients = {
            (series.name, trace_type, metric): (pearson_r, spearman_rho)
            for series, trace_type, metric, pearson_r, spearman_rho in zip(
                table["covariate"],
                table["trace_type"],
                table["metric"],
                table["pearson_r"],
                table["spearman_rho"],
                strict=True,
            )
        }
        for row in correlations.itertuples(index=False):
            key = (row.covariate, *GUPPY_METRIC_TO_TRACE_TYPE_AND_METRIC[row.metric])
            assert exported_coefficients[key] == pytest.approx((row.pearson_r, row.spearman_rho))

    def test_tonic_epochs_match_the_hdf5(self, guppy_module, exported):
        means = pd.read_hdf(os.path.join(exported["output_directory"], f"tonic_{RECORDING_SITE}.h5"), key="df")
        table = guppy_module["tonic_epochs"].to_dataframe()

        assert len(table) == len(TONIC_EPOCHS) * len(TRACE_TYPES)
        assert list(table["label"]) == ["baseline", "baseline", "late", "late"]
        np.testing.assert_allclose(table["start_time"], np.repeat(TONIC_EPOCHS["start"], len(TRACE_TYPES)))
        np.testing.assert_allclose(table["stop_time"], np.repeat(TONIC_EPOCHS["end"], len(TRACE_TYPES)))

        for trace_type, mean_column in zip(TRACE_TYPES, ["mean_zscore", "mean_dff"], strict=True):
            rows = table[table["trace_type"] == trace_type].reset_index(drop=True)
            np.testing.assert_allclose(rows["mean"], means[mean_column].to_numpy())

    def test_the_run_records_that_binned_metrics_were_computed(self, nwbfile):
        assert nwbfile.lab_meta_data["guppy_parameters"].compute_binned_metrics


class TestSpontaneousModeOutputs:
    """A run whose own transients stood in for external TTLs must export as such."""

    @pytest.fixture(scope="class")
    def exported(self, tmp_path_factory) -> str:
        base_directory = tmp_path_factory.mktemp("export_spontaneous")
        session = str(base_directory / SPONTANEOUS_SESSION_NAME)
        source = STUBBED_TESTING_DATA / "csv" / SPONTANEOUS_SESSION_NAME
        shutil.copytree(source, session, ignore=shutil.ignore_patterns("*_output_*"))

        common = dict(base_dir=str(base_directory), selected_folders=[session])
        step1(**common, store_id_to_store_label=SPONTANEOUS_STORE_ID_TO_STORE_LABEL)
        output_directory = os.path.join(session, f"{SPONTANEOUS_SESSION_NAME}_output_1")
        selected_runs = {session: ["1"]}
        step2(**common, selected_runs=selected_runs)
        step3(**common, selected_runs=selected_runs)
        step4(**common, selected_runs=selected_runs, use_transients_as_events=True)

        return export_run(session=session, output_directory=output_directory, acquisition_format="csv")

    @pytest.fixture(scope="class")
    def nwbfile(self, exported):
        with NWBHDF5IO(exported, "r") as io:
            yield io.read()

    def test_the_transient_event_train_reaches_the_analyzed_events(self, nwbfile):
        analyzed_events = nwbfile.events["GuppyEvents"].to_dataframe()

        # The transients GuPPy aligned to are events in their own right, alongside the TTL the
        # session also recorded.
        assert TRANSIENT_EVENT in set(analyzed_events["event_type"])
        assert nwbfile.lab_meta_data["guppy_parameters"].use_transients_as_events

    def test_the_transient_onsets_are_the_ones_detected(self, nwbfile, exported):
        registry = nwbfile.processing["guppy"]["events"].to_dataframe()
        transient_rows = registry[registry["event_name"] == TRANSIENT_REGISTRY_NAME]

        assert len(transient_rows) == 1
        # The registry row selects exactly this event's onsets out of the merged events table.
        occurrences = transient_rows["events"].iloc[0]
        assert set(occurrences["event_type"]) == {TRANSIENT_EVENT}

        # Against the event train Step 4 actually aligned to, which is the subset of detected
        # transients it kept after dropping the ones too close to the recording start or each other.
        event_timestamps = np.asarray(read_hdf5(TRANSIENT_EVENT, os.path.dirname(exported), "ts")).ravel()
        np.testing.assert_allclose(np.sort(occurrences["timestamp"].to_numpy()), np.sort(event_timestamps))
