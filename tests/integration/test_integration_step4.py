import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.testing.api import step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.parametrize(
    "step3_fixture_name, expected_recording_site, expected_ttl",
    [
        (
            "step3_output_csv",
            "region",
            "ttl",
        ),
        (
            "step3_output_tdt",
            "dms",
            "port_entries_dms",
        ),
        (
            "step3_output_npm",
            "region1",
            "ttl_region1",
        ),
        (
            "step3_output_doric",
            "region",
            "ttl",
        ),
        (
            "step3_output_nwb",
            "region",
            "ttl",
        ),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step4(step3_fixture_name, expected_recording_site, expected_ttl, request):
    """
    Validate Step 4 outputs for the representative integration sessions.
    """
    pipeline_state = request.getfixturevalue(step3_fixture_name)
    base_directory = str(pipeline_state["base_directory"])
    session_copy = str(pipeline_state["session_copy"])

    step4(
        base_dir=base_directory,
        selected_folders=[str(session_copy)],
        npm_timestamp_column_name=pipeline_state["npm_timestamp_column_name"],
        npm_time_unit=pipeline_state["npm_time_unit"],
        npm_split_events=pipeline_state["npm_split_events"],
        selected_runs={session_copy: ["1"]},
    )

    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = Path(output_directory) / "storesList.csv"
    assert Path(stores_file_path).exists(), "Missing storesList.csv after Steps 2-5"

    # Expected PSTH outputs (defaults compute z_score PSTH) - only for datasets with TTLs
    if expected_ttl is None:
        expected_ttl_names = []
    elif isinstance(expected_ttl, str):
        expected_ttl_names = [expected_ttl]
    else:
        expected_ttl_names = expected_ttl

    for expected_ttl_name in expected_ttl_names:
        psth_file_path = Path(output_directory) / (
            f"{expected_ttl_name}_{expected_recording_site}_z_score_{expected_recording_site}.h5"
        )
        baseline_uncorrected_psth_file_path = Path(output_directory) / (
            f"{expected_ttl_name}_{expected_recording_site}_baselineUncorrected_z_score_{expected_recording_site}.h5"
        )
        peak_auc_h5_file_path = Path(output_directory) / (
            f"peak_AUC_{expected_ttl_name}_{expected_recording_site}_z_score_{expected_recording_site}.h5"
        )
        peak_auc_csv_file_path = Path(output_directory) / (
            f"peak_AUC_{expected_ttl_name}_{expected_recording_site}_z_score_{expected_recording_site}.csv"
        )

        # Assert file creation
        assert Path(psth_file_path).exists(), f"Missing PSTH HDF5: {psth_file_path}"
        assert Path(
            baseline_uncorrected_psth_file_path
        ).exists(), f"Missing baseline-uncorrected PSTH HDF5: {baseline_uncorrected_psth_file_path}"
        assert Path(peak_auc_h5_file_path).exists(), f"Missing PSTH Peak/AUC HDF5: {peak_auc_h5_file_path}"
        assert Path(peak_auc_csv_file_path).exists(), f"Missing PSTH Peak/AUC CSV: {peak_auc_csv_file_path}"

        # Basic readability checks: PSTH HDF5 contains a DataFrame with expected columns
        psth_dataframe = pd.read_hdf(psth_file_path, key="df")
        assert "timestamps" in psth_dataframe.columns, f"'timestamps' column missing in {psth_file_path}"
        # The DataFrame should include a 'mean' column per create_Df implementation
        assert "mean" in psth_dataframe.columns, f"'mean' column missing in {psth_file_path}"

    # Additional artifacts from transients frequency/amplitude computation (Step 4 side-effect)
    frequency_and_amplitude_h5_file_path = Path(output_directory) / (f"freqAndAmp_z_score_{expected_recording_site}.h5")
    frequency_and_amplitude_csv_file_path = Path(output_directory) / (
        f"freqAndAmp_z_score_{expected_recording_site}.csv"
    )
    transients_occurrences_csv_file_path = Path(output_directory) / (
        f"transientsOccurrences_z_score_{expected_recording_site}.csv"
    )
    assert Path(
        frequency_and_amplitude_h5_file_path
    ).exists(), f"Missing freq/amp HDF5: {frequency_and_amplitude_h5_file_path}"
    assert Path(
        frequency_and_amplitude_csv_file_path
    ).exists(), f"Missing freq/amp CSV: {frequency_and_amplitude_csv_file_path}"
    assert Path(
        transients_occurrences_csv_file_path
    ).exists(), f"Missing transients occurrences CSV: {transients_occurrences_csv_file_path}"

    # Binned metrics are opt-in, so a default Step 4 must not produce them.
    binned_metrics_file_paths = list(Path(output_directory).glob("binned_metrics_*"))
    assert binned_metrics_file_paths == [], f"Unexpected binned metrics outputs: {binned_metrics_file_paths}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step4_rejects_events_that_share_no_timeline_with_the_signal(tmp_path):
    """Step 4 refuses an event store on a different clock than the signal.

    A CSV event file is read verbatim, so nothing before Step 4 can tell that its timestamps
    belong to another clock. Steps 1-3 therefore succeed and the mismatch used to surface as an
    out-of-range index inside the PSTH pools.
    """
    base_directory = tmp_path / "data_root"
    base_directory.mkdir()
    session_copy = base_directory / "sample_data_csv_1"
    shutil.copytree(Path(STUBBED_TESTING_DATA) / "csv" / "sample_data_csv_1", session_copy)
    for stale_output in session_copy.glob("sample_data_csv_1_output_*"):
        shutil.rmtree(stale_output)

    # The session's photometry spans [0, 411]s; these sit ~50000s away, as an unconverted
    # acquisition clock would.
    np.savetxt(
        session_copy / "Sample_TTL.csv",
        np.array([49956.0, 50531.0, 51107.0]),
        header="timestamps",
        comments="",
        fmt="%.6f",
    )

    base_dir = str(base_directory)
    selected_folders = [str(session_copy)]
    store_id_to_store_label = {
        "Sample_Control_Channel": "control_region",
        "Sample_Signal_Channel": "signal_region",
        "Sample_TTL": "ttl",
    }
    selected_runs = {str(session_copy): ["1"]}

    step1(base_dir=base_dir, selected_folders=selected_folders, store_id_to_store_label=store_id_to_store_label)
    step2(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step3(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)

    with pytest.raises(ValueError, match=r"no trial overlaps the 'region' signal"):
        step4(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)

    output_directory = next(session_copy.glob("sample_data_csv_1_output_*"))
    assert list(output_directory.glob("ttl_region_z_score_region.h5")) == []
