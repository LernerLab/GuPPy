import os

import pandas as pd
import pytest

from guppy.testing.api import step5


@pytest.mark.parametrize(
    "step4_fixture_name, expected_region, expected_ttl",
    [
        (
            "step4_output_csv",
            "region",
            "ttl",
        ),
        (
            "step4_output_tdt",
            "dms",
            "port_entries_dms",
        ),
        (
            "step4_output_npm",
            "region1",
            "ttl_region1",
        ),
        (
            "step4_output_doric",
            "region",
            "ttl",
        ),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5(step4_fixture_name, expected_region, expected_ttl, request):
    """
    Validate Step 5 outputs for the representative integration sessions.
    """
    pipeline_state = request.getfixturevalue(step4_fixture_name)
    base_directory = str(pipeline_state["base_directory"])
    session_copy = str(pipeline_state["session_copy"])

    step5(
        base_dir=base_directory,
        selected_folders=[str(session_copy)],
        npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
        npm_time_units=pipeline_state["npm_time_units"],
        npm_split_events=pipeline_state["npm_split_events"],
    )

    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = os.path.join(output_directory, "storesList.csv")
    assert os.path.exists(stores_file_path), "Missing storesList.csv after Steps 2-5"

    # Expected PSTH outputs (defaults compute z_score PSTH) - only for datasets with TTLs
    if expected_ttl is None:
        expected_ttl_names = []
    elif isinstance(expected_ttl, str):
        expected_ttl_names = [expected_ttl]
    else:
        expected_ttl_names = expected_ttl

    for expected_ttl_name in expected_ttl_names:
        psth_file_path = os.path.join(
            output_directory,
            f"{expected_ttl_name}_{expected_region}_z_score_{expected_region}.h5",
        )
        baseline_uncorrected_psth_file_path = os.path.join(
            output_directory,
            f"{expected_ttl_name}_{expected_region}_baselineUncorrected_z_score_{expected_region}.h5",
        )
        peak_auc_h5_file_path = os.path.join(
            output_directory,
            f"peak_AUC_{expected_ttl_name}_{expected_region}_z_score_{expected_region}.h5",
        )
        peak_auc_csv_file_path = os.path.join(
            output_directory,
            f"peak_AUC_{expected_ttl_name}_{expected_region}_z_score_{expected_region}.csv",
        )

        # Assert file creation
        assert os.path.exists(psth_file_path), f"Missing PSTH HDF5: {psth_file_path}"
        assert os.path.exists(
            baseline_uncorrected_psth_file_path
        ), f"Missing baseline-uncorrected PSTH HDF5: {baseline_uncorrected_psth_file_path}"
        assert os.path.exists(peak_auc_h5_file_path), f"Missing PSTH Peak/AUC HDF5: {peak_auc_h5_file_path}"
        assert os.path.exists(peak_auc_csv_file_path), f"Missing PSTH Peak/AUC CSV: {peak_auc_csv_file_path}"

        # Basic readability checks: PSTH HDF5 contains a DataFrame with expected columns
        psth_dataframe = pd.read_hdf(psth_file_path, key="df")
        assert "timestamps" in psth_dataframe.columns, f"'timestamps' column missing in {psth_file_path}"
        # The DataFrame should include a 'mean' column per create_Df implementation
        assert "mean" in psth_dataframe.columns, f"'mean' column missing in {psth_file_path}"

    # Additional artifacts from transients frequency/amplitude computation (Step 5 side-effect)
    frequency_and_amplitude_h5_file_path = os.path.join(output_directory, f"freqAndAmp_z_score_{expected_region}.h5")
    frequency_and_amplitude_csv_file_path = os.path.join(output_directory, f"freqAndAmp_z_score_{expected_region}.csv")
    transients_occurrences_csv_file_path = os.path.join(
        output_directory,
        f"transientsOccurrences_z_score_{expected_region}.csv",
    )
    assert os.path.exists(
        frequency_and_amplitude_h5_file_path
    ), f"Missing freq/amp HDF5: {frequency_and_amplitude_h5_file_path}"
    assert os.path.exists(
        frequency_and_amplitude_csv_file_path
    ), f"Missing freq/amp CSV: {frequency_and_amplitude_csv_file_path}"
    assert os.path.exists(
        transients_occurrences_csv_file_path
    ), f"Missing transients occurrences CSV: {transients_occurrences_csv_file_path}"
