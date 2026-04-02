import glob
import os
import shutil

import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2, step3, step4, step5

SESSION_SUBDIR = "csv/sample_data_csv_1"
STORENAMES_MAP = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
EXPECTED_REGION = "region"
EXPECTED_TTL = "ttl"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dff(tmp_path):
    """
    Integration test: run the full pipeline (Steps 2-5) with ΔF/F selected for both
    PSTH computation and transient detection, and assert that the expected output files
    are created with the correct structure.
    """
    source_session = STUBBED_TESTING_DATA / SESSION_SUBDIR
    assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path / "data_root"
    temporary_base_directory.mkdir(parents=True, exist_ok=True)
    session_name = source_session.name
    session_copy = temporary_base_directory / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
        assert os.path.isdir(output_directory)
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()

    common_kwargs = dict(
        base_dir=str(temporary_base_directory),
        selected_folders=[str(session_copy)],
    )

    step2(**common_kwargs, storenames_map=STORENAMES_MAP)
    step3(**common_kwargs)
    step4(**common_kwargs)
    step5(**common_kwargs, select_for_compute_psth="dff", select_for_transients="dff")

    output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = None
    for candidate in output_directories:
        if os.path.exists(os.path.join(candidate, "storesList.csv")):
            output_directory = candidate
            break
    assert output_directory is not None, f"No storesList.csv found in any output directory under {session_copy}"

    # PSTH outputs with dff naming
    psth_file_path = os.path.join(
        output_directory,
        f"{EXPECTED_TTL}_{EXPECTED_REGION}_dff_{EXPECTED_REGION}.h5",
    )
    baseline_uncorrected_file_path = os.path.join(
        output_directory,
        f"{EXPECTED_TTL}_{EXPECTED_REGION}_baselineUncorrected_dff_{EXPECTED_REGION}.h5",
    )
    peak_auc_h5_file_path = os.path.join(
        output_directory,
        f"peak_AUC_{EXPECTED_TTL}_{EXPECTED_REGION}_dff_{EXPECTED_REGION}.h5",
    )
    peak_auc_csv_file_path = os.path.join(
        output_directory,
        f"peak_AUC_{EXPECTED_TTL}_{EXPECTED_REGION}_dff_{EXPECTED_REGION}.csv",
    )

    assert os.path.exists(psth_file_path), f"Missing DFF PSTH HDF5: {psth_file_path}"
    assert os.path.exists(
        baseline_uncorrected_file_path
    ), f"Missing baseline-uncorrected DFF PSTH HDF5: {baseline_uncorrected_file_path}"
    assert os.path.exists(peak_auc_h5_file_path), f"Missing DFF peak/AUC HDF5: {peak_auc_h5_file_path}"
    assert os.path.exists(peak_auc_csv_file_path), f"Missing DFF peak/AUC CSV: {peak_auc_csv_file_path}"

    psth_dataframe = pd.read_hdf(psth_file_path, key="df")
    assert "timestamps" in psth_dataframe.columns, f"'timestamps' column missing in {psth_file_path}"
    assert "mean" in psth_dataframe.columns, f"'mean' column missing in {psth_file_path}"

    # Transient outputs with dff naming
    frequency_and_amplitude_h5_file_path = os.path.join(output_directory, f"freqAndAmp_dff_{EXPECTED_REGION}.h5")
    frequency_and_amplitude_csv_file_path = os.path.join(output_directory, f"freqAndAmp_dff_{EXPECTED_REGION}.csv")
    transients_occurrences_csv_file_path = os.path.join(
        output_directory, f"transientsOccurrences_dff_{EXPECTED_REGION}.csv"
    )

    assert os.path.exists(
        frequency_and_amplitude_h5_file_path
    ), f"Missing DFF freq/amp HDF5: {frequency_and_amplitude_h5_file_path}"
    assert os.path.exists(
        frequency_and_amplitude_csv_file_path
    ), f"Missing DFF freq/amp CSV: {frequency_and_amplitude_csv_file_path}"
    assert os.path.exists(
        transients_occurrences_csv_file_path
    ), f"Missing DFF transients occurrences CSV: {transients_occurrences_csv_file_path}"
