import glob
import os
import shutil

import pandas as pd
import pytest

from guppy.testing.api import step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_SUBDIR = "csv/sample_data_csv_1"
# Recording-site name deliberately contains an underscore — the core of issue #383. Before the
# fix, split("_")[-1] recovered "ms" instead of "d_ms" and the pipeline broke silently.
EXPECTED_RECORDING_SITE = "d_ms"
EXPECTED_TTL = "ttl"

# isosbestic_control=True uses a real control channel; False synthesizes one from the signal.
ISOSBESTIC_STORE_MAP = {
    "Sample_Control_Channel": f"control_{EXPECTED_RECORDING_SITE}",
    "Sample_Signal_Channel": f"signal_{EXPECTED_RECORDING_SITE}",
    "Sample_TTL": EXPECTED_TTL,
}
NO_ISOSBESTIC_STORE_MAP = {
    "Sample_Signal_Channel": f"signal_{EXPECTED_RECORDING_SITE}",
    "Sample_TTL": EXPECTED_TTL,
}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "isosbestic_control, store_id_to_store_label",
    [
        pytest.param(True, ISOSBESTIC_STORE_MAP, id="isosbestic"),
        pytest.param(False, NO_ISOSBESTIC_STORE_MAP, id="no_isosbestic"),
    ],
)
def test_underscore_recording_site_runs_end_to_end(tmp_path, isosbestic_control, store_id_to_store_label):
    """A recording-site name containing an underscore flows through the full pipeline (Steps 1-4).

    This is the direct regression test for issue #383: the recording-site key is threaded through
    timestamp correction, control fitting, dF/F, PSTH, peak/AUC and transients purely by
    constructing filenames from the known pair name, never by splitting on the last
    underscore — so an underscore in the name no longer corrupts the outputs.
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
    selected_runs = {str(session_copy): ["1"]}

    step1(**common_kwargs, store_id_to_store_label=store_id_to_store_label)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, isosbestic_control=isosbestic_control, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = None
    for candidate in output_directories:
        if os.path.exists(os.path.join(candidate, "storesList.csv")):
            output_directory = candidate
            break
    assert output_directory is not None, f"No storesList.csv found in any output directory under {session_copy}"

    # The full underscore recording site must appear intact in every derived artifact name.
    psth_file_path = os.path.join(
        output_directory, f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )
    peak_auc_h5_file_path = os.path.join(
        output_directory, f"peak_AUC_{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )
    frequency_and_amplitude_h5_file_path = os.path.join(
        output_directory, f"freqAndAmp_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )

    assert os.path.exists(psth_file_path), f"Missing PSTH HDF5: {psth_file_path}"
    assert os.path.exists(peak_auc_h5_file_path), f"Missing peak/AUC HDF5: {peak_auc_h5_file_path}"
    assert os.path.exists(
        frequency_and_amplitude_h5_file_path
    ), f"Missing freq/amp HDF5: {frequency_and_amplitude_h5_file_path}"

    psth_dataframe = pd.read_hdf(psth_file_path, key="df")
    assert "timestamps" in psth_dataframe.columns, f"'timestamps' column missing in {psth_file_path}"
    assert "mean" in psth_dataframe.columns, f"'mean' column missing in {psth_file_path}"
