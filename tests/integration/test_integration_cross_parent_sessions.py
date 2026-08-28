import glob
import os
import shutil

import pandas as pd
import pytest

from guppy.testing.api import step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA


def _stage_session(*, session_subdir: str, destination_parent) -> str:
    """Copy a stubbed session into its own parent directory and clear prior outputs."""
    source_session = os.path.join(str(STUBBED_TESTING_DATA), session_subdir)
    assert os.path.isdir(source_session), f"Sample data not available at expected path: {source_session}"

    destination_parent.mkdir(parents=True, exist_ok=True)
    session_name = os.path.basename(source_session)
    session_copy = destination_parent / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
        assert os.path.isdir(output_directory)
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()
    return str(session_copy)


def _assert_psth_outputs(*, session_copy: str, recording_site: str, ttl: str) -> None:
    session_name = os.path.basename(session_copy)
    output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = output_directories[0]

    assert os.path.exists(os.path.join(output_directory, "storesList.csv")), "Missing storesList.csv"

    psth_file_path = os.path.join(output_directory, f"{ttl}_{recording_site}_z_score_{recording_site}.h5")
    assert os.path.exists(psth_file_path), f"Missing PSTH HDF5: {psth_file_path}"
    psth_dataframe = pd.read_hdf(psth_file_path, key="df")
    assert "timestamps" in psth_dataframe.columns
    assert "mean" in psth_dataframe.columns


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sessions_from_different_parent_directories(tmp_path):
    """
    Sessions kept in separate data roots are analyzed together in one pipeline run.

    A TDT session and a CSV session are staged under two different parent directories,
    mirroring a user whose acquisition formats live in separate folders. Steps 1-4 run
    across both selections at once and each session produces its own PSTH outputs.
    """
    tdt_session = _stage_session(
        session_subdir="tdt/Photo_63_207-181030-103332", destination_parent=tmp_path / "SampleData_Clean"
    )
    csv_session = _stage_session(session_subdir="csv/sample_data_csv_1", destination_parent=tmp_path / "SampleData_csv")
    assert os.path.dirname(tdt_session) != os.path.dirname(csv_session)

    base_dir = str(tmp_path)

    # Step 1 runs per session: each session's storesList.csv holds only its own channels.
    step1(
        base_dir=base_dir,
        selected_folders=[tdt_session],
        store_id_to_store_label={"Dv1A": "control_dms", "Dv2A": "signal_dms", "PrtN": "port_entries_dms"},
    )
    step1(
        base_dir=base_dir,
        selected_folders=[csv_session],
        store_id_to_store_label={
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
    )

    selected_folders = [tdt_session, csv_session]
    selected_runs = {folder: ["1"] for folder in selected_folders}
    step2(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step3(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step4(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)

    _assert_psth_outputs(session_copy=tdt_session, recording_site="dms", ttl="port_entries_dms")
    _assert_psth_outputs(session_copy=csv_session, recording_site="region", ttl="ttl")
