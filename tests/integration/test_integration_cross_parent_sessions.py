import shutil
from pathlib import Path

import pandas as pd
import pytest

from guppy.testing.api import step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA


def _stage_session(*, session_subdir: str, destination_parent) -> str:
    """Copy a stubbed session into its own parent directory and clear prior outputs."""
    source_session = Path(str(STUBBED_TESTING_DATA)) / session_subdir
    assert Path(source_session).is_dir(), f"Sample data not available at expected path: {source_session}"

    destination_parent.mkdir(parents=True, exist_ok=True)
    session_name = Path(source_session).name
    session_copy = destination_parent / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in list(Path(session_copy).glob(f"{session_name}_output_*")):
        assert Path(output_directory).is_dir()
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()
    return str(session_copy)


def _assert_psth_outputs(*, session_copy: str, recording_site: str, ttl: str) -> None:
    session_name = Path(session_copy).name
    output_directories = sorted(list(Path(session_copy).glob(f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = output_directories[0]

    assert (Path(output_directory) / "storesList.csv").exists(), "Missing storesList.csv"

    psth_file_path = Path(output_directory) / (f"{ttl}_{recording_site}_z_score_{recording_site}.h5")
    assert Path(psth_file_path).exists(), f"Missing PSTH HDF5: {psth_file_path}"
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
    assert Path(tdt_session).parent != Path(csv_session).parent

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
