import shutil
from pathlib import Path

import pandas as pd
import pytest

from guppy.testing.api import group_analysis, label_groups, step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bin_psth_trials_by_number_of_trials(tmp_path):
    """
    Verify that step4 with bin_psth_trials=2 and use_time_or_trials='# of trials'
    produces bin columns in the per-session PSTH output HDF5, and that running
    the Group Analysis step on that binned output also produces
    bin columns in the averaged PSTH — exercising the bin-averaging branch in
    psth_average.averageForGroup.
    """
    session_subdir = "csv/sample_data_csv_1"
    store_id_to_store_label = {
        "Sample_Control_Channel": "control_region",
        "Sample_Signal_Channel": "signal_region",
        "Sample_TTL": "ttl",
    }
    expected_recording_site = "region"
    expected_ttl = "ttl"

    source_session = Path(str(STUBBED_TESTING_DATA)) / session_subdir
    assert Path(source_session).is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base = tmp_path / "data_root"
    temporary_base.mkdir(parents=True, exist_ok=True)
    session_name = Path(source_session).name
    session_copy = temporary_base / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in list(Path(session_copy).glob(f"{session_name}_output_*")):
        assert Path(output_directory).is_dir()
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()

    base_dir = str(temporary_base)
    selected_folders = [str(session_copy)]
    selected_runs = {folder: ["1"] for folder in selected_folders}

    step1(
        base_dir=base_dir,
        selected_folders=selected_folders,
        store_id_to_store_label=store_id_to_store_label,
    )

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=None,
        npm_time_unit=None,
        npm_split_events=[True, True],
        selected_runs=selected_runs,
    )

    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=None,
        npm_time_unit=None,
        npm_split_events=[True, True],
        selected_runs=selected_runs,
    )

    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=None,
        npm_time_unit=None,
        npm_split_events=[True, True],
        bin_psth_trials=2,
        use_time_or_trials="# of trials",
        selected_runs=selected_runs,
    )

    output_directories = sorted(Path(session_copy).glob(f"{session_name}_output_*"))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = output_directories[0]

    psth_file_path = Path(output_directory) / (
        f"{expected_ttl}_{expected_recording_site}_z_score_{expected_recording_site}.h5"
    )
    assert Path(psth_file_path).exists(), f"Missing PSTH HDF5: {psth_file_path}"

    psth_dataframe = pd.read_hdf(psth_file_path, key="df")
    bin_columns = [column for column in psth_dataframe.columns if column.startswith("bin_(")]
    assert bin_columns, (
        f"Expected at least one 'bin_(...)' column in PSTH output with bin_psth_trials=2, "
        f"but got columns: {list(psth_dataframe.columns)}"
    )

    # Run group averaging on the binned per-session output.  This exercises the
    # `if len(bin_columns) > 0:` branch inside psth_average.average_psth_for_group, which
    # concatenates and aggregates bin columns across the member runs.
    label_groups(
        member_run_folders=[Path(folder) / (f"{Path(folder).name}_output_1") for folder in selected_folders],
        destination_directory=base_dir,
        group_name="binned",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "binned_group"])

    average_directory = Path(base_dir) / "binned_group"
    assert Path(average_directory).is_dir(), f"No group directory found under {base_dir}"

    average_psth_file_path = Path(average_directory) / (
        f"{expected_ttl}_{expected_recording_site}_z_score_{expected_recording_site}.h5"
    )
    assert Path(average_psth_file_path).exists(), f"Missing averaged PSTH HDF5: {average_psth_file_path}"

    average_psth_dataframe = pd.read_hdf(average_psth_file_path, key="df")
    average_bin_columns = [column for column in average_psth_dataframe.columns if column.startswith("bin_(")]
    assert average_bin_columns, (
        f"Expected bin columns in averaged PSTH output, " f"but got columns: {list(average_psth_dataframe.columns)}"
    )
