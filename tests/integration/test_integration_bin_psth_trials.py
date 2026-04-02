import glob
import os
import shutil

import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2, step3, step4, step5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bin_psth_trials_by_number_of_trials(tmp_path):
    """
    Verify that step5 with bin_psth_trials=2 and use_time_or_trials='# of trials'
    produces bin columns in the per-session PSTH output HDF5, and that running
    group averaging (average_for_group=True) on that binned output also produces
    bin columns in the averaged PSTH — exercising the bin-averaging branch in
    psth_average.averageForGroup.
    """
    session_subdir = "csv/sample_data_csv_1"
    storenames_map = {
        "Sample_Control_Channel": "control_region",
        "Sample_Signal_Channel": "signal_region",
        "Sample_TTL": "ttl",
    }
    expected_region = "region"
    expected_ttl = "ttl"

    source_session = os.path.join(str(STUBBED_TESTING_DATA), session_subdir)
    assert os.path.isdir(source_session), f"Sample data not available at expected path: {source_session}"

    temporary_base = tmp_path / "data_root"
    temporary_base.mkdir(parents=True, exist_ok=True)
    session_name = os.path.basename(source_session)
    session_copy = temporary_base / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
        assert os.path.isdir(output_directory)
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()

    base_dir = str(temporary_base)
    selected_folders = [str(session_copy)]

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map=storenames_map,
        npm_timestamp_column_names=None,
        npm_time_units=None,
        npm_split_events=[True, True],
    )

    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=None,
        npm_time_units=None,
        npm_split_events=[True, True],
    )

    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=None,
        npm_time_units=None,
        npm_split_events=[True, True],
    )

    step5(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=None,
        npm_time_units=None,
        npm_split_events=[True, True],
        bin_psth_trials=2,
        use_time_or_trials="# of trials",
    )

    output_directories = sorted(glob.glob(os.path.join(str(session_copy), f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"
    output_directory = output_directories[0]

    psth_file_path = os.path.join(
        output_directory,
        f"{expected_ttl}_{expected_region}_z_score_{expected_region}.h5",
    )
    assert os.path.exists(psth_file_path), f"Missing PSTH HDF5: {psth_file_path}"

    psth_dataframe = pd.read_hdf(psth_file_path, key="df")
    bin_columns = [column for column in psth_dataframe.columns if column.startswith("bin_(")]
    assert bin_columns, (
        f"Expected at least one 'bin_(...)' column in PSTH output with bin_psth_trials=2, "
        f"but got columns: {list(psth_dataframe.columns)}"
    )

    # Run group averaging on the binned per-session output.  This exercises the
    # `if len(bins_cols) > 0:` branch inside psth_average.averageForGroup, which
    # concatenates and aggregates bin columns across sessions.
    step5(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=None,
        npm_time_units=None,
        npm_split_events=[True, True],
        average_for_group=True,
        group_folders=selected_folders,
        bin_psth_trials=2,
        use_time_or_trials="# of trials",
    )

    average_directory = os.path.join(base_dir, "average")
    assert os.path.isdir(average_directory), f"No average directory found under {base_dir}"

    average_psth_file_path = os.path.join(
        average_directory,
        f"{expected_ttl}_{expected_region}_z_score_{expected_region}.h5",
    )
    assert os.path.exists(average_psth_file_path), f"Missing averaged PSTH HDF5: {average_psth_file_path}"

    average_psth_dataframe = pd.read_hdf(average_psth_file_path, key="df")
    average_bin_columns = [column for column in average_psth_dataframe.columns if column.startswith("bin_(")]
    assert average_bin_columns, (
        f"Expected bin columns in averaged PSTH output, " f"but got columns: {list(average_psth_dataframe.columns)}"
    )
