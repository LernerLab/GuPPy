import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.testing.api import (
    group_analysis,
    label_groups,
    locate_run_folder,
    step1,
    step2,
    step3,
    step4,
)
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bin_psth_trials_by_number_of_trials(tmp_path):
    """
    Verify that step4 with bin_psth_trials=2 and use_time_or_trials='# of trials'
    writes the bin columns to binned_<stem>.csv companions of the per-session PSTH and
    peak/AUC CSVs, and that running
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

    temporary_base = tmp_path / "input_root_folder"
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

    output_directory = locate_run_folder(session=str(session_copy))

    stem = f"{expected_ttl}_{expected_recording_site}_z_score_{expected_recording_site}"
    psth_file_path = Path(output_directory) / f"{stem}.csv"
    binned_psth_file_path = Path(output_directory) / f"binned_{stem}.csv"
    peak_auc_file_path = Path(output_directory) / f"peak_AUC_{stem}.csv"
    binned_peak_auc_file_path = Path(output_directory) / f"binned_peak_AUC_{stem}.csv"

    # Five TTLs binned two at a time give the bins [0, 2), [2, 4) and [4, 5).
    expected_bins = ["bin_(0-2)", "bin_(2-4)", "bin_(4-5)"]
    expected_binned_columns = [
        "timestamps",
        "bin_(0-2)",
        "bin_err_(0-2)",
        "bin_(2-4)",
        "bin_err_(2-4)",
        "bin_(4-5)",
        "bin_err_(4-5)",
    ]

    psth_dataframe = pd.read_csv(psth_file_path)
    assert [column for column in psth_dataframe.columns if column.startswith("bin_")] == []
    binned_psth_dataframe = pd.read_csv(binned_psth_file_path)
    assert list(binned_psth_dataframe.columns) == expected_binned_columns
    np.testing.assert_array_equal(binned_psth_dataframe["timestamps"], psth_dataframe["timestamps"])

    peak_auc = pd.read_csv(peak_auc_file_path)
    assert list(peak_auc["trial"])[-1] == "mean"
    assert len(peak_auc) == 6  # five trials plus the mean row
    binned_peak_auc = pd.read_csv(binned_peak_auc_file_path)
    assert list(binned_peak_auc.columns[:2]) == ["session", "bin"]
    assert list(binned_peak_auc["bin"]) == expected_bins
    assert list(binned_peak_auc["session"]) == [session_name] * 3

    # Run group averaging on the binned per-session output.  This exercises the
    # `if len(bin_columns) > 0:` branch inside psth_average.average_psth_for_group, which
    # concatenates and aggregates bin columns across the member runs.
    label_groups(
        member_run_folders=[locate_run_folder(session=folder) for folder in selected_folders],
        destination_directory=base_dir,
        group_name="binned",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "binned_group"])

    average_directory = Path(base_dir) / "binned_group"
    assert Path(average_directory).is_dir(), f"No group directory found under {base_dir}"

    assert Path(average_directory / f"{stem}.csv").exists(), f"Missing averaged PSTH CSV in {average_directory}"
    # Group averaging writes every bin mean before the bin errors.
    average_binned_psth_dataframe = pd.read_csv(average_directory / f"binned_{stem}.csv")
    assert list(average_binned_psth_dataframe.columns) == [
        "timestamps",
        "bin_(0-2)",
        "bin_(2-4)",
        "bin_(4-5)",
        "bin_err_(0-2)",
        "bin_err_(2-4)",
        "bin_err_(4-5)",
    ]
    average_binned_peak_auc = pd.read_csv(average_directory / f"binned_peak_AUC_{stem}.csv")
    assert list(average_binned_peak_auc["bin"]) == expected_bins
    assert list(average_binned_peak_auc["session"]) == [session_name] * 3
