import glob
import os
import shutil

import pytest
from conftest import TESTING_DATA, event_ts_offset_for

from guppy.testing import compare_output_folders
from guppy.testing.api import step1, step2, step3, step4

SESSION_SUBDIR = "SampleData_csv/sample_data_csv_1"
STANDARD_OUTPUT_SUBDIR = "StandardOutputs_dff/sample_data_csv_1/sample_data_csv_1_output_1"

STORENAMES_MAP = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}


@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency_dff(tmp_path):
    """
    Consistency test: run the full pipeline (Steps 2-5) with ΔF/F selected for both
    PSTH computation and transient detection, and assert that the output is numerically
    identical (within tolerance) to the reference output.
    """
    src_session = TESTING_DATA / SESSION_SUBDIR
    assert src_session.is_dir(), f"Sample data not found: {src_session}"

    standard_output_dir = TESTING_DATA / STANDARD_OUTPUT_SUBDIR
    assert standard_output_dir.is_dir(), f"Standard output not found: {standard_output_dir}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = src_session.name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )

    selected_runs = {folder: ["1"] for folder in common_kwargs["selected_folders"]}
    step1(**common_kwargs, store_id_to_store_label=STORENAMES_MAP)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, control_fit_method="OLS", selected_runs=selected_runs)
    step4(**common_kwargs, select_for_compute_psth="dff", select_for_transients="dff", selected_runs=selected_runs)

    run_folders = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert run_folders, f"No output directory found under {session_copy}"
    actual_output_dir = run_folders[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
        event_ts_offset=event_ts_offset_for(tmp_base),
    )
