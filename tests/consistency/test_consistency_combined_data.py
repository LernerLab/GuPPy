import glob
import os
import shutil

import pytest
from conftest import TESTING_DATA, event_ts_offset_for

from guppy.testing import compare_output_folders
from guppy.testing.api import step1, step2, step3, step4

SESSION_SUBDIRS = [
    "SampleData_Clean/Photo_63_207-181030-103332",
    "SampleData_with_artifacts/Photo_048_392-200728-121222",
]

STANDARD_OUTPUT_SUBDIRS = [
    "StandardOutputs_combined/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1",
    "StandardOutputs_combined/Photo_048_392-200728-121222/Photo_048_392-200728-121222_output_1",
]

STORE_ID_TO_STORE_LABEL = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}


@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(tmp_path):
    """
    Consistency test for the combine_data pipeline: run Steps 2–5 with combine_data=True
    across two sessions and assert that the outputs are numerically identical (within
    tolerance) to the reference outputs.

    The second session (Photo_048_392) receives the full PSTH outputs; the first session
    (Photo_63_207) receives only preprocessing outputs.
    """
    src_sessions = [TESTING_DATA / s for s in SESSION_SUBDIRS]
    for src in src_sessions:
        assert src.is_dir(), f"Sample data not found: {src}"

    standard_output_dirs = [TESTING_DATA / s for s in STANDARD_OUTPUT_SUBDIRS]
    for ref in standard_output_dirs:
        assert ref.is_dir(), f"Standard output not found: {ref}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copies = []
    for src in src_sessions:
        dest_name = src.name
        session_copy = tmp_base / dest_name
        shutil.copytree(src, session_copy)
        for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
            shutil.rmtree(d)
        params_fp = session_copy / "GuPPyParamtersUsed.json"
        if params_fp.exists():
            params_fp.unlink()
        session_copies.append(session_copy)

    selected_folders = [str(s) for s in session_copies]

    selected_runs = {folder: ["1"] for folder in selected_folders}
    step1(base_dir=str(tmp_base), selected_folders=selected_folders, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(base_dir=str(tmp_base), selected_folders=selected_folders, selected_runs=selected_runs)
    step3(
        base_dir=str(tmp_base),
        selected_folders=selected_folders,
        combine_data=True,
        control_fit_method="OLS",
        selected_runs=selected_runs,
    )
    # Step 4 is called on the second session only; it receives the full combined PSTH outputs.
    step4(
        base_dir=str(tmp_base),
        selected_folders=[selected_folders[1]],
        selected_runs={selected_folders[1]: ["1"]},
    )

    for session_copy, standard_output_dir in zip(session_copies, standard_output_dirs):
        dest_name = session_copy.name
        run_folders = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
        assert run_folders, f"No output directory found under {session_copy}"
        actual_output_dir = run_folders[0]

        compare_output_folders(
            actual_dir=actual_output_dir,
            expected_dir=str(standard_output_dir),
            event_ts_offset=event_ts_offset_for(tmp_base),
        )
