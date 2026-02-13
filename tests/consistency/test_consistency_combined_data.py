import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

SESSION_SUBDIRS = [
    "SampleData_Clean/Photo_63_207-181030-103332",
    "SampleData_with_artifacts/Photo_048_392-200728-121222",
]

STANDARD_OUTPUT_SUBDIRS = [
    "StandardOutputs_combined/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1",
    "StandardOutputs_combined/Photo_048_392-200728-121222/Photo_048_392-200728-121222_output_1",
]

STORENAMES_MAP = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(tmp_path, monkeypatch):
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

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

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

    step2(base_dir=str(tmp_base), selected_folders=selected_folders, modality="tdt", storenames_map=STORENAMES_MAP)
    step3(base_dir=str(tmp_base), selected_folders=selected_folders, modality="tdt")
    step4(base_dir=str(tmp_base), selected_folders=selected_folders, modality="tdt", combine_data=True)
    # Step 5 is called on the second session only; it receives the full combined PSTH outputs.
    step5(base_dir=str(tmp_base), selected_folders=[selected_folders[1]], modality="tdt")

    for session_copy, standard_output_dir in zip(session_copies, standard_output_dirs):
        dest_name = session_copy.name
        output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
        assert output_dirs, f"No output directory found under {session_copy}"
        actual_output_dir = output_dirs[0]

        compare_output_folders(
            actual_dir=actual_output_dir,
            expected_dir=str(standard_output_dir),
        )
