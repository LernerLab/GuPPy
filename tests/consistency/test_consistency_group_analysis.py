import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

SESSION_SUBDIRS = [
    "SampleData_with_artifacts/Photo_048_392-200728-121222",
    "SampleData_Clean/Photo_63_207-181030-103332",
]

STANDARD_OUTPUT_SUBDIR = "StandardOutputs_group_analysis/average"

STORENAMES_MAP = {
    "Dv1A": "control_region",
    "Dv2A": "signal_region",
    "LNRW": "rewarded_nose_pokes",
    "LNnR": "unrewarded_nose_pokes",
}


@pytest.mark.daily_only
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency_group_analysis(tmp_path):
    """
    Consistency test: run the full pipeline (Steps 2-5) on two TDT sessions and then
    perform group-level averaging, asserting that the averaged output is numerically
    identical (within tolerance) to the reference output.

    Source sessions are taken from SampleData_Clean and SampleData_with_artifacts
    (both available on Google Drive for CI), rather than from SampleData_combined
    which is not distributed.
    """
    src_sessions = [TESTING_DATA / s for s in SESSION_SUBDIRS]
    for src in src_sessions:
        assert src.is_dir(), f"Sample data not found: {src}"

    standard_output_dir = TESTING_DATA / STANDARD_OUTPUT_SUBDIR
    assert standard_output_dir.is_dir(), f"Standard output not found: {standard_output_dir}"

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

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=selected_folders,
    )

    step2(**common_kwargs, storenames_map=STORENAMES_MAP)
    step3(**common_kwargs)
    step4(**common_kwargs)
    step5(**common_kwargs)

    step5(**common_kwargs, average_for_group=True, group_folders=selected_folders)

    actual_output_dir = str(tmp_base / "average")
    assert os.path.isdir(actual_output_dir), f"No average directory found under {tmp_base}"

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
    )
