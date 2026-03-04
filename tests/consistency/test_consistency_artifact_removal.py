import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

# Artifact window coordinates (x/time values only; col 1 is y and not used by the analysis).
# Each array has shape (4, 2): four click events defining two artifact windows.
# These were determined interactively and are hardcoded here to make the test self-contained.
_COORDS_CONCATENATE = np.array(
    [
        [1.83404952e00, 0.0],
        [2.72244707e02, 0.0],
        [3.29535101e02, 0.0],
        [3.08405723e03, 0.0],
    ]
)

_COORDS_NAN = np.array(
    [
        [4.12566526e00, 0.0],
        [2.72244707e02, 0.0],
        [3.29535101e02, 0.0],
        [3.07718238e03, 0.0],
    ]
)

STORENAMES_MAP = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}

CONSISTENCY_CASES = [
    (
        "SampleData_with_artifacts/Photo_048_392-200728-121222",
        "StandardOutputs_with_artifacts/Photo_048_392-200728-121222_concatenate/Photo_048_392-200728-121222_output_1",
        "concatenate",
        _COORDS_CONCATENATE,
        {},
    ),
    (
        "SampleData_with_artifacts/Photo_048_392-200728-121222",
        "StandardOutputs_with_artifacts/Photo_048_392-200728-121222_nan/Photo_048_392-200728-121222_output_1",
        "replace with NaN",
        _COORDS_NAN,
        {},
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, artifact_removal_method, coords, compare_kwargs",
    CONSISTENCY_CASES,
    ids=[
        "artifacts_concatenate",
        "artifacts_nan",
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    session_subdir,
    standard_output_subdir,
    artifact_removal_method,
    coords,
    compare_kwargs,
):
    """
    Consistency test for artifact removal: run the full pipeline (Steps 2-5) with
    artifact removal enabled and assert that the output is numerically identical
    (within tolerance) to the reference output.

    Artifact coordinates are hardcoded in this file (read from the reference .npy
    files) and written to the output directory between Step 3 and Step 4 to bypass
    the interactive artifact-selection UI.
    """
    src_session = TESTING_DATA / session_subdir
    assert src_session.is_dir(), f"Sample data not found: {src_session}"

    standard_output_dir = TESTING_DATA / standard_output_subdir
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

    step2(**common_kwargs, storenames_map=STORENAMES_MAP)
    step3(**common_kwargs)
    step4(
        **common_kwargs,
        remove_artifacts=True,
        artifact_removal_method=artifact_removal_method,
        artifact_coords={"dms": coords},
    )
    step5(**common_kwargs)

    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert output_dirs, f"No output directory found under {session_copy}"
    actual_output_dir = output_dirs[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
        **compare_kwargs,
    )
