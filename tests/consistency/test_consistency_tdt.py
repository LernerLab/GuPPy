import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

CONSISTENCY_CASES = [
    (
        "SampleData_Clean/Photo_63_207-181030-103332",
        "StandardOutputs_Clean/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1",
        {
            "Dv1A": "control_DMS",
            "Dv2A": "signal_DMS",
            "PrtN": "port_entries_dms",
        },
    ),
    (
        "SampleData_Clean/Photometry-161823",
        "StandardOutputs_Clean/Photometry-161823/Photometry-161823_output_1",
        {
            "405R": "control_region",
            "490R": "signal_region",
            "PAB/": "ttl",
        },
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, storenames_map",
    CONSISTENCY_CASES,
    ids=[
        "tdt_clean",
        "tdt_split_event",
    ],
)
@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    session_subdir,
    standard_output_subdir,
    storenames_map,
):
    """
    Consistency test: run the full pipeline (Steps 2-5) and assert that the output
    is numerically identical (within tolerance) to the reference output from v1.3.0.
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

    step2(**common_kwargs, storenames_map=storenames_map)
    step3(**common_kwargs)
    step4(**common_kwargs)
    step5(**common_kwargs)

    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert output_dirs, f"No output directory found under {session_copy}"
    actual_output_dir = output_dirs[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
    )
