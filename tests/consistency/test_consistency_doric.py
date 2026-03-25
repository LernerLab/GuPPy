import glob
import os
import shutil

import pytest
from conftest import TESTING_DATA

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

CONSISTENCY_CASES = [
    (
        "SampleData_Doric/sample_doric_1",
        "StandardOutputs_Doric/sample_data_doric_1/sample_doric_1_output_1",
        {
            "AIn-1 - Raw": "control_region",
            "AIn-2 - Raw": "signal_region",
            "DI--O-1": "ttl",
        },
        # scipy 1.5→1.17 and numpy 1.18→2.x cause up to ~1% drift in filtfilt/polyfit;
        # widened tolerance accommodates known dependency changes without masking real regressions.
        {"rtol": 1e-2, "atol": 2e-3},
    ),
    (
        "SampleData_Doric/sample_doric_2",
        "StandardOutputs_Doric/sample_data_doric_2/sample_doric_2_output_1",
        {
            "AIn-1 - Dem (ref)": "control_region",
            "AIn-1 - Dem (da)": "signal_region",
            "DI/O-1": "ttl",
        },
        {},
    ),
    (
        "SampleData_Doric/sample_doric_3",
        "StandardOutputs_Doric/sample_data_doric_3/sample_doric_3_output_1",
        {
            "CAM1_EXC1/ROI01": "control_region",
            "CAM1_EXC2/ROI01": "signal_region",
            "DigitalIO/CAM1": "ttl",
        },
        {},
    ),
    (
        "SampleData_Doric/sample_doric_4",
        "StandardOutputs_Doric/sample_data_doric_4/sample_doric_4_output_1",
        {
            "Series0001/AIN01xAOUT01-LockIn": "control_region",
            "Series0001/AIN01xAOUT02-LockIn": "signal_region",
        },
        {},
    ),
    (
        "SampleData_Doric/sample_doric_5",
        "StandardOutputs_Doric/sample_data_doric_5/sample_doric_5_output_1",
        {
            "Series0001/AIN01xAOUT01-LockIn": "control_region",
            "Series0001/AIN01xAOUT02-LockIn": "signal_region",
        },
        {},
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, storenames_map, compare_kwargs",
    CONSISTENCY_CASES,
    ids=[
        "sample_doric_1",
        "sample_doric_2",
        "sample_doric_3",
        "sample_doric_4",
        "sample_doric_5",
    ],
)
@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    session_subdir,
    standard_output_subdir,
    storenames_map,
    compare_kwargs,
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
        **compare_kwargs,
    )
