import glob
import os
import shutil

import pytest
from conftest import TESTING_DATA

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

CONSISTENCY_CASES = [
    (
        "SampleData_csv/sample_data_csv_1",
        "StandardOutputs_csv/sample_data_csv_1/sample_data_csv_1_output_1",
        {
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, storenames_map",
    CONSISTENCY_CASES,
    ids=[
        "csv_generic",
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
