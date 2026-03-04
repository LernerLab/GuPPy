import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing import compare_npm_session_files, compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

CONSISTENCY_CASES = [
    (
        "SampleData_Neurophotometrics/sampleData_NPM_2",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_2/sampleData_NPM_2_output_1",
        {
            "file0_chev6": "control_region",
            "file1_chev6": "signal_region",
        },
        {"npm_split_events": [True, True]},
        {},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_3",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_3/sampleData_NPM_3_output_1",
        {
            "file0_chev3": "control_region3",
            "file0_chod3": "signal_region3",
            "event3": "ttl_region3",
        },
        {
            "npm_timestamp_column_names": ["ComputerTimestamp", None],
            "npm_time_units": ["milliseconds", "seconds"],
            "npm_split_events": [False, True],
        },
        {},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_4",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_4/sampleData_NPM_4_output_1",
        {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "eventTrue": "ttl_true_region1",
        },
        {"npm_split_events": [True, True]},
        {},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_5",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_5/sampleData_NPM_5_output_1",
        {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "event0": "ttl_region1",
        },
        {"npm_split_events": None},
        {},
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, storenames_map, extra_kwargs, compare_kwargs",
    CONSISTENCY_CASES,
    ids=[
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    monkeypatch,
    session_subdir,
    standard_output_subdir,
    storenames_map,
    extra_kwargs,
    compare_kwargs,
):
    """
    Consistency test: run the full pipeline (Steps 2-5) and assert that the output
    is numerically identical (within tolerance) to the reference output from v1.3.0.

    In addition to the standard output folder comparison, this test also validates
    the intermediate CSV files that NPM Step 2 writes directly into the session
    folder (e.g. ``file0_chev1.csv``, ``event0.csv``).
    """
    src_session = TESTING_DATA / session_subdir
    assert src_session.is_dir(), f"Sample data not found: {src_session}"

    standard_output_dir = TESTING_DATA / standard_output_subdir
    assert standard_output_dir.is_dir(), f"Standard output not found: {standard_output_dir}"

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

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

    step2(**common_kwargs, storenames_map=storenames_map, **extra_kwargs)
    step3(**common_kwargs, **extra_kwargs)
    step4(**common_kwargs, **extra_kwargs)
    step5(**common_kwargs, **extra_kwargs)

    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert output_dirs, f"No output directory found under {session_copy}"
    actual_output_dir = output_dirs[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
        **compare_kwargs,
    )

    # NPM also writes intermediate CSVs into the session folder itself; compare
    # those against the top-level files in the parent of the standard output dir.
    standard_session_dir = (TESTING_DATA / standard_output_subdir).parent
    compare_npm_session_files(
        actual_session_dir=str(session_copy),
        expected_session_dir=str(standard_session_dir),
        **compare_kwargs,
    )
