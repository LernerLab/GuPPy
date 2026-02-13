import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step2, step3, step4, step5

TESTING_DATA = Path(".") / "testing_data"

# Each entry: (session_subdir, standard_output_subdir, storenames_map, modality, extra_kwargs)
# standard_output_subdir is relative to testing_data/
# extra_kwargs: forwarded to step2–step5 (used for NPM-specific parameters)
CONSISTENCY_CASES = [
    (
        "SampleData_Clean/Photo_63_207-181030-103332",
        "StandardOutputs_Clean/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1",
        {
            "Dv1A": "control_DMS",
            "Dv2A": "signal_DMS",
            "PrtN": "port_entries_dms",
        },
        "tdt",
        {},
    ),
    (
        "SampleData_Clean/Photometry-161823",
        "StandardOutputs_Clean/Photometry-161823/Photometry-161823_output_1",
        {
            "405R": "control_region",
            "490R": "signal_region",
            "PAB/": "ttl",
        },
        "tdt",
        {},
    ),
    (
        "SampleData_csv/sample_data_csv_1",
        "StandardOutputs_csv/sample_data_csv_1/sample_data_csv_1_output_1",
        {
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
        "csv",
        {},
    ),
    (
        "SampleData_Doric/sample_doric_1",
        "StandardOutputs_Doric/sample_data_doric_1/sample_doric_1_output_1",
        {
            "AIn-1 - Raw": "control_region",
            "AIn-2 - Raw": "signal_region",
            "DI--O-1": "ttl",
        },
        "doric",
        {},
    ),
    (
        "SampleData_Doric/sample_doric_2",
        "StandardOutputs_Doric/sample_data_doric_2/sample_doric_2_output_1",
        {
            "AIn-1 - Dem (ref)": "control_region",
            "AIn-1 - Dem (da)": "signal_region",
            "DI/O-1": "ttl",
        },
        "doric",
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
        "doric",
        {},
    ),
    (
        "SampleData_Doric/sample_doric_4",
        "StandardOutputs_Doric/sample_data_doric_4/sample_doric_4_output_1",
        {
            "Series0001/AIN01xAOUT01-LockIn": "control_region",
            "Series0001/AIN01xAOUT02-LockIn": "signal_region",
        },
        "doric",
        {},
    ),
    (
        "SampleData_Doric/sample_doric_5",
        "StandardOutputs_Doric/sample_data_doric_5/sample_doric_5_output_1",
        {
            "Series0001/AIN01xAOUT01-LockIn": "control_region",
            "Series0001/AIN01xAOUT02-LockIn": "signal_region",
        },
        "doric",
        {},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_2",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_2/sampleData_NPM_2_output_1",
        {
            "file0_chev6": "control_region",
            "file1_chev6": "signal_region",
        },
        "npm",
        {"npm_split_events": [True, True]},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_3",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_3/sampleData_NPM_3_output_1",
        {
            "file0_chev3": "control_region3",
            "file0_chod3": "signal_region3",
            "event3": "ttl_region3",
        },
        "npm",
        {
            "npm_timestamp_column_names": ["ComputerTimestamp", None],
            "npm_time_units": ["milliseconds", "seconds"],
            "npm_split_events": [False, True],
        },
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_4",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_4/sampleData_NPM_4_output_1",
        {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "eventTrue": "ttl_true_region1",
        },
        "npm",
        {"npm_split_events": [True, True]},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_5",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_5/sampleData_NPM_5_output_1",
        {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "event0": "ttl_region1",
        },
        "npm",
        {"npm_split_events": None},
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, storenames_map, modality, extra_kwargs",
    CONSISTENCY_CASES,
    ids=[
        "tdt_clean",
        "tdt_split_event",
        "csv_generic",
        "sample_doric_1",
        "sample_doric_2",
        "sample_doric_3",
        "sample_doric_4",
        "sample_doric_5",
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path, monkeypatch, session_subdir, standard_output_subdir, storenames_map, modality, extra_kwargs
):
    """
    Consistency test: run the full pipeline (Steps 2-5) and assert that the output
    is numerically identical (within tolerance) to the reference output from v1.3.0.

    The reference outputs live in ``testing_data/StandardOutputs_*/`` and were
    generated by running GuPPy v1.3.0 against the same sample data. This test
    catches silent numerical regressions introduced by refactoring or dependency
    updates, complementing the structural integration tests in test_step*.py.
    """
    src_session = TESTING_DATA / session_subdir
    if not src_session.is_dir():
        pytest.skip(f"Sample data not found: {src_session}")

    standard_output_dir = TESTING_DATA / standard_output_subdir
    if not standard_output_dir.is_dir():
        pytest.skip(f"Standard output not found: {standard_output_dir}")

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = src_session.name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any pre-copied output directories and parameter files
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        modality=modality,
    )

    step2(**common_kwargs, storenames_map=storenames_map, **extra_kwargs)
    step3(**common_kwargs, **extra_kwargs)
    step4(**common_kwargs, **extra_kwargs)
    step5(**common_kwargs, **extra_kwargs)

    # Find the output directory produced by the pipeline
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert output_dirs, f"No output directory found under {session_copy}"
    actual_output_dir = output_dirs[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
    )
