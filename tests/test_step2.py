import csv
import glob
import os
import shutil
from pathlib import Path

import pytest

from guppy.testing.api import step2


@pytest.mark.parametrize(
    "session_subdir, storenames_map",
    [
        (
            "SampleData_csv/sample_data_csv_1",
            {
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
                "Sample_TTL": "ttl",
            },
        ),
        (
            "SampleData_Doric/sample_doric_1",
            {
                "AIn-1 - Raw": "control_region",
                "AIn-2 - Raw": "signal_region",
                "DI--O-1": "ttl",
            },
        ),
        (
            "SampleData_Doric/sample_doric_2",
            {
                "AIn-1 - Dem (ref)": "control_region",
                "AIn-1 - Dem (da)": "signal_region",
                "DI/O-1": "ttl",
            },
        ),
        (
            "SampleData_Doric/sample_doric_3",
            {
                "CAM1_EXC1/ROI01": "control_region",
                "CAM1_EXC2/ROI01": "signal_region",
                "DigitalIO/CAM1": "ttl",
            },
        ),
        (
            "SampleData_Doric/sample_doric_4",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
        ),
        (
            "SampleData_Doric/sample_doric_5",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
        ),
        (
            "SampleData_Clean/Photo_63_207-181030-103332",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
        ),
        (
            "SampleData_with_artifacts/Photo_048_392-200728-121222",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
        ),
        # TODO: Add sampleData_NPM_1 after fixing Doric vs. NPM determination bug.
        (
            "SampleData_Neurophotometrics/sampleData_NPM_2",
            {
                "file0_chev6": "control_region",
                "file1_chev6": "signal_region",
            },
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_3",
            {
                "file0_chev3": "control_region3",
                "file0_chod3": "signal_region3",
                "event3": "ttl_region3",
            },
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_4",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "eventTrue": "ttl_true_region1",
            },
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_5",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "event0": "ttl_region1",
            },
        ),
    ],
    ids=[
        "csv_generic",
        "sample_doric_1",
        "sample_doric_2",
        "sample_doric_3",
        "sample_doric_4",
        "sample_doric_5",
        "tdt_clean",
        "tdt_with_artifacts",
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
    ],
)
def test_step2(tmp_path, session_subdir, storenames_map):
    """
    Step 2 integration test (Save Storenames) using real sample data, isolated to a temporary workspace.
    For each dataset:
      - Copies the session into a temp workspace
      - Cleans any copied *_output_* artifacts (using a specific glob to avoid non-dirs)
      - Calls step2 headlessly with an explicit, deterministic storenames_map
      - Asserts storesList.csv exists and exactly matches the provided mapping (2xN)
    """
    if session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_3":
        npm_timestamp_column_name = "ComputerTimestamp"
        npm_time_unit = "milliseconds"
    else:
        npm_timestamp_column_name = None
        npm_time_unit = None
    if session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_5":
        npm_split_events = False
    else:
        npm_split_events = True

    # Source sample data
    src_base_dir = str(Path(".") / "testing_data")
    src_session = os.path.join(src_base_dir, session_subdir)
    if not os.path.isdir(src_session):
        pytest.skip(f"Sample data not available at expected path: {src_session}")

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = os.path.basename(src_session)
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session; match only this session's output directory(ies)
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)

    # Remove any copied GuPPyParamtersUsed.json to ensure a fresh run
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    # Run Step 2 headlessly using the explicit mapping
    step2(base_dir=str(tmp_base), selected_folders=[str(session_copy)], storenames_map=storenames_map, npm_timestamp_column_name=npm_timestamp_column_name, npm_time_unit=npm_time_unit, npm_split_events=npm_split_events)

    # Validate storesList.csv exists and matches the mapping exactly (order-preserved)
    basename = os.path.basename(session_copy)
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert output_dirs, f"No output directories found in {session_copy}"

    out_dir = None
    for d in output_dirs:
        if os.path.exists(os.path.join(d, "storesList.csv")):
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"

    out_fp = os.path.join(out_dir, "storesList.csv")
    assert os.path.exists(out_fp), f"Missing storesList.csv: {out_fp}"

    with open(out_fp, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 rows (storenames, names_for_storenames), got {len(rows)}"
    assert rows[0] == list(storenames_map.keys()), "Row 0 (storenames) mismatch"
    assert rows[1] == list(storenames_map.values()), "Row 1 (names_for_storenames) mismatch"

    # Additional NPM assertions: ensure Step 2 created the expected CSV files for Neurophotometrics
    if session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_1":
        expected_files = [
            "bl72bl82_12feb2024_fp.csv",
            "bl72bl82_12feb2024_stimuli.csv",
            "eventAfVf.csv",
            "eventAfVm.csv",
            "eventAfVn.csv",
            "eventAfVu.csv",
            "eventAmVf.csv",
            "eventAmVm.csv",
            "eventAmVn.csv",
            "eventAmVu.csv",
            "eventAnVf.csv",
            "eventAnVm.csv",
            "eventAnVu.csv",
            "eventAuVf.csv",
            "eventAuVm.csv",
            "eventAuVn.csv",
            "eventAuVu.csv",
            "eventblankvideo.csv",
            "eventpinknoise.csv",
            "eventtone.csv",
            "eventwhitenoise.csv",
            "file0_chev1.csv",
            "file0_chod1.csv",
        ]
        for rel in expected_files:
            fp = os.path.join(session_copy, rel)
            assert os.path.exists(fp), f"Missing expected NPM file at Step 2: {fp}"
    elif session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_2":
        expected_files = [
            "file0_chev1.csv",
            "file0_chev2.csv",
            "file0_chev3.csv",
            "file0_chev4.csv",
            "file0_chev5.csv",
            "file0_chev6.csv",
            "file0_chev7.csv",
            "file1_chev1.csv",
            "file1_chev2.csv",
            "file1_chev3.csv",
            "file1_chev4.csv",
            "file1_chev5.csv",
            "file1_chev6.csv",
            "file1_chev7.csv",
            "FiberData415.csv",
            "FiberData470.csv",
        ]
        for rel in expected_files:
            fp = os.path.join(session_copy, rel)
            assert os.path.exists(fp), f"Missing expected NPM file at Step 2: {fp}"
    elif session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_3":
        expected_files = [
            "event1.csv",
            "event3.csv",
            "file0_chev1.csv",
            "file0_chev2.csv",
            "file0_chev3.csv",
            "file0_chod1.csv",
            "file0_chod2.csv",
            "file0_chod3.csv",
        ]
        for rel in expected_files:
            fp = os.path.join(session_copy, rel)
            assert os.path.exists(fp), f"Missing expected NPM file at Step 2: {fp}"
    elif session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_4":
        expected_files = [
            "eventTrue.csv",
            "eventFalse.csv",
            "file0_chev1.csv",
            "file0_chev2.csv",
            "file0_chev3.csv",
            "file0_chod1.csv",
            "file0_chod2.csv",
            "file0_chod3.csv",
        ]
        for rel in expected_files:
            fp = os.path.join(session_copy, rel)
            assert os.path.exists(fp), f"Missing expected NPM file at Step 2: {fp}"
    elif session_subdir == "SampleData_Neurophotometrics/sampleData_NPM_5":
        expected_files = [
            "event0.csv",
            "file0_chev1.csv",
            "file0_chev2.csv",
            "file0_chev3.csv",
            "file0_chod1.csv",
            "file0_chod2.csv",
            "file0_chod3.csv",
            "PagCeAVgatFear_1512_1.csv",
            "PagCeAVgatFear_1512_ts0.csv",
        ]
        for rel in expected_files:
            fp = os.path.join(session_copy, rel)
            assert os.path.exists(fp), f"Missing expected NPM file at Step 2: {fp}"
