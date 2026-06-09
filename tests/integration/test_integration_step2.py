import csv
import glob
import os
import shutil

import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2


@pytest.mark.parametrize(
    "session_subdir, storenames_map",
    [
        (
            "csv/sample_data_csv_1",
            {
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
                "Sample_TTL": "ttl",
            },
        ),
        (
            "doric/sample_doric_1",
            {
                "AIn-1 - Raw": "control_region",
                "AIn-2 - Raw": "signal_region",
                "DI--O-1": "ttl",
            },
        ),
        (
            "doric/sample_doric_2",
            {
                "AIn-1 - Dem (ref)": "control_region",
                "AIn-1 - Dem (da)": "signal_region",
                "DI/O-1": "ttl",
            },
        ),
        (
            "doric/sample_doric_3",
            {
                "CAM1_EXC1/ROI01": "control_region",
                "CAM1_EXC2/ROI01": "signal_region",
                "DigitalIO/CAM1": "ttl",
            },
        ),
        (
            "doric/sample_doric_4",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
        ),
        (
            "doric/sample_doric_5",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
        ),
        (
            "tdt/Photo_63_207-181030-103332",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
        ),
        (
            "tdt/Photo_048_392-200728-121222",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
        ),
        (
            # PAB/ is an epoc store that splits into one sub-event per marker value; those
            # sub-events are enumerated at step 2, so the map references them directly.
            "tdt/Photometry-161823",
            {
                "405R": "control_region",
                "490R": "signal_region",
                "PAB0": "ttl_0",
                "PAB16": "ttl_16",
                "PAB2064": "ttl_2064",
            },
        ),
        (
            "npm/sampleData_NPM_1",
            {
                "file0_chev1": "signal_region",
                "file0_chod1": "control_region",
            },
        ),
        (
            "npm/sampleData_NPM_2",
            {
                "file0_chev6": "control_region",
                "file1_chev6": "signal_region",
            },
        ),
        (
            "npm/sampleData_NPM_3",
            {
                "file0_chev3": "control_region3",
                "file0_chod3": "signal_region3",
                "event3": "ttl_region3",
            },
        ),
        (
            "npm/sampleData_NPM_4",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "eventTrue": "ttl_true_region1",
            },
        ),
        (
            "npm/sampleData_NPM_5",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "event0": "ttl_region1",
            },
        ),
        (
            "nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
            {
                "fiber_photometry_response_series_0": "control_region",
                "fiber_photometry_response_series_1": "signal_region",
                "events": "ttl",
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
        "tdt_split_event",
        "tdt_with_artifacts",
        "sample_npm_1",
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
        "nwb_mock",
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
    if session_subdir == "npm/sampleData_NPM_1":
        npm_timestamp_column_names = None
        npm_time_units = None
        npm_split_events = [False, True]
    elif session_subdir == "npm/sampleData_NPM_3":
        npm_timestamp_column_names = ["ComputerTimestamp", None]
        npm_time_units = ["milliseconds", "seconds"]
        npm_split_events = [False, True]
    else:
        npm_timestamp_column_names = None
        npm_time_units = None
        npm_split_events = [True, True]
    if session_subdir == "npm/sampleData_NPM_5":
        npm_split_events = None
    # Source sample data
    src_base_dir = str(STUBBED_TESTING_DATA)
    src_session = os.path.join(src_base_dir, session_subdir)
    assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"

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
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        storenames_map=storenames_map,
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

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

    # NPM now demultiplexes in memory: Step 2 must NOT write any intermediate CSVs into the
    # source session folder, and must persist the decomposition params next to storesList.csv.
    if session_subdir.startswith("npm/"):
        intermediates = (
            glob.glob(os.path.join(session_copy, "file*_chev*.csv"))
            + glob.glob(os.path.join(session_copy, "file*_chod*.csv"))
            + glob.glob(os.path.join(session_copy, "file*_chpr*.csv"))
            + glob.glob(os.path.join(session_copy, "event*.csv"))
        )
        assert intermediates == [], f"NPM Step 2 wrote intermediate CSVs into the source folder: {intermediates}"

        npm_params_fp = os.path.join(out_dir, ".npm_params.json")
        assert os.path.exists(npm_params_fp), f"Missing persisted NPM params at Step 2: {npm_params_fp}"
