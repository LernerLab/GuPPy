import csv
import json
import shutil
from pathlib import Path

import pytest

from guppy.testing.api import step1
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.parametrize(
    "session_subdir, store_id_to_store_label",
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
            # sub-events are enumerated at step 1, so the map references them directly.
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
def test_step1(tmp_path, session_subdir, store_id_to_store_label):
    """
    Step 1 integration test (Label Stores) using real sample data, isolated to a temporary workspace.
    For each dataset:
      - Copies the session into a temp workspace
      - Cleans any copied *_output_* artifacts (using a specific glob to avoid non-dirs)
      - Calls step1 headlessly with an explicit, deterministic store_id_to_store_label
      - Asserts storesList.csv exists and exactly matches the provided mapping (2xN)
    """
    npm_timestamp_column_name = None
    npm_time_unit = None
    npm_split_events = None
    if session_subdir in ("npm/sampleData_NPM_1", "npm/sampleData_NPM_4"):
        # file1 is the only file with multiple event TTLs, so it is the only one that can split.
        npm_split_events = [False, True]
    elif session_subdir == "npm/sampleData_NPM_3":
        npm_timestamp_column_name = "ComputerTimestamp"
        npm_time_unit = "milliseconds"
        npm_split_events = [False, True]
    elif session_subdir == "npm/sampleData_NPM_5":
        # Header-less session: its clock is in milliseconds, which only the user can state.
        npm_time_unit = "milliseconds"
    # Source sample data
    src_base_dir = str(STUBBED_TESTING_DATA)
    src_session = Path(src_base_dir) / session_subdir
    assert Path(src_session).is_dir(), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = Path(src_session).name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session; match only this session's output directory(ies)
    for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
        assert Path(d).is_dir(), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)

    # Remove any copied GuPPyParamtersUsed.json to ensure a fresh run
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    # Run Step 1 headlessly using the explicit mapping
    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label=store_id_to_store_label,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
    )

    # Validate storesList.csv exists and matches the mapping exactly (order-preserved)
    basename = Path(session_copy).name
    run_folders = sorted(list(Path(session_copy).glob(f"{basename}_output_*")))
    assert run_folders, f"No output directories found in {session_copy}"

    out_dir = None
    for d in run_folders:
        if (Path(d) / "storesList.csv").exists():
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"

    out_fp = Path(out_dir) / "storesList.csv"
    assert Path(out_fp).exists(), f"Missing storesList.csv: {out_fp}"

    with Path(out_fp).open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 rows (store_ids, store_labels), got {len(rows)}"
    assert rows[0] == list(store_id_to_store_label.keys()), "Row 0 (store_ids) mismatch"
    assert rows[1] == list(store_id_to_store_label.values()), "Row 1 (store_labels) mismatch"

    # NPM now demultiplexes in memory: Step 1 must NOT write any intermediate CSVs into the
    # source session folder, and must persist the decomposition params next to storesList.csv.
    if session_subdir.startswith("npm/"):
        intermediates = (
            list(Path(session_copy).glob("file*_chev*.csv"))
            + list(Path(session_copy).glob("file*_chod*.csv"))
            + list(Path(session_copy).glob("file*_chpr*.csv"))
            + list(Path(session_copy).glob("event*.csv"))
        )
        assert intermediates == [], f"NPM Step 1 wrote intermediate CSVs into the source folder: {intermediates}"

        npm_params_fp = Path(out_dir) / ".npm_params.json"
        assert Path(npm_params_fp).exists(), f"Missing persisted NPM params at Step 1: {npm_params_fp}"

        # The persisted unit is the only record of the clock a run was read with, so it must
        # state the unit that was actually applied — not a default the extractor overrode
        # (issue #411). "seconds" is what an unset unit resolves to.
        with Path(npm_params_fp).open() as npm_params_file:
            npm_params = json.load(npm_params_file)
        assert npm_params["npm_time_unit"] == (npm_time_unit or "seconds")
        # Sessions offering more than one timestamp column persist the confirmed selection
        # (the form default when the caller supplied none); single-column sessions persist None.
        if session_subdir == "npm/sampleData_NPM_1":
            # The form default, not the setting this session should be analysed with: its stimuli
            # file rides ComputerTimestamp (see stubbed_testing_data/README.md). Step 2 is where
            # that mismatch is caught, so step 1 still persists whatever was confirmed here.
            assert npm_params["npm_timestamp_column_name"] == "SystemTimestamp"
        else:
            assert npm_params["npm_timestamp_column_name"] == npm_timestamp_column_name
