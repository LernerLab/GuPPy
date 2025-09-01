import os
import csv
import glob
import shutil

import h5py
import pytest
from pathlib import Path

from guppy.testing.api import step2, step3


@pytest.fixture(scope="function")
def storenames_map():
    return {
        "Sample_Control_Channel": "control_region",
        "Sample_Signal_Channel": "signal_region",
        "Sample_TTL": "ttl",
    }

@pytest.mark.parametrize(
    "session_subdir, storenames_map",
    [
        (
            "SampleData_csv",
            {
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
                "Sample_TTL": "ttl",
            },
        ),
        (
            "SampleData_Doric",
            {
                "AIn-1 - Dem (ref)": "control_region",
                "AIn-1 - Dem (da)": "signal_region",
                "DI/O-1": "ttl",
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
        (
            "SampleData_Neurophotometrics/1442",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "eventTrue": "ttl_true_region1",
            },
        ),
    ],
    ids=[
        "csv_generic",
        "doric_csv",
        "tdt_clean",
        "tdt_with_artifacts",
        "neurophotometrics_csv",
    ],
)
def test_step3(tmp_path, storenames_map, session_subdir):
    """
    Full integration test for Step 3 (Read Raw Data) using real CSV sample data,
    isolated to a temporary workspace to avoid mutating shared sample data.

    Behavior:
    - Copies the SampleData_csv session under GDriveSampleData into tmp_path.
    - Cleans any copied artifacts (*_output_* dirs, GuPPyParamtersUsed.json).
    - Derives a minimal storenames_map from the copied session and runs Step 2
      to create storesList.csv in the temp copy.
    - Runs Step 3 headlessly and verifies per-storename HDF5 outputs exist in
      the temp copy (never touching the original sample path).
    """
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

    # Remove any copied artifacts in the temp session
    # Use a specific glob that uniquely matches this session's output directory(ies)
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    # Step 2: create storesList.csv in the temp copy
    step2(base_dir=str(tmp_base), selected_folders=[str(session_copy)], storenames_map=storenames_map)

    # Step 3: read raw data in the temp copy
    step3(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

    # Validate outputs exist in the temp copy
    basename = os.path.basename(session_copy)
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert output_dirs, f"No output directories found in {session_copy}"
    out_dir = None
    for d in output_dirs:
        if os.path.exists(os.path.join(d, "storesList.csv")):
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"
    stores_fp = os.path.join(out_dir, "storesList.csv")

    # Assert: storesList.csv structure is 2xN
    with open(stores_fp, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 2, "storesList.csv should be 2 rows (storenames, names_for_storenames)"
    storenames = rows[0]
    assert storenames, "Expected at least one storename in storesList.csv"

    # For each storename, ensure an HDF5 was produced; minimally check timestamps dataset exists.
    for storename in storenames:
        safe = storename.replace("\\", "_").replace("/", "_")
        h5_path = os.path.join(out_dir, f"{safe}.hdf5")
        assert os.path.exists(h5_path), f"Missing HDF5 for storename {storename!r} at {h5_path}"

        with h5py.File(h5_path, "r") as f:
            assert "timestamps" in f, "Expected 'timestamps' dataset in HDF5"
