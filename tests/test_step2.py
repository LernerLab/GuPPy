import csv
import os
import glob
import shutil

import pytest

from guppy.testing.api import step2


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
def test_step2(tmp_path, session_subdir, storenames_map):
    """
    Step 2 integration test (Save Storenames) using real sample data, isolated to a temporary workspace.
    For each dataset:
      - Copies the session into a temp workspace
      - Cleans any copied *_output_* artifacts (using a specific glob to avoid non-dirs)
      - Calls step2 headlessly with an explicit, deterministic storenames_map
      - Asserts storesList.csv exists and exactly matches the provided mapping (2xN)
    """
    # Source sample data
    src_base_dir = "/Users/pauladkisson/Documents/CatalystNeuro/Guppy/GDriveSampleData"
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
    step2(base_dir=str(tmp_base), selected_folders=[str(session_copy)], storenames_map=storenames_map)

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
