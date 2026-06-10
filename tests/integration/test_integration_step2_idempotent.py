import csv
import glob
import os
import shutil

from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2


def test_step2_npm_idempotent(tmp_path):
    """
    Step 2 run twice in a row on an NPM dataset should succeed both times.

    NPM demultiplexes in memory and writes no intermediate CSVs into the source folder, so a
    second run sees the same raw files as the first and detect_acquisition_formats reports the
    same formats. This guards that re-running Step 2 stays idempotent (no leftover artifacts
    that could change modality detection or the storesList on the second run).
    """
    session_subdir = "npm/sampleData_NPM_4"
    storenames_map = {
        "file0_chev1": "control_region1",
        "file0_chod1": "signal_region1",
        "eventTrue": "ttl_true_region1",
    }

    src_base_dir = str(STUBBED_TESTING_DATA)
    src_session = os.path.join(src_base_dir, session_subdir)
    assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copy = tmp_base / os.path.basename(src_session)
    shutil.copytree(src_session, session_copy)

    step2_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        storenames_map=storenames_map,
        npm_split_events=[True, True],
    )

    # First run: baseline, should always succeed
    step2(**step2_kwargs)

    # Second run: should also succeed, but currently fails due to modality misdetection
    step2(**step2_kwargs)

    # Validate storesList.csv exists and matches the mapping after the second run
    basename = os.path.basename(session_copy)
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert output_dirs, f"No output directories found in {session_copy}"
    out_dir = next((d for d in output_dirs if os.path.exists(os.path.join(d, "storesList.csv"))), None)
    assert out_dir is not None, f"No storesList.csv found under {session_copy}"

    with open(os.path.join(out_dir, "storesList.csv"), newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[0] == list(storenames_map.keys())
    assert rows[1] == list(storenames_map.values())
