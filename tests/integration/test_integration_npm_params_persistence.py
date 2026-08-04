"""Regression test for NPM decomposition-parameter persistence across Step 1 → Step 2.

NPM demultiplexes in memory, so the Step-1 choices (event splitting, the session's timestamp
unit and column) are persisted to the output dir as ``.npm_params.json`` and reloaded in Step 2.
This guards the GUI flow where Step 2 does not receive those params directly: Step 2 must still
reproduce the split-event streams, and read the timestamps on the recorded clock, from the
persisted file alone.
"""

import glob
import json
import os
import shutil

import h5py
import numpy as np

from guppy.testing.api import step1, step2
from guppy_test_data import STUBBED_TESTING_DATA


def test_step2_reproduces_split_events_from_persisted_params(tmp_path):
    src_session = os.path.join(str(STUBBED_TESTING_DATA), "npm", "sampleData_NPM_4")
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copy = tmp_base / "sampleData_NPM_4"
    shutil.copytree(src_session, session_copy)

    store_id_to_store_label = {
        "file0_chev1": "control_region1",
        "file0_chod1": "signal_region1",
        "eventTrue": "ttl_true_region1",
    }

    # Step 1 with split events — persists npm_split_events into the output dir's .npm_params.json.
    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label=store_id_to_store_label,
        npm_split_events=[True, True],
    )

    run_folders = sorted(glob.glob(os.path.join(session_copy, "sampleData_NPM_4_output_*")))
    assert run_folders, "Step 1 did not create an output directory"
    run_folder = run_folders[0]
    assert os.path.exists(os.path.join(run_folder, ".npm_params.json")), "Step 1 did not persist .npm_params.json"

    # Step 2 WITHOUT npm_split_events (as in the GUI, where Step 2 has no access to the
    # interactive choice) must still reproduce the split-event stream from the persisted file.
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        selected_runs={str(session_copy): ["1"]},
    )

    for store_id in store_id_to_store_label:
        hdf5_path = os.path.join(run_folder, f"{store_id}.hdf5")
        assert os.path.exists(hdf5_path), f"Missing HDF5 for split-event store_id {store_id!r}: {hdf5_path}"


def test_step2_reads_timestamps_on_the_unit_recorded_by_step1(tmp_path):
    # sampleData_NPM_5 is header-less: nothing in the file says its clock is in
    # milliseconds, so .npm_params.json is the only record of it (issue #411).
    src_session = os.path.join(str(STUBBED_TESTING_DATA), "npm", "sampleData_NPM_5")
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copy = tmp_base / "sampleData_NPM_5"
    shutil.copytree(src_session, session_copy)

    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label={"file0_chev1": "control_region1", "file0_chod1": "signal_region1"},
        npm_time_unit="milliseconds",
    )

    run_folder = sorted(glob.glob(os.path.join(session_copy, "sampleData_NPM_5_output_*")))[0]
    with open(os.path.join(run_folder, ".npm_params.json")) as npm_params_file:
        npm_params = json.load(npm_params_file)
    assert npm_params["npm_time_unit"] == "milliseconds"

    # Step 2 without the unit must apply the recorded one, not its default of seconds.
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        selected_runs={str(session_copy): ["1"]},
    )

    # chev takes every other row from row 0, whose raw timestamp in
    # PagCeAVgatFear_1512_1.csv is 40263510.4768 ms → 40263.5104768 s. Read as seconds it
    # would have stayed at 40263510.4768.
    with h5py.File(os.path.join(run_folder, "file0_chev1.hdf5"), "r") as hdf5_file:
        first_timestamp = np.asarray(hdf5_file["timestamps"])[0]
    np.testing.assert_allclose(first_timestamp, 40263.5104768, atol=1e-6)
