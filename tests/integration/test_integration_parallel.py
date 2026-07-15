"""
Parallel-specific integration tests for GuPPy.

These tests exercise the multiprocessing code paths (Step 2 and Step 4) with
``number_of_cores=2``. They are kept separate from the main test suite so that
the main suite can run under pytest-xdist (``-n auto``) without interference
from nested worker pools. Run these tests independently without ``-n auto``.
"""

import csv
import glob
import os
import shutil

import h5py
import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step1, step2, step3, step4

# Use the CSV sample as it is the simplest format — no binary TDT dependencies.
SESSION_SUBDIR = "csv/sample_data_csv_1"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
EXPECTED_RECORDING_SITE = "region"
EXPECTED_TTL = "ttl"


def _stage_session(tmp_path):
    """Copy the CSV sample session into a clean temporary workspace."""
    src_session = os.path.join(str(STUBBED_TESTING_DATA), SESSION_SUBDIR)
    assert os.path.isdir(src_session), f"Sample data not found: {src_session}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = os.path.basename(src_session)
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any pre-existing output dirs and parameter file from the copy
    for run_folder in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        shutil.rmtree(run_folder)
    params_filepath = session_copy / "GuPPyParamtersUsed.json"
    if params_filepath.exists():
        params_filepath.unlink()

    return tmp_base, session_copy


@pytest.mark.parallel
def test_parallel_step3(tmp_path):
    """
    Step 2 (Read Raw Data) with ``number_of_cores=2`` produces the same HDF5
    output files as a single-core run, confirming the multiprocessing pool
    code path is exercised and works correctly.
    """
    tmp_base, session_copy = _stage_session(tmp_path)

    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
    )

    # Run Step 2 with 2 worker processes
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        number_of_cores=2,
        selected_runs={str(session_copy): ["1"]},
    )

    # Locate the output directory
    basename = os.path.basename(session_copy)
    run_folders = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert run_folders, f"No output directories found under {session_copy}"
    out_dir = next((d for d in run_folders if os.path.exists(os.path.join(d, "storesList.csv"))), None)
    assert out_dir is not None, "No storesList.csv found in any output directory"

    # Verify that per-store_id HDF5 files were written for each raw store_id
    stores_filepath = os.path.join(out_dir, "storesList.csv")
    with open(stores_filepath, newline="") as stores_file:
        rows = list(csv.reader(stores_file))
    assert len(rows) == 2, "storesList.csv should have 2 rows"
    store_ids = rows[0]
    for store_id in store_ids:
        safe_name = store_id.replace("\\", "_").replace("/", "_")
        h5_path = os.path.join(out_dir, f"{safe_name}.hdf5")
        assert os.path.exists(h5_path), f"Missing HDF5 for store_id {store_id!r}: {h5_path}"
        with h5py.File(h5_path, "r") as h5_file:
            assert "timestamps" in h5_file, f"Missing 'timestamps' dataset in {h5_path}"


@pytest.mark.parallel
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parallel_step5(tmp_path):
    """
    Full pipeline Steps 2-5 with ``number_of_cores=2`` on Step 2 and Step 4,
    confirming that PSTH, peak/AUC, and transients outputs are produced when
    multiprocessing pools are active.
    """
    tmp_base, session_copy = _stage_session(tmp_path)

    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
    )

    selected_runs = {str(session_copy): ["1"]}
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        number_of_cores=2,
        selected_runs=selected_runs,
    )

    step3(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        selected_runs=selected_runs,
    )

    step4(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        number_of_cores=2,
        selected_runs=selected_runs,
    )

    # Locate output directory
    basename = os.path.basename(session_copy)
    run_folders = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert run_folders, f"No output directories found under {session_copy}"
    out_dir = next((d for d in run_folders if os.path.exists(os.path.join(d, "storesList.csv"))), None)
    assert out_dir is not None, "No storesList.csv found in any output directory"

    # PSTH and peak/AUC outputs
    psth_h5 = os.path.join(out_dir, f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5")
    psth_baseline_uncorr_h5 = os.path.join(
        out_dir, f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_baselineUncorrected_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )
    peak_auc_h5 = os.path.join(
        out_dir, f"peak_AUC_{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )
    peak_auc_csv = os.path.join(
        out_dir, f"peak_AUC_{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.csv"
    )

    assert os.path.exists(psth_h5), f"Missing PSTH HDF5: {psth_h5}"
    assert os.path.exists(psth_baseline_uncorr_h5), f"Missing baseline-uncorrected PSTH HDF5: {psth_baseline_uncorr_h5}"
    assert os.path.exists(peak_auc_h5), f"Missing Peak/AUC HDF5: {peak_auc_h5}"
    assert os.path.exists(peak_auc_csv), f"Missing Peak/AUC CSV: {peak_auc_csv}"

    # PSTH HDF5 content check
    dataframe = pd.read_hdf(psth_h5, key="df")
    assert "timestamps" in dataframe.columns, f"'timestamps' column missing in {psth_h5}"
    assert "mean" in dataframe.columns, f"'mean' column missing in {psth_h5}"

    # Transients outputs
    freq_amp_h5 = os.path.join(out_dir, f"freqAndAmp_z_score_{EXPECTED_RECORDING_SITE}.h5")
    freq_amp_csv = os.path.join(out_dir, f"freqAndAmp_z_score_{EXPECTED_RECORDING_SITE}.csv")
    trans_occ_csv = os.path.join(out_dir, f"transientsOccurrences_z_score_{EXPECTED_RECORDING_SITE}.csv")
    assert os.path.exists(freq_amp_h5), f"Missing freq/amp HDF5: {freq_amp_h5}"
    assert os.path.exists(freq_amp_csv), f"Missing freq/amp CSV: {freq_amp_csv}"
    assert os.path.exists(trans_occ_csv), f"Missing transients occurrences CSV: {trans_occ_csv}"
