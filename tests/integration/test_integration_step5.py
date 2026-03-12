import glob
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from guppy.testing.api import step2, step3, step4, step5


@pytest.mark.parametrize(
    "session_subdir, storenames_map, expected_region, expected_ttl",
    [
        (
            "csv/sample_data_csv_1",
            {
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
                "Sample_TTL": "ttl",
            },
            "region",
            "ttl",
        ),
        (
            "doric/sample_doric_1",
            {
                "AIn-1 - Raw": "control_region",
                "AIn-2 - Raw": "signal_region",
                "DI--O-1": "ttl",
            },
            "region",
            "ttl",
        ),
        (
            "doric/sample_doric_2",
            {
                "AIn-1 - Dem (ref)": "control_region",
                "AIn-1 - Dem (da)": "signal_region",
                "DI/O-1": "ttl",
            },
            "region",
            "ttl",
        ),
        (
            "doric/sample_doric_3",
            {
                "CAM1_EXC1/ROI01": "control_region",
                "CAM1_EXC2/ROI01": "signal_region",
                "DigitalIO/CAM1": "ttl",
            },
            "region",
            "ttl",
        ),
        (
            "doric/sample_doric_4",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
            "region",
            None,
        ),
        (
            "doric/sample_doric_5",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
            "region",
            None,
        ),
        (
            "tdt/Photo_63_207-181030-103332",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
            "dms",
            "port_entries_dms",
        ),
        (
            "tdt/Photometry-161823",
            {
                "405R": "control_region",
                "490R": "signal_region",
                "PAB/": "ttl",
            },
            "region",
            ["PAB_0", "PAB_16", "PAB_2064"],  # This session has an event which gets split into three sub-events.
        ),
        (
            "tdt/Photo_048_392-200728-121222",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
            "dms",
            "port_entries_dms",
        ),
        (
            "npm/sampleData_NPM_1",
            {
                "file0_chev1": "signal_region",
                "file0_chod1": "control_region",
            },
            "region",
            None,
        ),
        (
            "npm/sampleData_NPM_2",
            {
                "file0_chev6": "control_region",
                "file1_chev6": "signal_region",
            },
            "region",
            None,
        ),
        (
            "npm/sampleData_NPM_3",
            {
                "file0_chev3": "control_region3",
                "file0_chod3": "signal_region3",
                "event3": "ttl_region3",
            },
            "region3",
            "ttl_region3",
        ),
        (
            "npm/sampleData_NPM_4",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "eventTrue": "ttl_true_region1",
            },
            "region1",
            "ttl_true_region1",
        ),
        (
            "npm/sampleData_NPM_5",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "event0": "ttl_region1",
            },
            "region1",
            "ttl_region1",
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
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5(tmp_path, session_subdir, storenames_map, expected_region, expected_ttl):
    """
    Full integration test for Step 5 (PSTH Computation) using real CSV sample data,
    isolated to a temporary workspace to avoid mutating shared sample data.

    Pipeline executed on a temp copy:
      - Step 2: save storenames (storesList.csv)
      - Step 3: read raw data (per-storename HDF5 outputs)
      - Step 4: extract timestamps/signal, z-score/dFF, time corrections
      - Step 5: compute PSTH and peak/AUC outputs

    Assertions confirm creation and basic readability of PSTH-related outputs from Step 5.
    Defaults are used for input parameters; PSTH computation defaults to z_score.
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

    # Use the sample session
    src_base_dir = str(Path(".") / "stubbed_testing_data")
    src_session = os.path.join(src_base_dir, session_subdir)
    assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = os.path.basename(src_session)
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session (match only this session's output dirs)
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    # Step 2: create storesList.csv in the temp copy with explicit naming
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        storenames_map=storenames_map,
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Step 3: read raw data in the temp copy
    step3(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Step 4: extract timestamps and signal in the temp copy
    step4(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Step 5: compute PSTH in the temp copy (headless)
    step5(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Locate output directory
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
    assert os.path.exists(stores_fp), "Missing storesList.csv after Steps 2-5"

    # Expected PSTH outputs (defaults compute z_score PSTH) - only for datasets with TTLs
    if expected_ttl is None:
        expected_ttls = []
    elif isinstance(expected_ttl, str):
        expected_ttls = [expected_ttl]
    else:
        expected_ttls = expected_ttl
    for expected_ttl in expected_ttls:
        psth_h5 = os.path.join(out_dir, f"{expected_ttl}_{expected_region}_z_score_{expected_region}.h5")
        psth_baseline_uncorr_h5 = os.path.join(
            out_dir, f"{expected_ttl}_{expected_region}_baselineUncorrected_z_score_{expected_region}.h5"
        )
        peak_auc_h5 = os.path.join(out_dir, f"peak_AUC_{expected_ttl}_{expected_region}_z_score_{expected_region}.h5")
        peak_auc_csv = os.path.join(out_dir, f"peak_AUC_{expected_ttl}_{expected_region}_z_score_{expected_region}.csv")

        # Assert file creation
        assert os.path.exists(psth_h5), f"Missing PSTH HDF5: {psth_h5}"
        assert os.path.exists(
            psth_baseline_uncorr_h5
        ), f"Missing baseline-uncorrected PSTH HDF5: {psth_baseline_uncorr_h5}"
        assert os.path.exists(peak_auc_h5), f"Missing PSTH Peak/AUC HDF5: {peak_auc_h5}"
        assert os.path.exists(peak_auc_csv), f"Missing PSTH Peak/AUC CSV: {peak_auc_csv}"

        # Basic readability checks: PSTH HDF5 contains a DataFrame with expected columns
        df = pd.read_hdf(psth_h5, key="df")
        assert "timestamps" in df.columns, f"'timestamps' column missing in {psth_h5}"
        # The DataFrame should include a 'mean' column per create_Df implementation
        assert "mean" in df.columns, f"'mean' column missing in {psth_h5}"

    # Additional artifacts from transients frequency/amplitude computation (Step 5 side-effect)
    freq_amp_h5 = os.path.join(out_dir, f"freqAndAmp_z_score_{expected_region}.h5")
    freq_amp_csv = os.path.join(out_dir, f"freqAndAmp_z_score_{expected_region}.csv")
    trans_occ_csv = os.path.join(out_dir, f"transientsOccurrences_z_score_{expected_region}.csv")
    assert os.path.exists(freq_amp_h5), f"Missing freq/amp HDF5: {freq_amp_h5}"
    assert os.path.exists(freq_amp_csv), f"Missing freq/amp CSV: {freq_amp_csv}"
    assert os.path.exists(trans_occ_csv), f"Missing transients occurrences CSV: {trans_occ_csv}"
