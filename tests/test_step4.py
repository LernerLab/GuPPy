import glob
import os
import shutil
from pathlib import Path

import h5py
import pytest

from guppy.testing.api import step2, step3, step4


@pytest.mark.parametrize(
    "session_subdir, storenames_map, expected_region, expected_ttl, modality",
    [
        (
            "SampleData_csv/sample_data_csv_1",
            {
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
                "Sample_TTL": "ttl",
            },
            "region",
            "ttl",
            "csv",
        ),
        (
            "SampleData_Doric/sample_doric_1",
            {
                "AIn-1 - Raw": "control_region",
                "AIn-2 - Raw": "signal_region",
                "DI--O-1": "ttl",
            },
            "region",
            "ttl",
            "doric",
        ),
        (
            "SampleData_Doric/sample_doric_2",
            {
                "AIn-1 - Dem (ref)": "control_region",
                "AIn-1 - Dem (da)": "signal_region",
                "DI/O-1": "ttl",
            },
            "region",
            "ttl",
            "doric",
        ),
        (
            "SampleData_Doric/sample_doric_3",
            {
                "CAM1_EXC1/ROI01": "control_region",
                "CAM1_EXC2/ROI01": "signal_region",
                "DigitalIO/CAM1": "ttl",
            },
            "region",
            "ttl",
            "doric",
        ),
        (
            "SampleData_Doric/sample_doric_4",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
            "region",
            None,
            "doric",
        ),
        (
            "SampleData_Doric/sample_doric_5",
            {
                "Series0001/AIN01xAOUT01-LockIn": "control_region",
                "Series0001/AIN01xAOUT02-LockIn": "signal_region",
            },
            "region",
            None,
            "doric",
        ),
        (
            "SampleData_Clean/Photo_63_207-181030-103332",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
            "dms",
            "port_entries_dms",
            "tdt",
        ),
        (
            "SampleData_Clean/Photometry-161823",
            {
                "405R": "control_region",
                "490R": "signal_region",
                "PAB/": "ttl",
            },
            "region",
            "ttl",
            "tdt",
        ),
        (
            "SampleData_with_artifacts/Photo_048_392-200728-121222",
            {
                "Dv1A": "control_dms",
                "Dv2A": "signal_dms",
                "PrtN": "port_entries_dms",
            },
            "dms",
            "port_entries_dms",
            "tdt",
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_2",
            {
                "file0_chev6": "control_region",
                "file1_chev6": "signal_region",
            },
            "region",
            None,
            "npm",
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_3",
            {
                "file0_chev3": "control_region3",
                "file0_chod3": "signal_region3",
                "event3": "ttl_region3",
            },
            "region3",
            "ttl_region3",
            "npm",
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_4",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "eventTrue": "ttl_true_region1",
            },
            "region1",
            "ttl_true_region1",
            "npm",
        ),
        (
            "SampleData_Neurophotometrics/sampleData_NPM_5",
            {
                "file0_chev1": "control_region1",
                "file0_chod1": "signal_region1",
                "event0": "ttl_region1",
            },
            "region1",
            "ttl_region1",
            "npm",
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
        "tdt_check_data",
        "tdt_with_artifacts",
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step4(tmp_path, monkeypatch, session_subdir, storenames_map, expected_region, expected_ttl, modality):
    """
    Full integration test for Step 4 (Extract timestamps and signal) using real CSV sample data,
    isolated to a temporary workspace to avoid mutating shared sample data.

    Pipeline executed on a temp copy:
      - Step 2: create storesList.csv (derived from sample data if not present)
      - Step 3: read raw data (per-storename HDF5 files)
      - Step 4: extract timestamps/signal, compute z-score/dFF, time corrections, etc.

    Notes:
      - matplotlib plotting in preprocess uses a GUI backend; to avoid blocking, we stub plt.show().
      - Assertions confirm creation of key HDF5 outputs expected from Step 4.
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

    # Use the CSV sample session
    src_base_dir = str(Path(".") / "testing_data")
    src_session = os.path.join(src_base_dir, session_subdir)
    if not os.path.isdir(src_session):
        pytest.skip(f"Sample data not available at expected path: {src_session}")

    # Stub matplotlib.pyplot.show to avoid GUI blocking
    import matplotlib.pyplot as plt  # noqa: F401

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

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

    # Step 2: create storesList.csv in the temp copy
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        storenames_map=storenames_map,
        modality=modality,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
    )

    # Step 3: read raw data in the temp copy
    step3(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        modality=modality,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
    )

    # Step 4: extract timestamps and signal in the temp copy
    step4(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        modality=modality,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
    )

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
    assert os.path.exists(stores_fp), "Missing storesList.csv after Step 2/3/4"

    # Ensure timeCorrection_<region>.hdf5 exists with 'timestampNew'
    timecorr = os.path.join(out_dir, f"timeCorrection_{expected_region}.hdf5")
    assert os.path.exists(timecorr), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    # If TTLs exist, check their per-region 'ts' outputs
    if expected_ttl is not None:
        ttl_fp = os.path.join(out_dir, f"{expected_ttl}_{expected_region}.hdf5")
        assert os.path.exists(ttl_fp), f"Missing TTL-aligned file {ttl_fp}"
        with h5py.File(ttl_fp, "r") as f:
            assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"
