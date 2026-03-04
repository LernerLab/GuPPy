import glob
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from guppy.testing.api import step2, step3, step4, step5


def _stage_session(src_base_dir, session_subdir, tmp_base):
    """Copy a session to a temp workspace, clean output dirs and param files."""
    src_session = os.path.join(src_base_dir, session_subdir)
    if not os.path.isdir(src_session):
        pytest.skip(f"Sample data not available at expected path: {src_session}")
    dest_name = os.path.basename(src_session)
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        assert os.path.isdir(d)
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()
    return session_copy


def _assert_intra_session_outputs(session_copy, expected_region, expected_ttl):
    dest_name = os.path.basename(session_copy)
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert output_dirs, f"No output directories found in {session_copy}"
    out_dir = None
    for d in output_dirs:
        if os.path.exists(os.path.join(d, "storesList.csv")):
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found under {session_copy}"

    timecorr = os.path.join(out_dir, f"timeCorrection_{expected_region}.hdf5")
    assert os.path.exists(timecorr), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f

    ttl_fp = os.path.join(out_dir, f"{expected_ttl}_{expected_region}.hdf5")
    assert os.path.exists(ttl_fp), f"Missing TTL-aligned file {ttl_fp}"
    with h5py.File(ttl_fp, "r") as f:
        assert "ts" in f


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_tdt_csv_ttl(tmp_path):
    """
    Intra-session mixed modality: TDT photometry channels + CSV event TTL.

    A single TDT session is staged in a temporary workspace. A synthesized CSV TTL file
    (event_csv format: single 'timestamps' column, epoch time) is written into the session
    folder alongside the TDT binary files. The pipeline auto-detects both formats and routes
    TDT stores to TdtRecordingExtractor and the CSV event file to CsvRecordingExtractor.
    """
    src_base_dir = str(Path(".") / "testing_data")
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "SampleData_Clean/Photo_63_207-181030-103332", tmp_base)

    # Five epoch timestamps known to fall inside the session's recording window
    # (~1540913634–1540917275 s), spaced 10 minutes apart
    csv_ttl_timestamps = np.array([1540914000.0, 1540914600.0, 1540915200.0, 1540915800.0, 1540916400.0])
    np.savetxt(session_copy / "csv_port_entries.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    base_dir = str(tmp_base)
    selected_folders = [str(session_copy)]

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map={"Dv1A": "control_dms", "Dv2A": "signal_dms", "csv_port_entries": "port_entries_dms"},
    )
    step3(base_dir=base_dir, selected_folders=selected_folders)
    step4(base_dir=base_dir, selected_folders=selected_folders)
    step5(base_dir=base_dir, selected_folders=selected_folders)

    _assert_intra_session_outputs(session_copy, expected_region="dms", expected_ttl="port_entries_dms")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_doric_csv_ttl(tmp_path):
    """
    Intra-session mixed modality: Doric photometry channels + CSV event TTL.

    A single Doric session is staged in a temporary workspace. A synthesized CSV TTL file
    (single 'timestamps' column, relative seconds from recording start) is written into the
    session folder alongside the .doric binary file. The pipeline auto-detects both formats
    and routes Doric photometry to DoricRecordingExtractor and the CSV event to
    CsvRecordingExtractor within the same session folder.

    The Doric recording window for sample_doric_3 runs from 0 to ~1811 seconds (relative time).
    """
    src_base_dir = str(Path(".") / "testing_data")
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "SampleData_Doric/sample_doric_3", tmp_base)

    # Five timestamps within the Doric recording window (0–1811 s relative), spaced 5 min apart
    csv_ttl_timestamps = np.array([300.0, 600.0, 900.0, 1200.0, 1500.0])
    np.savetxt(session_copy / "csv_doric_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    base_dir = str(tmp_base)
    selected_folders = [str(session_copy)]

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map={
            "CAM1_EXC1/ROI01": "control_region",
            "CAM1_EXC2/ROI01": "signal_region",
            "csv_doric_event": "ttl_region",
        },
    )
    step3(base_dir=base_dir, selected_folders=selected_folders)
    step4(base_dir=base_dir, selected_folders=selected_folders)
    step5(base_dir=base_dir, selected_folders=selected_folders)

    _assert_intra_session_outputs(session_copy, expected_region="region", expected_ttl="ttl_region")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_npm_csv_ttl(tmp_path):
    """
    Intra-session mixed modality: NPM photometry channels + external CSV event TTL.

    A single NPM session is staged in a temporary workspace. A synthesized CSV TTL file
    (single 'timestamps' column, relative seconds matching the NPM recording's output domain)
    is written into the session folder alongside the NPM data files. detect_acquisition_formats
    detects both 'npm' and 'csv' formats: NPM-generated split-event files (named event*.csv)
    are suppressed as before, while the external CSV (named csv_event.csv) is passed through
    to CsvRecordingExtractor.

    NPM_1 photometry timestamps are rescaled to relative seconds (~0–1858 s). The external
    event CSV uses timestamps in that same relative domain so PSTH alignment succeeds.
    """
    src_base_dir = str(Path(".") / "testing_data")
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "SampleData_Neurophotometrics/sampleData_NPM_1", tmp_base)

    # Five timestamps in relative seconds matching the NPM_1 output domain (0–~1858 s),
    # spaced ~5 min apart. CsvRecordingExtractor reads these as-is without rescaling.
    csv_ttl_timestamps = np.array([300.0, 600.0, 900.0, 1200.0, 1500.0])
    np.savetxt(session_copy / "csv_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    base_dir = str(tmp_base)
    selected_folders = [str(session_copy)]

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map={
            "file0_chev1": "signal_region",
            "file0_chod1": "control_region",
            "csv_event": "ttl_region",
        },
        npm_split_events=[False, True],
    )
    step3(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[False, True])
    step4(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[False, True])
    step5(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[False, True])

    _assert_intra_session_outputs(session_copy, expected_region="region", expected_ttl="ttl_region")
