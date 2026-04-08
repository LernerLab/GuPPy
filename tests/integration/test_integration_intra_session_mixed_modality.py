import glob
import os
import shutil

import h5py
import numpy as np
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2, step3, step4, step5


def _stage_session(src_base_dir, session_subdir, tmp_base):
    """Copy a session to a temp workspace, clean output dirs and param files."""
    src_session = os.path.join(src_base_dir, session_subdir)
    assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"
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
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "tdt/Photo_63_207-181030-103332", tmp_base)

    # Five epoch timestamps known to fall inside the session's recording window
    # (~1540913634–1540913791.4 s), spaced 30s apart
    csv_ttl_timestamps = np.array([1540913664.0, 1540913694.0, 1540913724.0, 1540913754.0, 1540913784.0])
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

    The Doric recording window for sample_doric_3 runs from 0 to ~16 seconds (relative time).
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "doric/sample_doric_3", tmp_base)

    # Five timestamps within the Doric recording window (0–16 s relative), spaced 3s apart
    csv_ttl_timestamps = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
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

    NPM_1 photometry timestamps are rescaled to relative seconds (~0–120 s). The external
    event CSV uses timestamps in that same relative domain so PSTH alignment succeeds.
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "npm/sampleData_NPM_1", tmp_base)

    # Five timestamps in relative seconds matching the NPM_1 output domain (0–~120 s),
    # spaced ~20 s apart. CsvRecordingExtractor reads these as-is without rescaling.
    csv_ttl_timestamps = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_nwb_csv_ttl(tmp_path):
    """
    Intra-session mixed modality: NWB photometry channels + external CSV event TTL.

    A single NWB session is staged in a temporary workspace. A synthesized CSV TTL file
    (single 'timestamps' column, relative seconds within the NWB recording window) is written
    into the session folder alongside the .nwb file. The pipeline auto-detects both formats and
    routes NWB photometry to NwbRecordingExtractor and the CSV event file to CsvRecordingExtractor.

    The mock NWB recording window runs from 0 to ~99.97 seconds (3000 samples at 30 Hz).
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    session_copy = _stage_session(src_base_dir, "nwb/mock_nwbfile", tmp_base)

    # Five timestamps within the NWB recording window (0–~100 s), spaced ~20 s apart
    csv_ttl_timestamps = np.array([20.0, 40.0, 60.0, 80.0])
    np.savetxt(session_copy / "csv_nwb_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    base_dir = str(tmp_base)
    selected_folders = [str(session_copy)]

    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map={
            "fiber_photometry_response_series_0": "control_region",
            "fiber_photometry_response_series_1": "signal_region",
            "csv_nwb_event": "ttl_region",
        },
    )
    step3(base_dir=base_dir, selected_folders=selected_folders)
    step4(base_dir=base_dir, selected_folders=selected_folders)
    step5(base_dir=base_dir, selected_folders=selected_folders)

    _assert_intra_session_outputs(session_copy, expected_region="region", expected_ttl="ttl_region")
