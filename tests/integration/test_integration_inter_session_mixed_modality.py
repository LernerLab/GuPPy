import shutil
from pathlib import Path

import h5py
import pytest

from guppy.testing.api import locate_run_folder, step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA

# One session per acquisition format, with the labels Step 1 gives it and the outputs Steps 2-4
# must then write for it.
SESSIONS = {
    "tdt": {
        "session_subdir": "tdt/Photo_63_207-181030-103332",
        "store_id_to_store_label": {"Dv1A": "control_dms", "Dv2A": "signal_dms", "PrtN": "port_entries_dms"},
        "expected_recording_site": "dms",
        "expected_ttl": "port_entries_dms",
    },
    "npm": {
        # sampleData_NPM_4 splits its event file into one event per state.
        "session_subdir": "npm/sampleData_NPM_4",
        "store_id_to_store_label": {
            "PagCeAVgatFear_14421_415nm_Region0G": "control_region1",
            "PagCeAVgatFear_14421_470nm_Region0G": "signal_region1",
            "eventTrue": "ttl_true_region1",
        },
        "expected_recording_site": "region1",
        "expected_ttl": "ttl_true_region1",
    },
    "doric": {
        "session_subdir": "doric/sample_doric_3",
        "store_id_to_store_label": {
            "CAM1_EXC1/ROI01": "control_region",
            "CAM1_EXC2/ROI01": "signal_region",
            "DigitalIO/CAM1": "ttl",
        },
        "expected_recording_site": "region",
        "expected_ttl": "ttl",
    },
    "csv": {
        "session_subdir": "csv/sample_data_csv_1",
        "store_id_to_store_label": {
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
        "expected_recording_site": "region",
        "expected_ttl": "ttl",
    },
    "nwb": {
        "session_subdir": "nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
        "store_id_to_store_label": {
            "fiber_photometry_response_series_0": "control_region",
            "fiber_photometry_response_series_1": "signal_region",
            "events": "ttl",
        },
        "expected_recording_site": "region",
        "expected_ttl": "ttl",
    },
}


def _stage_session(src_base_dir, session_subdir, tmp_base):
    """Copy a session to a temp workspace, clean output dirs and param files."""
    src_session = Path(src_base_dir) / session_subdir
    assert Path(src_session).is_dir(), f"Sample data not available at expected path: {src_session}"
    dest_name = Path(src_session).name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)
    for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
        assert Path(d).is_dir()
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()
    return session_copy


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality(tmp_path):
    """
    Inter-session mixed modality: one session of every acquisition format processed together.

    Each session uses its own acquisition format; modality is auto-detected per folder.
    Step 1 runs separately per session; steps 2–4 run together across all sessions.

    Data is copied from the individual modality source directories, not SampleData_mixed_modality/,
    because the mixed-modality folder does not carry actual data files on CI.
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "input_root_folder"
    tmp_base.mkdir(parents=True, exist_ok=True)
    base_dir = str(tmp_base)

    session_copies = {
        acquisition_format: _stage_session(src_base_dir, session["session_subdir"], tmp_base)
        for acquisition_format, session in SESSIONS.items()
    }

    # step1 must run per-session: each session's storesList.csv must contain only its own channels.
    # The pipeline would otherwise try to read one format's channels from another's folder.
    for acquisition_format, session in SESSIONS.items():
        step1(
            base_dir=base_dir,
            selected_folders=[str(session_copies[acquisition_format])],
            store_id_to_store_label=session["store_id_to_store_label"],
            npm_split_events={"PagCeAVgatFear_1442_ts0.csv": True} if acquisition_format == "npm" else None,
        )

    # Steps 2–4 run once with every session; each session's storesList.csv is read independently.
    selected_folders = [str(session_copy) for session_copy in session_copies.values()]
    selected_runs = {folder: ["1"] for folder in selected_folders}
    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events={"PagCeAVgatFear_1442_ts0.csv": True},
        selected_runs=selected_runs,
    )
    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events={"PagCeAVgatFear_1442_ts0.csv": True},
        selected_runs=selected_runs,
    )
    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events={"PagCeAVgatFear_1442_ts0.csv": True},
        selected_runs=selected_runs,
    )

    for acquisition_format, session in SESSIONS.items():
        _assert_pipeline_outputs(
            session_copies[acquisition_format],
            expected_recording_site=session["expected_recording_site"],
            expected_ttl=session["expected_ttl"],
        )


def _assert_pipeline_outputs(session_copy, expected_recording_site, expected_ttl):
    out_dir = locate_run_folder(session=str(session_copy))
    assert (Path(out_dir) / "storesList.csv").exists(), "Missing storesList.csv"

    timecorr = Path(out_dir) / (f"timeCorrection_{expected_recording_site}.hdf5")
    assert Path(timecorr).exists(), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    ttl_fp = Path(out_dir) / (f"{expected_ttl}_{expected_recording_site}.hdf5")
    assert Path(ttl_fp).exists(), f"Missing TTL-aligned file {ttl_fp}"
    with h5py.File(ttl_fp, "r") as f:
        assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"
