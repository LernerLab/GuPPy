import glob
import os
import shutil

import h5py
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step2, step3, step4, step5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality(tmp_path):
    """
    Integration test for auto modality detection across sessions with different acquisition formats.

    A Neurophotometrics session and a Doric session are staged in the same temporary workspace
    and processed together through the full pipeline in a single set of step calls. Each session's
    modality is auto-detected per folder, exercising the 'auto' default across two different formats
    within the same pipeline run.

    Data is copied from the individual modality source directories, not SampleData_mixed_modality/,
    because the mixed-modality folder does not carry actual data files on CI.
    """
    npm_session_subdir = "npm/sampleData_NPM_4"
    doric_session_subdir = "doric/sample_doric_3"

    npm_storenames_map = {
        "file0_chev1": "control_region1",
        "file0_chod1": "signal_region1",
        "eventTrue": "ttl_true_region1",
    }
    doric_storenames_map = {
        "CAM1_EXC1/ROI01": "control_region",
        "CAM1_EXC2/ROI01": "signal_region",
        "DigitalIO/CAM1": "ttl",
    }

    src_base_dir = str(STUBBED_TESTING_DATA)
    npm_src = os.path.join(src_base_dir, npm_session_subdir)
    doric_src = os.path.join(src_base_dir, doric_session_subdir)

    # Stage a clean copy of each session into a shared temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    npm_dest = tmp_base / "sampleData_NPM_4"
    doric_dest = tmp_base / "sample_doric_3"
    shutil.copytree(npm_src, npm_dest)
    shutil.copytree(doric_src, doric_dest)

    for session_copy in [npm_dest, doric_dest]:
        dest_name = os.path.basename(session_copy)
        for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
            assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
            shutil.rmtree(d)
        params_fp = session_copy / "GuPPyParamtersUsed.json"
        if params_fp.exists():
            params_fp.unlink()

    base_dir = str(tmp_base)
    npm_folder = str(npm_dest)
    doric_folder = str(doric_dest)
    selected_folders = [npm_folder, doric_folder]

    # step2 must run per-session: each session's storesList.csv must contain only its own channels.
    # The pipeline would otherwise try to read Doric channels from the NPM folder (and vice versa).
    step2(
        base_dir=base_dir,
        selected_folders=[npm_folder],
        storenames_map=npm_storenames_map,
        npm_split_events=[True, True],
    )
    step2(
        base_dir=base_dir,
        selected_folders=[doric_folder],
        storenames_map=doric_storenames_map,
    )

    # Steps 3–5 run once with both sessions; each session's storesList.csv is read independently.
    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events=[True, True],
    )
    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events=[True, True],
    )
    step5(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_split_events=[True, True],
    )

    # Validate NPM session outputs
    _assert_pipeline_outputs(npm_dest, expected_region="region1", expected_ttl="ttl_true_region1")

    # Validate Doric session outputs
    _assert_pipeline_outputs(doric_dest, expected_region="region", expected_ttl="ttl")


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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_tdt_doric(tmp_path):
    """
    Inter-session mixed modality: TDT session + Doric session processed together.

    Each session uses its own acquisition format; modality is auto-detected per folder.
    Step 2 runs separately per session; steps 3–5 run together across both sessions.
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    tdt_session = _stage_session(src_base_dir, "tdt/Photo_63_207-181030-103332", tmp_base)
    doric_session = _stage_session(src_base_dir, "doric/sample_doric_3", tmp_base)

    base_dir = str(tmp_base)

    step2(
        base_dir=base_dir,
        selected_folders=[str(tdt_session)],
        storenames_map={"Dv1A": "control_dms", "Dv2A": "signal_dms", "PrtN": "port_entries_dms"},
    )
    step2(
        base_dir=base_dir,
        selected_folders=[str(doric_session)],
        storenames_map={
            "CAM1_EXC1/ROI01": "control_region",
            "CAM1_EXC2/ROI01": "signal_region",
            "DigitalIO/CAM1": "ttl",
        },
    )

    selected_folders = [str(tdt_session), str(doric_session)]
    step3(base_dir=base_dir, selected_folders=selected_folders)
    step4(base_dir=base_dir, selected_folders=selected_folders)
    step5(base_dir=base_dir, selected_folders=selected_folders)

    _assert_pipeline_outputs(tdt_session, expected_region="dms", expected_ttl="port_entries_dms")
    _assert_pipeline_outputs(doric_session, expected_region="region", expected_ttl="ttl")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_tdt_npm(tmp_path):
    """
    Inter-session mixed modality: TDT session + NPM session processed together.

    Each session uses its own acquisition format; modality is auto-detected per folder.
    Step 2 runs separately per session; steps 3–5 run together across both sessions.
    The NPM session (sampleData_NPM_4) uses split events.
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    tdt_session = _stage_session(src_base_dir, "tdt/Photo_63_207-181030-103332", tmp_base)
    npm_session = _stage_session(src_base_dir, "npm/sampleData_NPM_4", tmp_base)

    base_dir = str(tmp_base)

    step2(
        base_dir=base_dir,
        selected_folders=[str(tdt_session)],
        storenames_map={"Dv1A": "control_dms", "Dv2A": "signal_dms", "PrtN": "port_entries_dms"},
    )
    step2(
        base_dir=base_dir,
        selected_folders=[str(npm_session)],
        storenames_map={
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "eventTrue": "ttl_true_region1",
        },
        npm_split_events=[True, True],
    )

    selected_folders = [str(tdt_session), str(npm_session)]
    step3(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[True, True])
    step4(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[True, True])
    step5(base_dir=base_dir, selected_folders=selected_folders, npm_split_events=[True, True])

    _assert_pipeline_outputs(tdt_session, expected_region="dms", expected_ttl="port_entries_dms")
    _assert_pipeline_outputs(npm_session, expected_region="region1", expected_ttl="ttl_true_region1")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mixed_modality_tdt_csv_data(tmp_path):
    """
    Inter-session mixed modality: TDT session + CSV data session processed together.

    Each session uses its own acquisition format; modality is auto-detected per folder.
    Step 2 runs separately per session; steps 3–5 run together across both sessions.
    """
    src_base_dir = str(STUBBED_TESTING_DATA)
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)

    tdt_session = _stage_session(src_base_dir, "tdt/Photo_63_207-181030-103332", tmp_base)
    csv_session = _stage_session(src_base_dir, "csv/sample_data_csv_1", tmp_base)

    base_dir = str(tmp_base)

    step2(
        base_dir=base_dir,
        selected_folders=[str(tdt_session)],
        storenames_map={"Dv1A": "control_dms", "Dv2A": "signal_dms", "PrtN": "port_entries_dms"},
    )
    step2(
        base_dir=base_dir,
        selected_folders=[str(csv_session)],
        storenames_map={
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
    )

    selected_folders = [str(tdt_session), str(csv_session)]
    step3(base_dir=base_dir, selected_folders=selected_folders)
    step4(base_dir=base_dir, selected_folders=selected_folders)
    step5(base_dir=base_dir, selected_folders=selected_folders)

    _assert_pipeline_outputs(tdt_session, expected_region="dms", expected_ttl="port_entries_dms")
    _assert_pipeline_outputs(csv_session, expected_region="region", expected_ttl="ttl")


def _assert_pipeline_outputs(session_copy, expected_region, expected_ttl):
    basename = os.path.basename(session_copy)
    output_dirs = sorted(glob.glob(os.path.join(session_copy, f"{basename}_output_*")))
    assert output_dirs, f"No output directories found in {session_copy}"
    out_dir = None
    for d in output_dirs:
        if os.path.exists(os.path.join(d, "storesList.csv")):
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"
    assert os.path.exists(os.path.join(out_dir, "storesList.csv")), "Missing storesList.csv"

    timecorr = os.path.join(out_dir, f"timeCorrection_{expected_region}.hdf5")
    assert os.path.exists(timecorr), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    ttl_fp = os.path.join(out_dir, f"{expected_ttl}_{expected_region}.hdf5")
    assert os.path.exists(ttl_fp), f"Missing TTL-aligned file {ttl_fp}"
    with h5py.File(ttl_fp, "r") as f:
        assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"
