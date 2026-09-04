import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import holoviews as hv
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_combine_data(tmp_path):
    session_subdirs = [
        "tdt/Photo_048_392-200728-121222",
        "tdt/Photo_63_207-181030-103332",
    ]
    store_id_to_store_label = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "PrtN": "port_entries_dms",
    }
    expected_recording_site = "dms"
    expected_ttl = "port_entries_dms"

    npm_timestamp_column_name = None
    npm_time_unit = None
    npm_split_events = None

    # Use the CSV sample session
    src_base_dir = str(STUBBED_TESTING_DATA)
    src_sessions = [Path(src_base_dir) / session_subdir for session_subdir in session_subdirs]
    for src_session in src_sessions:
        assert Path(src_session).is_dir(), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copies = []
    for src_session in src_sessions:
        dest_name = Path(src_session).name
        session_copy = tmp_base / dest_name
        shutil.copytree(src_session, session_copy)
        session_copies.append(session_copy)

    for session_copy in session_copies:
        # Remove any copied artifacts in the temp session (match only this session's output dirs)
        for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
            assert Path(d).is_dir(), f"Expected output directory for cleanup, got non-directory: {d}"
            shutil.rmtree(d)
        params_fp = session_copy / "GuPPyParamtersUsed.json"
        if params_fp.exists():
            params_fp.unlink()

    selected_folders = [str(session_copy) for session_copy in session_copies]
    base_dir = str(tmp_base)

    # Step 1: create storesList.csv in the temp copy
    step1(
        base_dir=base_dir,
        selected_folders=selected_folders,
        store_id_to_store_label=store_id_to_store_label,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
    )

    selected_runs = {selected_folder: ["1"] for selected_folder in selected_folders}

    # Step 2: read raw data in the temp copy
    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        selected_runs=selected_runs,
    )

    # Step 3: extract timestamps and signal in the temp copy
    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=True,
        selected_runs=selected_runs,
    )

    # Step 4: compute PSTH in the temp copy (headless)
    step4(
        base_dir=str(tmp_base),
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=True,
        selected_runs=selected_runs,
    )

    # Validate outputs exist in the temp copy
    session_copy = selected_folders[0]  # Outputs are written to the first session folder
    basename = Path(session_copy).name
    run_folders = sorted(list(Path(session_copy).glob(f"{basename}_output_*")))
    assert run_folders, f"No output directories found in {session_copy}"
    out_dir = None
    for d in run_folders:
        if (Path(d) / "storesList.csv").exists():
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"
    stores_fp = Path(out_dir) / "storesList.csv"
    assert Path(stores_fp).exists(), "Missing storesList.csv after Step 1/2/3"

    # Ensure timeCorrection_<recording_site>.hdf5 exists with 'timestampNew'
    timecorr = Path(out_dir) / (f"timeCorrection_{expected_recording_site}.hdf5")
    assert Path(timecorr).exists(), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    # If TTLs exist, check their per-recording-site 'ts' outputs
    if expected_ttl is None:
        expected_ttls = []
    elif isinstance(expected_ttl, str):
        expected_ttls = [expected_ttl]
    else:
        expected_ttls = expected_ttl
    for expected_ttl in expected_ttls:
        ttl_fp = Path(out_dir) / (f"{expected_ttl}_{expected_recording_site}.hdf5")
        assert Path(ttl_fp).exists(), f"Missing TTL-aligned file {ttl_fp}"
        with h5py.File(ttl_fp, "r") as f:
            assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"

    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step5(
                base_dir=base_dir,
                selected_folders=[str(session_copies[0])],
                selected_runs={str(session_copies[0]): ["1"]},
            )

    assert len(captured_dashboards) >= 1, "step5 created no VisualizationDashboard instances"
