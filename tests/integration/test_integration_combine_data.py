import glob
import os
import shutil
from unittest.mock import patch

import h5py
import holoviews as hv
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step2, step3, step4, step5, step6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_combine_data(tmp_path):
    session_subdirs = [
        "tdt/Photo_048_392-200728-121222",
        "tdt/Photo_63_207-181030-103332",
    ]
    storenames_map = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "PrtN": "port_entries_dms",
    }
    expected_region = "dms"
    expected_ttl = "port_entries_dms"

    npm_timestamp_column_names = None
    npm_time_units = None
    npm_split_events = [True, True]

    # Use the CSV sample session
    src_base_dir = str(STUBBED_TESTING_DATA)
    src_sessions = [os.path.join(src_base_dir, session_subdir) for session_subdir in session_subdirs]
    for src_session in src_sessions:
        assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copies = []
    for src_session in src_sessions:
        dest_name = os.path.basename(src_session)
        session_copy = tmp_base / dest_name
        shutil.copytree(src_session, session_copy)
        session_copies.append(session_copy)

    for session_copy in session_copies:
        # Remove any copied artifacts in the temp session (match only this session's output dirs)
        for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
            assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
            shutil.rmtree(d)
        params_fp = session_copy / "GuPPyParamtersUsed.json"
        if params_fp.exists():
            params_fp.unlink()

    selected_folders = [str(session_copy) for session_copy in session_copies]
    base_dir = str(tmp_base)

    # Step 2: create storesList.csv in the temp copy
    step2(
        base_dir=base_dir,
        selected_folders=selected_folders,
        storenames_map=storenames_map,
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Step 3: read raw data in the temp copy
    step3(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
    )

    # Step 4: extract timestamps and signal in the temp copy
    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
        combine_data=True,
    )

    # Step 5: compute PSTH in the temp copy (headless)
    step5(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        npm_timestamp_column_names=npm_timestamp_column_names,
        npm_time_units=npm_time_units,
        npm_split_events=npm_split_events,
        combine_data=True,
    )

    # Validate outputs exist in the temp copy
    session_copy = selected_folders[0]  # Outputs are written to the first session folder
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
    if expected_ttl is None:
        expected_ttls = []
    elif isinstance(expected_ttl, str):
        expected_ttls = [expected_ttl]
    else:
        expected_ttls = expected_ttl
    for expected_ttl in expected_ttls:
        ttl_fp = os.path.join(out_dir, f"{expected_ttl}_{expected_region}.hdf5")
        assert os.path.exists(ttl_fp), f"Missing TTL-aligned file {ttl_fp}"
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
            step6(base_dir=base_dir, selected_folders=[str(session_copies[0])])

    assert len(captured_dashboards) >= 1, "step6 created no VisualizationDashboard instances"
