import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import pandas as pd
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5
from guppy_test_data import STUBBED_TESTING_DATA


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cross_correlation(tmp_path):
    """
    Integration test for Step 4 cross-correlation computation.

    Uses a two-recording-site TDT session (DMS + DLS) with a generic TTL event to verify
    that cross-correlation output files are created when compute_corr=True.

    Pipeline executed on a temp copy:
      - Step 1: save store_ids (storesList.csv)
      - Step 2: read raw data (per-store_id HDF5 outputs)
      - Step 3: extract timestamps/signal, z-score/dFF, time corrections
      - Step 4: compute PSTH and cross-correlation with compute_corr=True

    Notes:
      - Cross-correlation requires at least two distinct signal recording sites.
      - getCorrCombinations returns np.unique(["dls", "dms"]) → ["dls", "dms"],
        so the output file is corr_port_entries_z_score_dls_dms.h5.
    """
    session_subdir = "tdt/Photo_63_207-181030-103332"
    store_id_to_store_label = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "Dv3B": "control_dls",
        "Dv4B": "signal_dls",
        "PrtN": "port_entries",
    }
    src_base_dir = str(STUBBED_TESTING_DATA)
    src_session = Path(src_base_dir) / session_subdir
    assert Path(src_session).is_dir(), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = Path(src_session).name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session
    for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
        assert Path(d).is_dir(), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    step1(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        store_id_to_store_label=store_id_to_store_label,
    )
    selected_runs = {str(session_copy): ["1"]}
    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
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
        compute_corr=True,
        selected_runs=selected_runs,
    )

    # Locate output directory
    basename = Path(session_copy).name
    run_folders = sorted(list(Path(session_copy).glob(f"{basename}_output_*")))
    assert run_folders, f"No output directories found in {session_copy}"
    out_dir = None
    for d in run_folders:
        if (Path(d) / "storesList.csv").exists():
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"

    # Standard PSTH outputs for both recording sites
    for recording_site in ("dms", "dls"):
        freq_amp_h5 = Path(out_dir) / (f"freqAndAmp_z_score_{recording_site}.h5")
        freq_amp_csv = Path(out_dir) / (f"freqAndAmp_z_score_{recording_site}.csv")
        trans_occ_csv = Path(out_dir) / (f"transientsOccurrences_z_score_{recording_site}.csv")
        assert Path(freq_amp_h5).exists(), f"Missing freq/amp HDF5: {freq_amp_h5}"
        assert Path(freq_amp_csv).exists(), f"Missing freq/amp CSV: {freq_amp_csv}"
        assert Path(trans_occ_csv).exists(), f"Missing transients occurrences CSV: {trans_occ_csv}"

    # Cross-correlation outputs
    corr_dir = Path(out_dir) / "cross_correlation_output"
    assert Path(corr_dir).is_dir(), f"Missing cross_correlation_output directory: {corr_dir}"
    corr_h5 = Path(corr_dir) / "corr_port_entries_z_score_dls_dms.h5"
    assert Path(corr_h5).exists(), f"Missing cross-correlation HDF5: {corr_h5}"
    df = pd.read_hdf(corr_h5, key="df")
    assert "timestamps" in df.columns, f"'timestamps' column missing in {corr_h5}"
    assert "mean" in df.columns, f"'mean' column missing in {corr_h5}"

    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step5(
                base_dir=str(tmp_base),
                selected_folders=[str(session_copy)],
                selected_runs=selected_runs,
            )

    assert len(captured_dashboards) >= 1, "step5 created no VisualizationDashboard instances"
