import glob
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from guppy.testing.api import step2, step3, step4, step5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cross_correlation(tmp_path):
    """
    Integration test for Step 5 cross-correlation computation.

    Uses a two-region TDT session (DMS + DLS) with a generic TTL event to verify
    that cross-correlation output files are created when compute_corr=True.

    Pipeline executed on a temp copy:
      - Step 2: save storenames (storesList.csv)
      - Step 3: read raw data (per-storename HDF5 outputs)
      - Step 4: extract timestamps/signal, z-score/dFF, time corrections
      - Step 5: compute PSTH and cross-correlation with compute_corr=True

    Notes:
      - Cross-correlation requires at least two distinct signal regions.
      - getCorrCombinations returns np.unique(["dls", "dms"]) → ["dls", "dms"],
        so the output file is corr_port_entries_z_score_dls_dms.h5.
    """
    session_subdir = "SampleData_Clean/Photo_63_207-181030-103332"
    storenames_map = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "Dv3B": "control_dls",
        "Dv4B": "signal_dls",
        "PrtN": "port_entries",
    }
    src_base_dir = str(Path(".") / "testing_data")
    src_session = os.path.join(src_base_dir, session_subdir)
    assert os.path.isdir(src_session), f"Sample data not available at expected path: {src_session}"

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = os.path.basename(src_session)
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session
    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        assert os.path.isdir(d), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    step2(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        storenames_map=storenames_map,
    )
    step3(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )
    step4(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )
    step5(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
        compute_corr=True,
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

    # Standard PSTH outputs for both regions
    for region in ("dms", "dls"):
        freq_amp_h5 = os.path.join(out_dir, f"freqAndAmp_z_score_{region}.h5")
        freq_amp_csv = os.path.join(out_dir, f"freqAndAmp_z_score_{region}.csv")
        trans_occ_csv = os.path.join(out_dir, f"transientsOccurrences_z_score_{region}.csv")
        assert os.path.exists(freq_amp_h5), f"Missing freq/amp HDF5: {freq_amp_h5}"
        assert os.path.exists(freq_amp_csv), f"Missing freq/amp CSV: {freq_amp_csv}"
        assert os.path.exists(trans_occ_csv), f"Missing transients occurrences CSV: {trans_occ_csv}"

    # Cross-correlation outputs
    corr_dir = os.path.join(out_dir, "cross_correlation_output")
    assert os.path.isdir(corr_dir), f"Missing cross_correlation_output directory: {corr_dir}"
    corr_h5 = os.path.join(corr_dir, "corr_port_entries_z_score_dls_dms.h5")
    assert os.path.exists(corr_h5), f"Missing cross-correlation HDF5: {corr_h5}"
    df = pd.read_hdf(corr_h5, key="df")
    assert "timestamps" in df.columns, f"'timestamps' column missing in {corr_h5}"
    assert "mean" in df.columns, f"'mean' column missing in {corr_h5}"
