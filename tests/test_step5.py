import os
import glob
import shutil

import pytest
import pandas as pd

from guppy.testing.api import step2, step3, step4, step5


@pytest.fixture(scope="function")
def region():
    return "region"


@pytest.fixture(scope="function")
def ttl_display_name():
    return "ttl"


@pytest.fixture(scope="function")
def storenames_map(region, ttl_display_name):
    return {
        "Sample_Control_Channel": f"control_{region}",
        "Sample_Signal_Channel": f"signal_{region}",
        "Sample_TTL": f"{ttl_display_name}",
    }


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5(tmp_path, monkeypatch, region, ttl_display_name, storenames_map):
    """
    Full integration test for Step 5 (PSTH Computation) using real CSV sample data,
    isolated to a temporary workspace to avoid mutating shared sample data.

    Pipeline executed on a temp copy:
      - Step 2: save storenames (storesList.csv)
      - Step 3: read raw data (per-storename HDF5 outputs)
      - Step 4: extract timestamps/signal, z-score/dFF, time corrections
      - Step 5: compute PSTH and peak/AUC outputs

    Notes:
      - matplotlib plotting in earlier steps may use a GUI backend; stub plt.show() to avoid blocking.
      - Assertions confirm creation and basic readability of PSTH-related outputs from Step 5.
      - Defaults are used for input parameters; PSTH computation defaults to z_score.
    """
    # Use the CSV sample session
    src_base_dir = "/Users/pauladkisson/Documents/CatalystNeuro/Guppy/GDriveSampleData"
    src_session = os.path.join(src_base_dir, "SampleData_csv")
    if not os.path.isdir(src_session):
        pytest.skip(f"Sample data not available at expected path: {src_session}")

    # Stub matplotlib.pyplot.show to avoid GUI blocking (used in earlier steps)
    import matplotlib.pyplot as plt  # noqa: F401
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

    # Stage a clean copy of the session into a temporary workspace
    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    session_copy = tmp_base / "SampleData_csv"
    shutil.copytree(src_session, session_copy)

    # Remove any copied artifacts in the temp session
    for d in glob.glob(os.path.join(session_copy, "*_output_*")):
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    # Step 2: create storesList.csv in the temp copy with explicit naming
    step2(base_dir=str(tmp_base), selected_folders=[str(session_copy)], storenames_map=storenames_map)

    # Step 3: read raw data in the temp copy
    step3(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

    # Step 4: extract timestamps and signal in the temp copy
    step4(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

    # Step 5: compute PSTH in the temp copy (headless)
    step5(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

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
    assert os.path.exists(stores_fp), "Missing storesList.csv after Steps 2–5"

    # Expected PSTH outputs (defaults compute z_score PSTH)
    psth_h5 = os.path.join(out_dir, f"{ttl_display_name}_{region}_z_score_{region}.h5")
    psth_baseline_uncorr_h5 = os.path.join(out_dir, f"{ttl_display_name}_{region}_baselineUncorrected_z_score_{region}.h5")
    peak_auc_h5 = os.path.join(out_dir, f"peak_AUC_{ttl_display_name}_{region}_z_score_{region}.h5")
    peak_auc_csv = os.path.join(out_dir, f"peak_AUC_{ttl_display_name}_{region}_z_score_{region}.csv")

    # Assert file creation
    assert os.path.exists(psth_h5), f"Missing PSTH HDF5: {psth_h5}"
    assert os.path.exists(psth_baseline_uncorr_h5), f"Missing baseline-uncorrected PSTH HDF5: {psth_baseline_uncorr_h5}"
    assert os.path.exists(peak_auc_h5), f"Missing PSTH Peak/AUC HDF5: {peak_auc_h5}"
    assert os.path.exists(peak_auc_csv), f"Missing PSTH Peak/AUC CSV: {peak_auc_csv}"

    # Basic readability checks: PSTH HDF5 contains a DataFrame with expected columns
    df = pd.read_hdf(psth_h5, key="df")
    assert "timestamps" in df.columns, f"'timestamps' column missing in {psth_h5}"
    # The DataFrame should include a 'mean' column per create_Df implementation
    assert "mean" in df.columns, f"'mean' column missing in {psth_h5}"

    # Additional artifacts from transients frequency/amplitude computation (Step 5 side-effect)
    freq_amp_h5 = os.path.join(out_dir, f"freqAndAmp_z_score_{region}.h5")
    freq_amp_csv = os.path.join(out_dir, f"freqAndAmp_z_score_{region}.csv")
    trans_occ_csv = os.path.join(out_dir, f"transientsOccurrences_z_score_{region}.csv")
    assert os.path.exists(freq_amp_h5), f"Missing freq/amp HDF5: {freq_amp_h5}"
    assert os.path.exists(freq_amp_csv), f"Missing freq/amp CSV: {freq_amp_csv}"
    assert os.path.exists(trans_occ_csv), f"Missing transients occurrences CSV: {trans_occ_csv}"
