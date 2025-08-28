import os
import glob
import shutil

import h5py
import pytest

from guppy.testing.api import step2, step3, step4

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
def test_step4(tmp_path, monkeypatch, region, ttl_display_name, storenames_map):
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
    # Use the CSV sample session
    src_base_dir = "/Users/pauladkisson/Documents/CatalystNeuro/Guppy/GDriveSampleData"
    src_session = os.path.join(src_base_dir, "SampleData_csv")
    if not os.path.isdir(src_session):
        pytest.skip(f"Sample data not available at expected path: {src_session}")

    # Stub matplotlib.pyplot.show to avoid GUI blocking
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

    # Step 2: create storesList.csv in the temp copy
    step2(base_dir=str(tmp_base), selected_folders=[str(session_copy)], storenames_map=storenames_map)

    # Step 3: read raw data in the temp copy
    step3(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

    # Step 4: extract timestamps and signal in the temp copy
    step4(base_dir=str(tmp_base), selected_folders=[str(session_copy)])

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
    timecorr = os.path.join(out_dir, f"timeCorrection_{region}.hdf5")
    assert os.path.exists(timecorr), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    # If TTLs exist, check their per-region 'ts' outputs
    ttl_fp = os.path.join(out_dir, f"{ttl_display_name}_{region}.hdf5")
    assert os.path.exists(ttl_fp), f"Missing TTL-aligned file {ttl_fp}"
    with h5py.File(ttl_fp, "r") as f:
        assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"
