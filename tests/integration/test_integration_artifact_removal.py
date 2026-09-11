import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from bokeh.document import Document
from bokeh.io.doc import set_curdoc

from guppy.testing.api import (
    remove_artifacts,
    select_artifact_windows,
    step1,
    step2,
    step3,
    step4,
)
from guppy_test_data import STUBBED_TESTING_DATA as TESTING_DATA

SESSION_SUBDIR = "tdt/Photo_048_392-200728-121222"
STORE_ID_TO_STORE_LABEL = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}

_COORDS_CONCATENATE = np.array(
    [
        [4.03172489, 0.0],
        [116.25228841, 0.0],
        [135.76890815, 0.0],
        [181.30768755, 0.0],
    ]
)

_COORDS_NAN = np.array(
    [
        [4.03172489, 0.0],
        [116.25228841, 0.0],
        [135.76890815, 0.0],
        [181.30768755, 0.0],
    ]
)


@pytest.fixture(autouse=True)
def fresh_bokeh_document():
    """Reset Bokeh's thread-local document before each test.

    build_homepage() creates a pn.template.BootstrapTemplate, which reads
    pn.state.curdoc. On a second call within the same xdist worker the old
    document may be stale or None, triggering an AttributeError inside Panel.
    Setting a fresh Document() before each test prevents that state leak.
    """
    set_curdoc(Document())
    yield


@pytest.mark.parametrize(
    "artifact_removal_method, coords",
    [
        ("concatenate", _COORDS_CONCATENATE),
        ("replace with NaN", _COORDS_NAN),
    ],
    ids=["concatenate", "nan"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_artifact_removal(tmp_path, artifact_removal_method, coords):
    """
    Integration test for artifact removal.

    Runs the full pipeline (Steps 2–5) with artifact removal enabled and asserts
    that key output files are created.
    """
    src_session = TESTING_DATA / SESSION_SUBDIR
    assert src_session.is_dir(), f"Sample data not available at expected path: {src_session}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = src_session.name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
        assert Path(d).is_dir(), f"Expected output directory for cleanup, got non-directory: {d}"
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )
    selected_runs = {str(session_copy): ["1"]}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    select_artifact_windows(
        **common_kwargs,
        artifact_coords={"dms": coords},
        artifact_removal_method=artifact_removal_method,
        selected_runs=selected_runs,
    )
    remove_artifacts(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    run_folders = sorted(list(Path(session_copy).glob(f"{dest_name}_output_*")))
    assert run_folders, f"No output directories found in {session_copy}"
    out_dir = None
    for d in run_folders:
        if (Path(d) / "storesList.csv").exists():
            out_dir = d
            break
    assert out_dir is not None, f"No storesList.csv found in any output directory under {session_copy}"

    assert (Path(out_dir) / "storesList.csv").exists(), "Missing storesList.csv"

    timecorr = Path(out_dir) / "timeCorrection_dms.hdf5"
    assert Path(timecorr).exists(), f"Missing {timecorr}"
    with h5py.File(timecorr, "r") as f:
        assert "timestampNew" in f, f"Expected 'timestampNew' dataset in {timecorr}"

    ttl_fp = Path(out_dir) / "port_entries_dms_dms.hdf5"
    assert Path(ttl_fp).exists(), f"Missing TTL-aligned file {ttl_fp}"
    with h5py.File(ttl_fp, "r") as f:
        assert "ts" in f, f"Expected 'ts' dataset in {ttl_fp}"

    zscore_fp = Path(out_dir) / "z_score_dms.hdf5"
    assert Path(zscore_fp).exists(), f"Missing processed signal file {zscore_fp}"
