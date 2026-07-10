"""End-to-end Doric pipeline with sparse NaN in a control channel (issue #292).

Previously any NaN in a Doric signal/control channel was rejected at Step 2. The
reader now drops sparse non-finite samples jointly, so a channel with a handful
of NaNs flows all the way through to a z-score without a downstream
``filtfilt``/``polyfit`` error. A channel that is entirely NaN still fails fast.
"""

import glob
import os
import shutil

import h5py
import numpy as np
import pytest
from conftest import (
    REPRESENTATIVE_SESSIONS,
    STUBBED_TESTING_DATA,
    _locate_output_directory,
)

from guppy.testing.api import step1, step2, step3
from guppy.utils.utils import parse_run_name

_CONTROL_CHANNEL = "AIn-1 - Raw"


def _prepare_doric_copy(tmp_path):
    source_session = os.path.join(str(STUBBED_TESTING_DATA), REPRESENTATIVE_SESSIONS["doric"]["session_subdir"])
    session_copy = tmp_path / os.path.basename(source_session)
    shutil.copytree(source_session, session_copy)
    for output_directory in glob.glob(os.path.join(str(session_copy), "*_output_*")):
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()
    return str(tmp_path), str(session_copy)


def _inject_control_values(session_copy, values_by_index):
    doric_path = glob.glob(os.path.join(session_copy, "*.doric"))[0]
    with h5py.File(doric_path, "r+") as doric_file:
        dataset = doric_file["Traces"]["Console"][_CONTROL_CHANNEL][_CONTROL_CHANNEL]
        for index, value in values_by_index.items():
            dataset[index] = value


def _run_through_step3(base_directory, session_copy):
    store_id_to_store_label = REPRESENTATIVE_SESSIONS["doric"]["store_id_to_store_label"]
    step1(base_dir=base_directory, selected_folders=[session_copy], store_id_to_store_label=store_id_to_store_label)
    output_directory = _locate_output_directory(session_copy=session_copy)
    selected_runs = {session_copy: [parse_run_name(output_directory)]}
    step2(base_dir=base_directory, selected_folders=[session_copy], selected_runs=selected_runs)
    step3(base_dir=base_directory, selected_folders=[session_copy], selected_runs=selected_runs)
    return _locate_output_directory(session_copy=session_copy)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sparse_nan_control_reaches_zscore(tmp_path):
    base_directory, session_copy = _prepare_doric_copy(tmp_path)
    # A few scattered NaN samples in the control channel — the kind of sparse
    # dropout that used to abort Step 2 entirely.
    _inject_control_values(session_copy, {100: np.nan, 5000: np.nan, 123456: np.nan})

    output_directory = _run_through_step3(base_directory, session_copy)

    z_score_path = os.path.join(output_directory, "z_score_region.hdf5")
    assert os.path.exists(z_score_path), "Step 3 should produce a z-score despite sparse NaN in the control channel"
    with h5py.File(z_score_path, "r") as z_score_file:
        z_score = np.asarray(z_score_file["data"])
    assert z_score.size > 0
    assert np.isfinite(z_score).all(), "z-score must not contain NaN once sparse samples are dropped"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_nan_control_fails_fast(tmp_path):
    base_directory, session_copy = _prepare_doric_copy(tmp_path)
    doric_path = glob.glob(os.path.join(session_copy, "*.doric"))[0]
    with h5py.File(doric_path, "r+") as doric_file:
        dataset = doric_file["Traces"]["Console"][_CONTROL_CHANNEL][_CONTROL_CHANNEL]
        dataset[...] = np.nan

    store_id_to_store_label = REPRESENTATIVE_SESSIONS["doric"]["store_id_to_store_label"]
    step1(base_dir=base_directory, selected_folders=[session_copy], store_id_to_store_label=store_id_to_store_label)
    output_directory = _locate_output_directory(session_copy=session_copy)
    selected_runs = {session_copy: [parse_run_name(output_directory)]}
    with pytest.raises(ValueError, match="entirely non-finite"):
        step2(base_dir=base_directory, selected_folders=[session_copy], selected_runs=selected_runs)
