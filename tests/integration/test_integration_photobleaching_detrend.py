import glob
import json
import os
import shutil

import numpy as np
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.testing.api import step1, step2, step3
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_SUBDIR = "csv/sample_data_csv_1"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
EXPECTED_RECORDING_SITE = "region"


@pytest.fixture
def run_preprocessing(tmp_path):
    """Return a callable that runs Steps 1-3 on a fresh copy of the stubbed CSV session.

    Each call gets its own workspace, so two runs configured differently can be compared
    against each other.
    """
    source_session = STUBBED_TESTING_DATA / SESSION_SUBDIR
    assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    def _run(workspace_name, **step3_kwargs):
        temporary_base_directory = tmp_path / workspace_name
        temporary_base_directory.mkdir(parents=True, exist_ok=True)
        session_name = source_session.name
        session_copy = temporary_base_directory / session_name
        shutil.copytree(source_session, session_copy)

        for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
            shutil.rmtree(output_directory)
        parameters_path = session_copy / "GuPPyParamtersUsed.json"
        if parameters_path.exists():
            parameters_path.unlink()

        common_kwargs = dict(base_dir=str(temporary_base_directory), selected_folders=[str(session_copy)])
        selected_runs = {str(session_copy): ["1"]}

        step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
        step2(**common_kwargs, selected_runs=selected_runs)
        step3(**common_kwargs, selected_runs=selected_runs, **step3_kwargs)

        output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
        for candidate in output_directories:
            if os.path.exists(os.path.join(candidate, "storesList.csv")):
                return candidate
        raise AssertionError(f"No storesList.csv found in any output directory under {session_copy}")

    return _run


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detrending_changes_the_fitted_control_and_the_dff(run_preprocessing):
    """The bleaching term is part of the control fit, so both the fit and the dF/F move."""
    plain_output = run_preprocessing("plain")
    detrended_output = run_preprocessing("detrended", photobleaching_detrend=True)

    def load(output_directory, prefix):
        return np.asarray(read_hdf5(f"{prefix}_{EXPECTED_RECORDING_SITE}", output_directory, "data")).ravel()

    plain_fit, detrended_fit = load(plain_output, "cntrl_sig_fit"), load(detrended_output, "cntrl_sig_fit")
    plain_dff, detrended_dff = load(plain_output, "dff"), load(detrended_output, "dff")

    assert plain_fit.shape == detrended_fit.shape
    # No separate trend file: the bleaching lives inside the control fit.
    assert not os.path.exists(os.path.join(detrended_output, f"photobleaching_trend_{EXPECTED_RECORDING_SITE}.hdf5"))
    assert np.abs(detrended_fit - plain_fit).max() > 0.0
    assert np.abs(detrended_dff - plain_dff).max() > 0.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detrending_choice_is_recorded_in_the_parameter_snapshot(run_preprocessing):
    output_directory = run_preprocessing("recorded", photobleaching_detrend=True)

    with open(os.path.join(output_directory, "GuPPyParamtersUsed.json")) as parameters_file:
        parameters = json.load(parameters_file)
    assert parameters["photobleaching_detrend"] is True


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detrending_without_isosbestic_control_raises(run_preprocessing):
    with pytest.raises(ValueError, match="requires an isosbestic control channel"):
        run_preprocessing("no_isosbestic", photobleaching_detrend=True, isosbestic_control=False)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detrending_with_baseline_epoch_fitting_raises(run_preprocessing):
    with pytest.raises(ValueError, match="cannot be combined with"):
        run_preprocessing(
            "baseline_epoch",
            photobleaching_detrend=True,
            control_fit_window_mode="baseline epoch",
            control_fit_window_start=1,
            control_fit_window_end=5,
        )
