"""End-to-end tests for baseline-epoch control fitting (issue #200).

Runs step1 -> step2 -> step3 headlessly on the synthetic injection CSV session
(``stubbed_testing_data/csv/sample_data_csv_injection_1``), whose 465 nm signal has a
sustained step at the injection time (t=60 s) while the 405 nm isosbestic control does not.
Full-trace fitting absorbs that step; baseline-epoch fitting recovers the pre-injection
relationship and preserves the step in dF/F. One test also enables artifact removal so the
chunking x baseline-epoch interaction is exercised through the real pipeline.
"""

import glob
import os

import numpy as np
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.testing.api import step1, step2, step3
from guppy.utils.utils import parse_run_name

SESSION_NAME = "sample_data_csv_injection_1"
SESSION_SUBDIR = f"csv/{SESSION_NAME}"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
INJECTION_TIME = 60.0
FIT_WINDOW = (2, 55)  # pre-injection; starts after timeForLightsTurnOn trims t < ~1 s


def _stubbed_data_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "stubbed_testing_data")


def _output_directory(session):
    return sorted(glob.glob(os.path.join(session, f"{SESSION_NAME}_output_*")))[0]


@pytest.fixture
def injection_session(tmp_path):
    """Copy the injection session and run step1 + step2; return locators for step3."""
    import shutil

    source = os.path.join(_stubbed_data_root(), SESSION_SUBDIR)
    base_dir = str(tmp_path)
    session = os.path.join(base_dir, SESSION_NAME)
    shutil.copytree(source, session)

    step1(base_dir=base_dir, selected_folders=[session], store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    selected_runs = {session: [parse_run_name(_output_directory(session))]}
    step2(base_dir=base_dir, selected_folders=[session], selected_runs=selected_runs)
    return {"base_dir": base_dir, "session": session, "selected_runs": selected_runs}


def _run_step3(injection_session, *, mode, artifact_coords=None):
    step3(
        base_dir=injection_session["base_dir"],
        selected_folders=[injection_session["session"]],
        control_fit_window_mode=mode,
        control_fit_window_start=FIT_WINDOW[0],
        control_fit_window_end=FIT_WINDOW[1],
        remove_artifacts=artifact_coords is not None,
        artifact_removal_method="replace with NaN" if artifact_coords is not None else None,
        artifact_coords=artifact_coords,
        selected_runs=injection_session["selected_runs"],
    )
    output_directory = _output_directory(injection_session["session"])
    timestamps = read_hdf5("timeCorrection_region", output_directory, "timestampNew")
    dff = read_hdf5("dff_region", output_directory, "data")
    return timestamps, dff


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestBaselineEpochFit:
    def test_baseline_epoch_preserves_injection_step(self, injection_session):
        timestamps, dff = _run_step3(injection_session, mode="baseline epoch")
        pre = dff[(timestamps > 3) & (timestamps < 55)]
        post = dff[(timestamps > 65) & (timestamps < 115)]
        # Fit estimated on the pre-injection window -> pre-injection dF/F sits at ~0.
        assert np.abs(np.nanmean(pre)) < 1.0
        # Ground truth step is 40 on a fitted control of ~165, so dF/F ~ 24%. The fit does not
        # invert (post dF/F is clearly positive, not negative).
        assert 20.0 < np.nanmean(post) < 27.0

    def test_full_trace_absorbs_the_step(self, injection_session):
        timestamps, dff = _run_step3(injection_session, mode="full trace")
        post = dff[(timestamps > 65) & (timestamps < 115)]
        # The whole-trace fit absorbs the step, so the post-injection deflection is largely lost.
        assert np.nanmean(post) < 8.0

    def test_baseline_epoch_recovers_more_of_the_step_than_full_trace(self, injection_session):
        _, baseline_epoch_dff = _run_step3(injection_session, mode="baseline epoch")
        timestamps, full_trace_dff = _run_step3(injection_session, mode="full trace")
        mask = (timestamps > 65) & (timestamps < 115)
        # Baseline-epoch recovers the real step; full-trace masks it — a wide, unambiguous gap.
        assert np.nanmean(baseline_epoch_dff[mask]) - np.nanmean(full_trace_dff[mask]) > 12.0

    def test_baseline_epoch_with_artifact_removal(self, injection_session):
        # Good chunks [2, 58] (pre) and [62, 118] (post); the removed gap straddles the injection.
        # fetchCoords reads column 0 as ordered good-chunk boundaries.
        coords = np.array([[2.0, 0.0], [58.0, 0.0], [62.0, 0.0], [118.0, 0.0]])
        timestamps, dff = _run_step3(injection_session, mode="baseline epoch", artifact_coords={"region": coords})

        gap = dff[(timestamps > 58.5) & (timestamps < 61.5)]
        assert gap.size > 0 and np.all(np.isnan(gap))

        # Coefficients estimated from the retained pre-injection window carry into the
        # post-injection chunk, so the step is preserved despite the chunking.
        post = dff[(timestamps > 65) & (timestamps < 115)]
        assert 20.0 < np.nanmean(post) < 27.0

    def test_baseline_epoch_requires_isosbestic_control(self, injection_session):
        with pytest.raises(ValueError, match="requires an isosbestic control"):
            step3(
                base_dir=injection_session["base_dir"],
                selected_folders=[injection_session["session"]],
                isosbestic_control=False,
                control_fit_window_mode="baseline epoch",
                control_fit_window_start=FIT_WINDOW[0],
                control_fit_window_end=FIT_WINDOW[1],
                selected_runs=injection_session["selected_runs"],
            )
