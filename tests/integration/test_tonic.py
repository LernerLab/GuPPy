"""End-to-end test for tonic/basal fluorescence analysis (issue #210).

Runs step1 -> step2 -> step3 -> tonic_analysis headlessly on the synthetic injection
CSV session (``stubbed_testing_data/csv/sample_data_csv_injection_1``), whose 465 nm signal
holds three equal 60 s phases: a flat baseline, a sustained plateau from the injection at
t=60 s, and a clearance to a 25% residual from t=120 s. Epoch windows are
written by ``tonic_analysis`` (bypassing the interactive epoch page), which also
averages the traces over them; the resulting ``tonic_region.h5`` is checked for correct
per-epoch means and for the rise-then-recover ordering across the three epochs.
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.testing.api import step1, step2, step3, tonic_analysis
from guppy.utils.utils import parse_run_name

SESSION_NAME = "sample_data_csv_injection_1"
SESSION_SUBDIR = f"csv/{SESSION_NAME}"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
FIT_WINDOW = (2, 55)  # pre-injection window for baseline-epoch control fitting
# The session's three equal 60 s phases. Each window sits inside its phase, clear of the
# transitions at t=60 s (injection) and t=120 s (washout onset).
BASELINE_EPOCH = (2.0, 55.0)  # pre-injection
WASH_IN_EPOCH = (65.0, 115.0)  # drug on board, plateau
WASH_OUT_EPOCH = (145.0, 178.0)  # clearance settled onto its residual level


def _stubbed_data_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "stubbed_testing_data")


def _output_directory(session):
    return sorted(glob.glob(os.path.join(session, f"{SESSION_NAME}_output_*")))[0]


@pytest.fixture
def injection_session(tmp_path):
    source = os.path.join(_stubbed_data_root(), SESSION_SUBDIR)
    base_dir = str(tmp_path)
    session = os.path.join(base_dir, SESSION_NAME)
    # Output dirs are gitignored, so running GuPPy against the stubbed data leaves them
    # behind; copying them would seed this session with another run's results.
    shutil.copytree(source, session, ignore=shutil.ignore_patterns("*_output_*"))

    step1(base_dir=base_dir, selected_folders=[session], store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    selected_runs = {session: [parse_run_name(_output_directory(session))]}
    step2(base_dir=base_dir, selected_folders=[session], selected_runs=selected_runs)
    return {"base_dir": base_dir, "session": session, "selected_runs": selected_runs}


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestTonicAnalysis:
    def test_tonic_analysis_writes_means_per_epoch(self, injection_session):
        epochs = pd.DataFrame(
            {
                "label": ["baseline", "wash_in", "wash_out"],
                "start": [BASELINE_EPOCH[0], WASH_IN_EPOCH[0], WASH_OUT_EPOCH[0]],
                "end": [BASELINE_EPOCH[1], WASH_IN_EPOCH[1], WASH_OUT_EPOCH[1]],
            }
        )
        step3(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            control_fit_window_mode="baseline epoch",
            control_fit_window_start=FIT_WINDOW[0],
            control_fit_window_end=FIT_WINDOW[1],
            selected_runs=injection_session["selected_runs"],
        )
        # Epochs are defined on the traces Step 3 produced; the optional step averages
        # them on save, so no second preprocessing run is needed.
        tonic_analysis(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            tonic_epochs={"region": epochs},
            selected_runs=injection_session["selected_runs"],
        )

        output_directory = _output_directory(injection_session["session"])
        tonic_path = os.path.join(output_directory, "tonic_region.h5")
        assert os.path.exists(tonic_path)

        tonic = pd.read_hdf(tonic_path, key="df")
        assert list(tonic.index) == ["baseline", "wash_in", "wash_out"]
        assert list(tonic.columns) == ["mean_zscore", "mean_dff"]

        # Cross-check the stored means against the pipeline's own preprocessed trace:
        # the orchestration must have loaded the right per-site files and averaged them.
        timestamps = np.asarray(read_hdf5("timeCorrection_region", output_directory, "timestampNew")).ravel()
        z_score = np.asarray(read_hdf5("z_score_region", output_directory, "data")).ravel()
        dff = np.asarray(read_hdf5("dff_region", output_directory, "data")).ravel()
        epoch_windows = {"baseline": BASELINE_EPOCH, "wash_in": WASH_IN_EPOCH, "wash_out": WASH_OUT_EPOCH}
        for label, (start, end) in epoch_windows.items():
            mask = (timestamps >= start) & (timestamps <= end)
            assert tonic.loc[label, "mean_zscore"] == pytest.approx(np.nanmean(z_score[mask]))
            assert tonic.loc[label, "mean_dff"] == pytest.approx(np.nanmean(dff[mask]))

        # Scientific sanity: the drug raises the signal and clearance brings it back down,
        # so the three epochs must order wash_in > wash_out > baseline.
        for column in ("mean_dff", "mean_zscore"):
            assert tonic.loc["wash_in", column] > tonic.loc["wash_out", column]
            assert tonic.loc["wash_out", column] > tonic.loc["baseline", column]

        # Clearance leaves a 25% residual, so the washout epoch keeps roughly a quarter of
        # the plateau: bounded on both sides, since the point is that it is neither still
        # elevated nor all the way back onto the baseline.
        recovered_fraction = tonic.loc["wash_out", "mean_dff"] / tonic.loc["wash_in", "mean_dff"]
        assert 0.15 < recovered_fraction < 0.40

    def test_step3_alone_writes_no_tonic_file(self, injection_session):
        step3(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            selected_runs=injection_session["selected_runs"],
        )
        output_directory = _output_directory(injection_session["session"])
        assert not os.path.exists(os.path.join(output_directory, "tonic_region.h5"))
