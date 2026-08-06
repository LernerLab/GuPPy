"""End-to-end test for tonic/basal fluorescence analysis (issue #210).

Runs step1 -> step2 -> step3 -> define_tonic_epochs headlessly on the synthetic injection
CSV session (``stubbed_testing_data/csv/sample_data_csv_injection_1``), whose 465 nm signal
walks through the three epochs of a bolus experiment: a flat baseline, a sustained plateau
from the injection at t=60 s, and an exponential clearance from t=160 s. Epoch windows are
written by ``define_tonic_epochs`` (bypassing the interactive epoch page), which also
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
from guppy.testing.api import define_tonic_epochs, step1, step2, step3
from guppy.utils.utils import parse_run_name

SESSION_NAME = "sample_data_csv_injection_1"
SESSION_SUBDIR = f"csv/{SESSION_NAME}"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
FIT_WINDOW = (2, 55)  # pre-injection window for baseline-epoch control fitting
# The three epochs of the bolus experiment. Each sits inside its phase, clear of the
# transitions at t=60 s (injection) and t=160 s (washout onset).
BASELINE_EPOCH = (2.0, 55.0)  # pre-injection
WASH_IN_EPOCH = (70.0, 155.0)  # drug on board, plateau
WASH_OUT_EPOCH = (200.0, 238.0)  # clearance largely complete


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
    def test_define_tonic_epochs_writes_means_per_epoch(self, injection_session):
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
        define_tonic_epochs(
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

        # The plateau carries the full ~24% step while the cleared epoch retains only a
        # fraction of it, so the recovery is unambiguous rather than a marginal dip.
        assert tonic.loc["wash_out", "mean_dff"] < 0.3 * tonic.loc["wash_in", "mean_dff"]

    def test_step3_alone_writes_no_tonic_file(self, injection_session):
        step3(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            selected_runs=injection_session["selected_runs"],
        )
        output_directory = _output_directory(injection_session["session"])
        assert not os.path.exists(os.path.join(output_directory, "tonic_region.h5"))
