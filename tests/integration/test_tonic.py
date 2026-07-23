"""End-to-end test for tonic/basal fluorescence analysis (issue #210).

Runs step1 -> step2 -> step3 headlessly on the synthetic injection CSV session
(``stubbed_testing_data/csv/sample_data_csv_injection_1``), whose 465 nm signal has a
sustained step at the injection time (t=60 s). Tonic epoch windows are injected via the
``tonic_epochs`` kwarg (bypassing the interactive epoch page), and the resulting
``tonic_region.h5`` is checked for correct per-epoch means and the expected baseline ->
post-injection increase.
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd
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
FIT_WINDOW = (2, 55)  # pre-injection window for baseline-epoch control fitting
BASELINE_EPOCH = (2.0, 55.0)  # pre-injection
POST_EPOCH = (65.0, 115.0)  # post-injection


def _stubbed_data_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "stubbed_testing_data")


def _output_directory(session):
    return sorted(glob.glob(os.path.join(session, f"{SESSION_NAME}_output_*")))[0]


@pytest.fixture
def injection_session(tmp_path):
    source = os.path.join(_stubbed_data_root(), SESSION_SUBDIR)
    base_dir = str(tmp_path)
    session = os.path.join(base_dir, SESSION_NAME)
    shutil.copytree(source, session)

    step1(base_dir=base_dir, selected_folders=[session], store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    selected_runs = {session: [parse_run_name(_output_directory(session))]}
    step2(base_dir=base_dir, selected_folders=[session], selected_runs=selected_runs)
    return {"base_dir": base_dir, "session": session, "selected_runs": selected_runs}


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestTonicAnalysis:
    def test_step3_writes_tonic_means_per_epoch(self, injection_session):
        epochs = pd.DataFrame(
            {
                "label": ["baseline", "post"],
                "start": [BASELINE_EPOCH[0], POST_EPOCH[0]],
                "end": [BASELINE_EPOCH[1], POST_EPOCH[1]],
            }
        )
        step3(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            control_fit_window_mode="baseline epoch",
            control_fit_window_start=FIT_WINDOW[0],
            control_fit_window_end=FIT_WINDOW[1],
            tonic_epochs={"region": epochs},
            selected_runs=injection_session["selected_runs"],
        )

        output_directory = _output_directory(injection_session["session"])
        tonic_path = os.path.join(output_directory, "tonic_region.h5")
        assert os.path.exists(tonic_path)

        tonic = pd.read_hdf(tonic_path, key="df")
        assert list(tonic.index) == ["baseline", "post"]
        assert list(tonic.columns) == ["mean_zscore", "mean_dff"]

        # Cross-check the stored means against the pipeline's own preprocessed trace:
        # the orchestration must have loaded the right per-site files and averaged them.
        timestamps = np.asarray(read_hdf5("timeCorrection_region", output_directory, "timestampNew")).ravel()
        z_score = np.asarray(read_hdf5("z_score_region", output_directory, "data")).ravel()
        dff = np.asarray(read_hdf5("dff_region", output_directory, "data")).ravel()
        for label, (start, end) in {"baseline": BASELINE_EPOCH, "post": POST_EPOCH}.items():
            mask = (timestamps >= start) & (timestamps <= end)
            assert tonic.loc[label, "mean_zscore"] == pytest.approx(np.nanmean(z_score[mask]))
            assert tonic.loc[label, "mean_dff"] == pytest.approx(np.nanmean(dff[mask]))

        # Scientific sanity: the injection raises the signal, so post > baseline.
        assert tonic.loc["post", "mean_dff"] > tonic.loc["baseline", "mean_dff"]
        assert tonic.loc["post", "mean_zscore"] > tonic.loc["baseline", "mean_zscore"]

    def test_step3_without_tonic_writes_no_tonic_file(self, injection_session):
        step3(
            base_dir=injection_session["base_dir"],
            selected_folders=[injection_session["session"]],
            selected_runs=injection_session["selected_runs"],
        )
        output_directory = _output_directory(injection_session["session"])
        assert not os.path.exists(os.path.join(output_directory, "tonic_region.h5"))
