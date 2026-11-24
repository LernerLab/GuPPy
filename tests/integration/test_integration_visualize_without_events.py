"""Step 5 on a session recorded without behavioral events.

Labels only the control and signal channels of the stubbed injection CSV session, so the
run has no event to build a PSTH from, then checks that Step 5 opens a dashboard for the
whole-session results the run does hold, and raises a clear error when it holds none.
"""

import shutil
from pathlib import Path

import holoviews as hv
import pandas as pd
import panel as pn
import pytest

from guppy.testing.api import (
    locate_run_folder,
    step1,
    step2,
    step3,
    step4,
    step5,
    tonic_analysis,
)
from guppy.utils import page_links
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_NAME = "sample_data_csv_injection_1"
# The session's TTL store is left unlabeled, so the run holds no behavioral event.
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
}
TONIC_EPOCHS = {"region": pd.DataFrame({"label": ["baseline", "wash_in"], "start": [2.0, 65.0], "end": [55.0, 115.0]})}


@pytest.fixture
def session_without_events(tmp_path):
    """Run Steps 1-4 on a copy of the injection session with its TTL store unlabeled."""
    session = tmp_path / SESSION_NAME
    shutil.copytree(STUBBED_TESTING_DATA / "csv" / SESSION_NAME, session, ignore=shutil.ignore_patterns("*_output_*"))
    common_kwargs = dict(base_dir=str(tmp_path), selected_folders=[str(session)])

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    selected_runs = {str(session): [parse_run_name(locate_run_folder(session=str(session)))]}
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs, compute_binned_metrics=True, binned_metrics_width=60)
    return {"common_kwargs": common_kwargs, "selected_runs": selected_runs, "session": session}


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestVisualizeWithoutEvents:
    def test_dashboard_shows_only_the_whole_session_tabs_with_results(self, session_without_events):
        tonic_analysis(
            **session_without_events["common_kwargs"],
            tonic_epochs=TONIC_EPOCHS,
            selected_runs=session_without_events["selected_runs"],
        )
        hv.extension("bokeh")

        with page_links.collect() as pages:
            step5(**session_without_events["common_kwargs"], selected_runs=session_without_events["selected_runs"])

        assert len(pages) == 1
        (tabs,) = pages[0].content.objects
        assert isinstance(tabs, pn.Tabs)
        assert tabs._names == ["Tonic", "Binned"]

    def test_raises_when_the_run_holds_nothing_to_visualize(self, session_without_events):
        output_directory = Path(locate_run_folder(session=str(session_without_events["session"])))
        for binned_metrics_path in output_directory.glob("binned_metrics_*.h5"):
            binned_metrics_path.unlink()

        with pytest.raises(ValueError, match="Nothing to visualize was found") as error:
            step5(**session_without_events["common_kwargs"], selected_runs=session_without_events["selected_runs"])

        assert str(output_directory) in str(error.value)
        assert "need behavioral events" in str(error.value)
