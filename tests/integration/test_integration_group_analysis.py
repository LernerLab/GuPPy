import glob
import os
import shutil
from unittest.mock import patch

import holoviews as hv
import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step2, step3, step4, step5, step6

SESSION_SUBDIRS = [
    "tdt/Photo_048_392-200728-121222",
    "tdt/Photo_63_207-181030-103332",
]
STORENAMES_MAP = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}
EXPECTED_REGION = "dms"
EXPECTED_TTL = "port_entries_dms"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis(tmp_path):
    """
    Integration test: run the full pipeline (Steps 2-5) on two TDT sessions and then
    perform group-level averaging, asserting that the average directory and expected
    output files are created with the correct structure.
    """
    source_sessions = [STUBBED_TESTING_DATA / subdir for subdir in SESSION_SUBDIRS]
    for source_session in source_sessions:
        assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path / "data_root"
    temporary_base_directory.mkdir(parents=True, exist_ok=True)

    session_copies = []
    for source_session in source_sessions:
        session_name = source_session.name
        session_copy = temporary_base_directory / session_name
        shutil.copytree(source_session, session_copy)
        for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
            assert os.path.isdir(output_directory)
            shutil.rmtree(output_directory)
        parameters_path = session_copy / "GuPPyParamtersUsed.json"
        if parameters_path.exists():
            parameters_path.unlink()
        session_copies.append(session_copy)

    selected_folders = [str(session_copy) for session_copy in session_copies]
    base_dir = str(temporary_base_directory)

    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)

    step2(**common_kwargs, storenames_map=STORENAMES_MAP)
    step3(**common_kwargs)
    step4(**common_kwargs)
    step5(**common_kwargs)

    # Run group averaging pass
    step5(**common_kwargs, average_for_group=True, group_folders=selected_folders)

    average_directory = temporary_base_directory / "average"
    assert average_directory.is_dir(), f"No average directory found under {temporary_base_directory}"

    group_psth_file_path = os.path.join(
        average_directory,
        f"{EXPECTED_TTL}_{EXPECTED_REGION}_z_score_{EXPECTED_REGION}.h5",
    )
    assert os.path.exists(group_psth_file_path), f"Missing group PSTH HDF5: {group_psth_file_path}"

    group_psth_dataframe = pd.read_hdf(group_psth_file_path, key="df")
    assert "timestamps" in group_psth_dataframe.columns, f"'timestamps' column missing in {group_psth_file_path}"
    assert "mean" in group_psth_dataframe.columns, f"'mean' column missing in {group_psth_file_path}"

    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step6(base_dir=base_dir, selected_folders=selected_folders)

    assert len(captured_dashboards) >= 1, "step6 created no VisualizationDashboard instances"
