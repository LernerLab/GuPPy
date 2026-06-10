import json

import panel as pn
import pytest

from guppy.orchestration.home import build_homepage
from guppy.orchestration.save_parameters import save_parameters

EXPECTED_JSON_KEYS = {
    "guppy_version",
    "combine_data",
    "isosbestic_control",
    "timeForLightsTurnOn",
    "filter_window",
    "removeArtifacts",
    "artifactsRemovalMethod",
    "noChannels",
    "zscore_method",
    "baselineWindowStart",
    "baselineWindowEnd",
    "nSecPrev",
    "nSecPost",
    "computeCorr",
    "timeInterval",
    "bin_psth_trials",
    "use_time_or_trials",
    "baselineCorrectionStart",
    "baselineCorrectionEnd",
    "peak_startPoint",
    "peak_endPoint",
    "selectForComputePsth",
    "selectForTransientsComputation",
    "moving_window",
    "highAmpFilt",
    "transientsThresh",
    "plot_zScore_dff",
    "visualize_zscore_or_dff",
    "averageForGroup",
}


@pytest.fixture(scope="session")
def panel_extension():
    pn.extension()


@pytest.fixture
def homepage(panel_extension):
    return build_homepage()


def test_save_parameters_writes_parameters_json(homepage, tmp_path):
    session_directory = tmp_path / "session1"
    session_directory.mkdir()
    homepage._widgets["files_1"].value = [str(session_directory)]
    save_parameters(homepage._hooks["getInputParameters"]())
    assert (session_directory / "GuPPyParamtersUsed.json").exists()


def test_parameters_json_contains_expected_keys(homepage, tmp_path):
    session_directory = tmp_path / "session1"
    session_directory.mkdir()
    homepage._widgets["files_1"].value = [str(session_directory)]
    save_parameters(homepage._hooks["getInputParameters"]())
    with open(session_directory / "GuPPyParamtersUsed.json") as json_file:
        saved_parameters = json.load(json_file)
    assert set(saved_parameters.keys()) == EXPECTED_JSON_KEYS


def test_get_input_parameters_keys_include_saved_keys(homepage, tmp_path):
    session_directory = tmp_path / "session1"
    session_directory.mkdir()
    homepage._widgets["files_1"].value = [str(session_directory)]
    save_parameters(homepage._hooks["getInputParameters"]())
    with open(session_directory / "GuPPyParamtersUsed.json") as json_file:
        saved_parameters = json.load(json_file)
    in_memory_parameters = homepage._hooks["getInputParameters"]()
    for key in saved_parameters:
        if key == "guppy_version":
            continue
        assert key in in_memory_parameters
