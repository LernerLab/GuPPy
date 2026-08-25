import json

import panel as pn
import pytest

from guppy.orchestration.home import build_homepage
from guppy.orchestration.save_parameters import save_parameters

# Written into the snapshot but not produced by the parameter form: the version stamp and
# the artifact-removal provenance recorded by the preprocessing steps.
NON_FORM_JSON_KEYS = {"guppy_version", "removeArtifacts", "artifactsRemovalMethod"}

EXPECTED_JSON_KEYS = {
    "guppy_version",
    "combine_data",
    "isosbestic_control",
    "control_fit_method",
    "controlFitWindowMode",
    "controlFitWindowStart",
    "controlFitWindowEnd",
    "photobleaching_detrend",
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
    "useTransientsAsEvents",
    "timeInterval",
    "bin_psth_trials",
    "use_time_or_trials",
    "baselineCorrectionStart",
    "baselineCorrectionEnd",
    "peak_startPoint",
    "peak_endPoint",
    "auc_units",
    "selectForComputePsth",
    "selectForTransientsComputation",
    "moving_window",
    "highAmpFilt",
    "transientsThresh",
    "computeBinnedMetrics",
    "binnedMetricsWidth",
    "visualize_zscore_or_dff",
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
        if key in NON_FORM_JSON_KEYS:
            continue
        assert key in in_memory_parameters


def test_derived_keys_are_recorded_but_absent_from_the_form(homepage, tmp_path):
    """The artifact keys are provenance written by the preprocessing steps, not form inputs."""
    session_directory = tmp_path / "session1"
    session_directory.mkdir()
    homepage._widgets["files_1"].value = [str(session_directory)]
    save_parameters(homepage._hooks["getInputParameters"]())
    with open(session_directory / "GuPPyParamtersUsed.json") as json_file:
        saved_parameters = json.load(json_file)

    in_memory_parameters = homepage._hooks["getInputParameters"]()
    for key in ("removeArtifacts", "artifactsRemovalMethod"):
        assert key in saved_parameters
        assert key not in in_memory_parameters
