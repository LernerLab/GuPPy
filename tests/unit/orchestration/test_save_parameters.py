import json
import os
from importlib.metadata import version

import pytest

from guppy.orchestration.save_parameters import save_parameters

PARAMETER_KEYS = {
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

EXPECTED_KEYS = PARAMETER_KEYS | {"guppy_version"}

ORCHESTRATION_ONLY_KEYS = {
    "folderNames",
    "step",
    "numberOfCores",
    "storenames_map",
    "mode",
    "dandi_uri_map",
    "abspath",
    "folderNamesForAvg",
    "visualizeAverageResults",
}


@pytest.fixture
def base_input_parameters(tmp_path):
    folder = tmp_path / "session1"
    folder.mkdir()
    return {
        "folderNames": [str(folder)],
        "combine_data": False,
        "isosbestic_control": True,
        "timeForLightsTurnOn": 5.0,
        "filter_window": 100,
        "removeArtifacts": False,
        "artifactsRemovalMethod": "concatenate",
        "noChannels": 2,
        "zscore_method": "standard",
        "baselineWindowStart": 0.0,
        "baselineWindowEnd": 2.0,
        "nSecPrev": 5,
        "nSecPost": 10,
        "computeCorr": False,
        "timeInterval": 0.5,
        "bin_psth_trials": 10,
        "use_time_or_trials": "time",
        "baselineCorrectionStart": -2.0,
        "baselineCorrectionEnd": 0.0,
        "peak_startPoint": 0.0,
        "peak_endPoint": 5.0,
        "selectForComputePsth": "z_score",
        "selectForTransientsComputation": "z_score",
        "moving_window": 15,
        "highAmpFilt": 3.0,
        "transientsThresh": 2.0,
        "plot_zScore_dff": "z_score",
        "visualize_zscore_or_dff": "z_score",
        "averageForGroup": False,
        # orchestration-only keys that should not be saved
        "step": 0,
        "numberOfCores": 4,
        "storenames_map": {},
        "mode": "tdt",
        "dandi_uri_map": {},
        "abspath": "/tmp/abs",
        "folderNamesForAvg": [],
        "visualizeAverageResults": False,
    }


def test_save_parameters_writes_json_to_each_folder(tmp_path, base_input_parameters):
    second_folder = tmp_path / "session2"
    second_folder.mkdir()
    base_input_parameters["folderNames"].append(str(second_folder))

    save_parameters(base_input_parameters)

    for folder in base_input_parameters["folderNames"]:
        assert os.path.exists(os.path.join(folder, "GuPPyParamtersUsed.json"))


def test_save_parameters_saves_exactly_expected_keys(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["folderNames"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert set(saved.keys()) == EXPECTED_KEYS


def test_save_parameters_excludes_orchestration_keys(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["folderNames"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert ORCHESTRATION_ONLY_KEYS.isdisjoint(saved.keys())


def test_save_parameters_preserves_values(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["folderNames"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    for key in PARAMETER_KEYS:
        assert saved[key] == base_input_parameters[key]


def test_save_parameters_writes_guppy_version(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["folderNames"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert saved["guppy_version"] == version("guppy-neuro")


def test_save_parameters_single_folder(tmp_path):
    folder = tmp_path / "only_session"
    folder.mkdir()
    input_parameters = {
        "folderNames": [str(folder)],
        "combine_data": True,
        "isosbestic_control": False,
        "timeForLightsTurnOn": 0.0,
        "filter_window": 200,
        "removeArtifacts": True,
        "artifactsRemovalMethod": "replace with NaN",
        "noChannels": 1,
        "zscore_method": "baseline",
        "baselineWindowStart": 1.0,
        "baselineWindowEnd": 3.0,
        "nSecPrev": 2,
        "nSecPost": 8,
        "computeCorr": True,
        "timeInterval": 1.0,
        "bin_psth_trials": 5,
        "use_time_or_trials": "trials",
        "baselineCorrectionStart": -1.0,
        "baselineCorrectionEnd": 0.0,
        "peak_startPoint": 1.0,
        "peak_endPoint": 4.0,
        "selectForComputePsth": "dff",
        "selectForTransientsComputation": "dff",
        "moving_window": 20,
        "highAmpFilt": 5.0,
        "transientsThresh": 3.0,
        "plot_zScore_dff": "dff",
        "visualize_zscore_or_dff": "dff",
        "averageForGroup": True,
    }

    save_parameters(input_parameters)

    json_path = os.path.join(str(folder), "GuPPyParamtersUsed.json")
    assert os.path.exists(json_path)
    with open(json_path) as file:
        saved = json.load(file)
    assert saved["zscore_method"] == "baseline"
    assert saved["combine_data"] is True
