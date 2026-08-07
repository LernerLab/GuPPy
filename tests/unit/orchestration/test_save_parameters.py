import json
import os
from importlib.metadata import version

import pytest

from guppy.orchestration.save_parameters import (
    read_artifact_provenance,
    save_parameters,
)

# Derived provenance, recorded by the preprocessing steps rather than taken from the form.
ARTIFACT_PROVENANCE_KEYS = {
    "removeArtifacts",
    "artifactsRemovalMethod",
}

PARAMETER_KEYS = {
    "combine_data",
    "isosbestic_control",
    "control_fit_method",
    "controlFitWindowMode",
    "controlFitWindowStart",
    "controlFitWindowEnd",
    "photobleaching_detrend",
    "timeForLightsTurnOn",
    "filter_window",
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
    "auc_units",
    "selectForComputePsth",
    "selectForTransientsComputation",
    "moving_window",
    "highAmpFilt",
    "transientsThresh",
    "visualize_zscore_or_dff",
    "averageForGroup",
}

EXPECTED_KEYS = PARAMETER_KEYS | ARTIFACT_PROVENANCE_KEYS | {"guppy_version"}

ORCHESTRATION_ONLY_KEYS = {
    "session_folders",
    "step",
    "numberOfCores",
    "store_id_to_store_label",
    "mode",
    "dandi_uri_map",
    "abspath",
    "group_session_folders",
    "visualizeAverageResults",
}


@pytest.fixture
def base_input_parameters(tmp_path):
    folder = tmp_path / "session1"
    folder.mkdir()
    return {
        "session_folders": [str(folder)],
        "combine_data": False,
        "isosbestic_control": True,
        "control_fit_method": "IRWLS",
        "controlFitWindowMode": "full trace",
        "controlFitWindowStart": 0,
        "controlFitWindowEnd": 0,
        "photobleaching_detrend": False,
        "timeForLightsTurnOn": 5.0,
        "filter_window": 100,
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
        "auc_units": "samples",
        "selectForComputePsth": "z_score",
        "selectForTransientsComputation": "z_score",
        "moving_window": 15,
        "highAmpFilt": 3.0,
        "transientsThresh": 2.0,
        "visualize_zscore_or_dff": "z_score",
        "averageForGroup": False,
        # orchestration-only keys that should not be saved
        "step": 0,
        "numberOfCores": 4,
        "store_id_to_store_label": {},
        "mode": "tdt",
        "dandi_uri_map": {},
        "abspath": "/tmp/abs",
        "group_session_folders": [],
        "visualizeAverageResults": False,
    }


def test_save_parameters_writes_json_to_each_folder(tmp_path, base_input_parameters):
    second_folder = tmp_path / "session2"
    second_folder.mkdir()
    base_input_parameters["session_folders"].append(str(second_folder))

    save_parameters(base_input_parameters)

    for folder in base_input_parameters["session_folders"]:
        assert os.path.exists(os.path.join(folder, "GuPPyParamtersUsed.json"))


def test_save_parameters_saves_exactly_expected_keys(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["session_folders"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert set(saved.keys()) == EXPECTED_KEYS


def test_save_parameters_excludes_orchestration_keys(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["session_folders"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert ORCHESTRATION_ONLY_KEYS.isdisjoint(saved.keys())


def test_save_parameters_preserves_values(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["session_folders"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    for key in PARAMETER_KEYS:
        assert saved[key] == base_input_parameters[key]


def test_save_parameters_writes_guppy_version(base_input_parameters):
    save_parameters(base_input_parameters)

    folder = base_input_parameters["session_folders"][0]
    with open(os.path.join(folder, "GuPPyParamtersUsed.json")) as file:
        saved = json.load(file)

    assert saved["guppy_version"] == version("guppy-neuro")


def test_save_parameters_single_folder(tmp_path):
    folder = tmp_path / "only_session"
    folder.mkdir()
    input_parameters = {
        "session_folders": [str(folder)],
        "combine_data": True,
        "isosbestic_control": False,
        "control_fit_method": "OLS",
        "controlFitWindowMode": "full trace",
        "controlFitWindowStart": 0,
        "controlFitWindowEnd": 0,
        "photobleaching_detrend": False,
        "timeForLightsTurnOn": 0.0,
        "filter_window": 200,
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
        "auc_units": "seconds",
        "selectForComputePsth": "dff",
        "selectForTransientsComputation": "dff",
        "moving_window": 20,
        "highAmpFilt": 5.0,
        "transientsThresh": 3.0,
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


def _make_output_dir(session_path, run_name):
    run_folder = os.path.join(session_path, f"{os.path.basename(session_path)}_output_{run_name}")
    os.mkdir(run_folder)
    # storesList.csv must exist so select_run_folders accepts the run name.
    open(os.path.join(run_folder, "storesList.csv"), "w").close()
    return run_folder


def test_save_parameters_raises_when_filter_missing_for_session_with_output_dirs(base_input_parameters):
    session = base_input_parameters["session_folders"][0]
    _make_output_dir(session, "baseline")

    with pytest.raises(ValueError, match="explicit non-empty list"):
        save_parameters(base_input_parameters)


def test_save_parameters_filters_to_selected_run_name(base_input_parameters):
    session = base_input_parameters["session_folders"][0]
    baseline_dir = _make_output_dir(session, "baseline")
    strict_dir = _make_output_dir(session, "strict")
    base_input_parameters["selected_runs"] = {session: ["baseline"]}

    save_parameters(base_input_parameters)

    assert os.path.exists(os.path.join(baseline_dir, "GuPPyParamtersUsed.json"))
    assert not os.path.exists(os.path.join(strict_dir, "GuPPyParamtersUsed.json"))


def test_save_parameters_falls_back_to_session_root_when_no_output_dirs(base_input_parameters):
    """Save parameters before Label Stores (Step 1): no output dirs yet, so the file lands at the session root."""
    session = base_input_parameters["session_folders"][0]

    save_parameters(base_input_parameters)

    assert os.path.exists(os.path.join(session, "GuPPyParamtersUsed.json"))


def test_save_parameters_raises_for_unknown_selected_run(base_input_parameters):
    session = base_input_parameters["session_folders"][0]
    _make_output_dir(session, "baseline")
    base_input_parameters["selected_runs"] = {session: ["nonexistent"]}

    with pytest.raises(ValueError, match="Output directory not found"):
        save_parameters(base_input_parameters)


class TestArtifactProvenance:
    """Artifact-removal state is recorded per output directory, not taken from the parameter form."""

    def test_read_returns_defaults_when_no_snapshot_exists(self, tmp_path):
        assert read_artifact_provenance(destination=str(tmp_path)) == (False, "replace with NaN")

    def test_defaults_are_written_for_a_fresh_folder(self, base_input_parameters):
        save_parameters(base_input_parameters)

        folder = base_input_parameters["session_folders"][0]
        assert read_artifact_provenance(destination=folder) == (False, "replace with NaN")

    def test_explicit_values_are_recorded(self, base_input_parameters):
        save_parameters(base_input_parameters, remove_artifacts=True, artifacts_removal_method="concatenate")

        folder = base_input_parameters["session_folders"][0]
        assert read_artifact_provenance(destination=folder) == (True, "concatenate")

    def test_omitted_values_carry_forward(self, base_input_parameters):
        save_parameters(base_input_parameters, remove_artifacts=True, artifacts_removal_method="concatenate")

        save_parameters(base_input_parameters)

        folder = base_input_parameters["session_folders"][0]
        assert read_artifact_provenance(destination=folder) == (True, "concatenate")

    def test_method_carries_forward_while_removal_state_is_overwritten(self, base_input_parameters):
        """The marking page records the method; the Remove Artifacts step then records only the state."""
        save_parameters(base_input_parameters, artifacts_removal_method="concatenate")
        folder = base_input_parameters["session_folders"][0]
        assert read_artifact_provenance(destination=folder) == (False, "concatenate")

        save_parameters(base_input_parameters, remove_artifacts=True)

        assert read_artifact_provenance(destination=folder) == (True, "concatenate")
