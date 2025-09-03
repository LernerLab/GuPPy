import json
import os

import numpy as np
import pytest

from guppy.testing.api import step1


@pytest.fixture(scope="function")
def default_parameters():
    return {
        "combine_data": False,
        "isosbestic_control": True,
        "timeForLightsTurnOn": 1,
        "filter_window": 100,
        "removeArtifacts": False,
        "noChannels": 2,
        "zscore_method": "standard z-score",
        "baselineWindowStart": 0,
        "baselineWindowEnd": 0,
        "nSecPrev": -10,
        "nSecPost": 20,
        "timeInterval": 2,
        "bin_psth_trials": 0,
        "use_time_or_trials": "Time (min)",
        "baselineCorrectionStart": -5,
        "baselineCorrectionEnd": 0,
        "peak_startPoint": [
            -5.0,
            0.0,
            5.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ],
        "peak_endPoint": [
            0.0,
            3.0,
            10.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ],
        "selectForComputePsth": "z_score",
        "selectForTransientsComputation": "z_score",
        "moving_window": 15,
        "highAmpFilt": 2,
        "transientsThresh": 3
    }


def test_step1(tmp_path, default_parameters):
    # Arrange: base directory with two sessions under the same parent
    session_names = ["session1", "session2"]
    base_name = "data_root"
    base_dir = tmp_path / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for name in session_names:
        path = base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        sessions.append(str(path))
    base_dir = str(base_dir)

    # Act: call actual Panel onclickProcess via the API helper (headless)
    step1(base_dir=base_dir, selected_folders=sessions)

    # Assert: JSON written for each session with key defaults
    for s in sessions:
        out_fp = os.path.join(s, "GuPPyParamtersUsed.json")
        assert os.path.exists(out_fp), f"Missing file: {out_fp}"
        with open(out_fp, "r") as f:
            data = json.load(f)

        # Check that JSON data matches default parameters
        for key, expected_value in default_parameters.items():
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(data[key], expected_value)
            elif isinstance(expected_value, list) and any(isinstance(x, float) and np.isnan(x) for x in expected_value):
                # Handle lists with NaN values
                actual = data[key]
                assert len(actual) == len(expected_value)
                for i, (a, e) in enumerate(zip(actual, expected_value)):
                    if np.isnan(e):
                        assert np.isnan(a) or a is None, f"Mismatch at index {i}: expected NaN, got {a}"
                    else:
                        assert a == e, f"Mismatch at index {i}: expected {e}, got {a}"
            else:
                assert data[key] == expected_value, f"Mismatch for {key}: expected {expected_value}, got {data[key]}"
