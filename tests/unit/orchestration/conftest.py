import pytest


@pytest.fixture
def base_input_parameters():
    """Fully-populated inputParameters dict with all keys needed by preprocess and psth tests."""
    return {
        "folderNames": [],
        "step": 0,
        "numberOfCores": 1,
        "noChannels": 2,
        "isosbestic_control": True,
        "timeForLightsTurnOn": 5.0,
        "combine_data": False,
        "removeArtifacts": False,
        "artifactsRemovalMethod": "replace with NaN",
        "filter_window": 100,
        "zscore_method": "standard z-score",
        "baselineWindowStart": 0.0,
        "baselineWindowEnd": 2.0,
        "plot_zScore_dff": "z_score",
        "selectForComputePsth": "z_score",
        "nSecPrev": 5.0,
        "nSecPost": 10.0,
        "bin_psth_trials": 0,
        "use_time_or_trials": "trials",
        "baselineCorrectionStart": -2.0,
        "baselineCorrectionEnd": 0.0,
        "timeInterval": 0.1,
        "computeCorr": False,
        "peak_startPoint": 0.0,
        "peak_endPoint": 5.0,
        "storenames_map": {},
        "averageForGroup": False,
        "selectForTransientsComputation": "z_score",
        "moving_window": 15,
        "highAmpFilt": 3.0,
        "transientsThresh": 2.0,
    }
