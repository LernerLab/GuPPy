import json
import logging
import os
from importlib.metadata import version

logger = logging.getLogger(__name__)


def save_parameters(inputParameters: dict):
    """
    Write the analysis configuration JSON to each session folder.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; a subset of keys is written to
        ``GuPPyParamtersUsed.json`` inside each folder listed under
        ``inputParameters['folderNames']``.
    """
    logger.debug("Saving Input Parameters file.")
    analysisParameters = {
        "guppy_version": version("guppy-neuro"),
        "combine_data": inputParameters["combine_data"],
        "isosbestic_control": inputParameters["isosbestic_control"],
        "timeForLightsTurnOn": inputParameters["timeForLightsTurnOn"],
        "filter_window": inputParameters["filter_window"],
        "removeArtifacts": inputParameters["removeArtifacts"],
        "artifactsRemovalMethod": inputParameters["artifactsRemovalMethod"],
        "noChannels": inputParameters["noChannels"],
        "zscore_method": inputParameters["zscore_method"],
        "baselineWindowStart": inputParameters["baselineWindowStart"],
        "baselineWindowEnd": inputParameters["baselineWindowEnd"],
        "nSecPrev": inputParameters["nSecPrev"],
        "nSecPost": inputParameters["nSecPost"],
        "computeCorr": inputParameters["computeCorr"],
        "timeInterval": inputParameters["timeInterval"],
        "bin_psth_trials": inputParameters["bin_psth_trials"],
        "use_time_or_trials": inputParameters["use_time_or_trials"],
        "baselineCorrectionStart": inputParameters["baselineCorrectionStart"],
        "baselineCorrectionEnd": inputParameters["baselineCorrectionEnd"],
        "peak_startPoint": inputParameters["peak_startPoint"],
        "peak_endPoint": inputParameters["peak_endPoint"],
        "selectForComputePsth": inputParameters["selectForComputePsth"],
        "selectForTransientsComputation": inputParameters["selectForTransientsComputation"],
        "moving_window": inputParameters["moving_window"],
        "highAmpFilt": inputParameters["highAmpFilt"],
        "transientsThresh": inputParameters["transientsThresh"],
        "plot_zScore_dff": inputParameters["plot_zScore_dff"],
        "visualize_zscore_or_dff": inputParameters["visualize_zscore_or_dff"],
        "averageForGroup": inputParameters["averageForGroup"],
    }
    for folder in inputParameters["folderNames"]:
        with open(os.path.join(folder, "GuPPyParamtersUsed.json"), "w") as f:
            json.dump(analysisParameters, f, indent=4)
        logger.info(f"Input Parameters file saved at {folder}")

    logger.info("#" * 400)
    logger.info("Input Parameters File Saved.")
