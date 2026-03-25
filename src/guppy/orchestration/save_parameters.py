import json
import logging
import os

logger = logging.getLogger(__name__)


def save_parameters(inputParameters: dict):
    logger.debug("Saving Input Parameters file.")
    analysisParameters = {
        "combine_data": inputParameters["combine_data"],
        "isosbestic_control": inputParameters["isosbestic_control"],
        "timeForLightsTurnOn": inputParameters["timeForLightsTurnOn"],
        "filter_window": inputParameters["filter_window"],
        "removeArtifacts": inputParameters["removeArtifacts"],
        "noChannels": inputParameters["noChannels"],
        "zscore_method": inputParameters["zscore_method"],
        "baselineWindowStart": inputParameters["baselineWindowStart"],
        "baselineWindowEnd": inputParameters["baselineWindowEnd"],
        "nSecPrev": inputParameters["nSecPrev"],
        "nSecPost": inputParameters["nSecPost"],
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
    }
    for folder in inputParameters["folderNames"]:
        with open(os.path.join(folder, "GuPPyParamtersUsed.json"), "w") as f:
            json.dump(analysisParameters, f, indent=4)
        logger.info(f"Input Parameters file saved at {folder}")

    logger.info("#" * 400)
    logger.info("Input Parameters File Saved.")
