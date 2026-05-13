import json
import logging
import os
from importlib.metadata import version

from guppy.utils.utils import discover_output_dirs, select_output_dirs

logger = logging.getLogger(__name__)


def save_parameters(inputParameters: dict[str, object]) -> None:
    """
    Write the analysis configuration JSON to each selected output directory.

    For every session listed under ``inputParameters['folderNames']`` the
    configuration is written into each output directory selected by the
    ``selectedOutputs`` filter. When a session has no output directories yet
    (e.g. the user clicked step 1 before step 2), the file is written at the
    session root as a fallback so the legacy ordering still works.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; a subset of keys is written to
        ``GuPPyParamtersUsed.json``.
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
    selected_outputs = inputParameters.get("selectedOutputs") or {}
    for session in inputParameters["folderNames"]:
        # Fall back to the session root when no output dirs exist yet so step 1
        # can still run before step 2 (legacy ordering).
        if not discover_output_dirs(session):
            destinations = [session]
        else:
            destinations = select_output_dirs(session, selected_outputs.get(session))
        for destination in destinations:
            with open(os.path.join(destination, "GuPPyParamtersUsed.json"), "w") as f:
                json.dump(analysisParameters, f, indent=4)
            logger.info(f"Input Parameters file saved at {destination}")

    logger.info("#" * 400)
    logger.info("Input Parameters File Saved.")
