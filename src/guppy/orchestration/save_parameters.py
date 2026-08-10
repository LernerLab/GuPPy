import json
import logging
import os
from importlib.metadata import version

from guppy.utils.utils import discover_run_folders, select_run_folders

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_REMOVAL_METHOD = "replace with NaN"


def read_artifact_provenance(*, destination: str) -> tuple[bool, str]:
    """
    Read the recorded artifact-removal state of one output directory.

    Parameters
    ----------
    destination : str
        Output directory holding a ``GuPPyParamtersUsed.json`` snapshot.

    Returns
    -------
    remove_artifacts : bool
        Whether the Remove Artifacts step has been run on this directory.
    artifacts_removal_method : str
        The method recorded for this directory.
    """
    parameters_path = os.path.join(destination, "GuPPyParamtersUsed.json")
    if not os.path.exists(parameters_path):
        return False, DEFAULT_ARTIFACTS_REMOVAL_METHOD

    with open(parameters_path) as parameters_file:
        parameters = json.load(parameters_file)
    return (
        parameters.get("removeArtifacts", False),
        parameters.get("artifactsRemovalMethod", DEFAULT_ARTIFACTS_REMOVAL_METHOD),
    )


def record_artifact_provenance(
    *, destination: str, remove_artifacts: bool | None = None, artifacts_removal_method: str | None = None
) -> None:
    """
    Update the artifact-removal state recorded for one output directory.

    Rewrites only the artifact keys of the directory's existing snapshot, leaving every
    other recorded parameter untouched.

    Parameters
    ----------
    destination : str
        Output directory holding a ``GuPPyParamtersUsed.json`` snapshot.
    remove_artifacts : bool, optional
        New artifact-removal state. When None, the recorded value is kept.
    artifacts_removal_method : str, optional
        New artifact-removal method. When None, the recorded value is kept.
    """
    parameters_path = os.path.join(destination, "GuPPyParamtersUsed.json")
    with open(parameters_path) as parameters_file:
        parameters = json.load(parameters_file)

    recorded_removal, recorded_method = read_artifact_provenance(destination=destination)
    parameters["removeArtifacts"] = recorded_removal if remove_artifacts is None else remove_artifacts
    parameters["artifactsRemovalMethod"] = (
        recorded_method if artifacts_removal_method is None else artifacts_removal_method
    )

    with open(parameters_path, "w") as parameters_file:
        json.dump(parameters, parameters_file, indent=4)
    logger.info(f"Artifact provenance updated at {destination}")


def save_parameters(
    inputParameters: dict[str, object],
    *,
    remove_artifacts: bool | None = None,
    artifacts_removal_method: str | None = None,
) -> None:
    """
    Write the analysis configuration JSON to each selected output directory.

    For every session listed under ``inputParameters['session_folders']`` the
    configuration is written into each output directory selected by the
    ``selected_runs`` filter. When a session has no output directories yet
    (e.g. parameters are saved before Label Stores (Step 1) creates any output
    directories), the file is written at the session root as a fallback so the
    legacy ordering still works.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; a subset of keys is written to
        ``GuPPyParamtersUsed.json``.
    remove_artifacts : bool, optional
        Recorded artifact-removal state to write. When None, each destination
        keeps whatever it already recorded.
    artifacts_removal_method : str, optional
        Recorded artifact-removal method to write. When None, each destination
        keeps whatever it already recorded.
    """
    logger.debug("Saving Input Parameters file.")
    analysisParameters = {
        "guppy_version": version("guppy-neuro"),
        "combine_data": inputParameters["combine_data"],
        "isosbestic_control": inputParameters["isosbestic_control"],
        "control_fit_method": inputParameters["control_fit_method"],
        "controlFitWindowMode": inputParameters["controlFitWindowMode"],
        "controlFitWindowStart": inputParameters["controlFitWindowStart"],
        "controlFitWindowEnd": inputParameters["controlFitWindowEnd"],
        "photobleaching_detrend": inputParameters["photobleaching_detrend"],
        "timeForLightsTurnOn": inputParameters["timeForLightsTurnOn"],
        "filter_window": inputParameters["filter_window"],
        # Resolved per destination below; listed here to fix their position in the file.
        "removeArtifacts": None,
        "artifactsRemovalMethod": None,
        "noChannels": inputParameters["noChannels"],
        "zscore_method": inputParameters["zscore_method"],
        "baselineWindowStart": inputParameters["baselineWindowStart"],
        "baselineWindowEnd": inputParameters["baselineWindowEnd"],
        "nSecPrev": inputParameters["nSecPrev"],
        "nSecPost": inputParameters["nSecPost"],
        "computeCorr": inputParameters["computeCorr"],
        "useTransientsAsEvents": inputParameters["useTransientsAsEvents"],
        "timeInterval": inputParameters["timeInterval"],
        "bin_psth_trials": inputParameters["bin_psth_trials"],
        "use_time_or_trials": inputParameters["use_time_or_trials"],
        "baselineCorrectionStart": inputParameters["baselineCorrectionStart"],
        "baselineCorrectionEnd": inputParameters["baselineCorrectionEnd"],
        "peak_startPoint": inputParameters["peak_startPoint"],
        "peak_endPoint": inputParameters["peak_endPoint"],
        "auc_units": inputParameters["auc_units"],
        "selectForComputePsth": inputParameters["selectForComputePsth"],
        "selectForTransientsComputation": inputParameters["selectForTransientsComputation"],
        "moving_window": inputParameters["moving_window"],
        "highAmpFilt": inputParameters["highAmpFilt"],
        "transientsThresh": inputParameters["transientsThresh"],
        "computeBinnedMetrics": inputParameters["computeBinnedMetrics"],
        "binnedMetricsWidth": inputParameters["binnedMetricsWidth"],
        "visualize_zscore_or_dff": inputParameters["visualize_zscore_or_dff"],
        "averageForGroup": inputParameters["averageForGroup"],
    }
    selected_runs = inputParameters.get("selected_runs") or {}
    for session in inputParameters["session_folders"]:
        # Fall back to the session root when no output dirs exist yet so parameter
        # saving can still run before Label Stores (Step 1) creates output dirs.
        if not discover_run_folders(session):
            destinations = [session]
        else:
            destinations = select_run_folders(session, selected_runs.get(session))
        for destination in destinations:
            recorded_removal, recorded_method = read_artifact_provenance(destination=destination)
            destinationParameters = dict(analysisParameters)
            destinationParameters["removeArtifacts"] = (
                recorded_removal if remove_artifacts is None else remove_artifacts
            )
            destinationParameters["artifactsRemovalMethod"] = (
                recorded_method if artifacts_removal_method is None else artifacts_removal_method
            )
            with open(os.path.join(destination, "GuPPyParamtersUsed.json"), "w") as parameters_file:
                json.dump(destinationParameters, parameters_file, indent=4)
            logger.info(f"Input Parameters file saved at {destination}")

    logger.info("#" * 400)
    logger.info("Input Parameters File Saved.")
