"""Step 7 orchestration: export selected GuPPy sessions to NWB.

Drives the neuroconv :class:`GuppyConverter` for each selected ``(session, run)``
pair, merging the converter's auto-filled metadata with the session-level YAML
overlay produced by Step 6, then writing one NWB file per output directory. Which
interfaces read a session's raw folder follows from the acquisition format detected
in it, so a session exports as whatever the pipeline processed it as.
"""

import json
import logging
import os

from .metadata import METADATA_FILENAME, _selected_session_runs
from ..utils import progress
from ..utils.acquisition_format import resolve_acquisition_format
from ..utils.progress import step_error_handler
from ..utils.utils import RAISE_ISSUE_URL, run_folder_for_run

logger = logging.getLogger(__name__)

# The artifact-removal method that re-stamps surviving samples onto a fresh, continuous
# timeline, destroying the anchor to the raw acquisition clock. NWB export needs that anchor
# to align processed traces back to the source streams, so this method is unsupported here.
_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD = "concatenate"


def _validate_artifact_removal_methods(pairs: list[tuple[str, str]]) -> None:
    """Abort the NWB export batch if any selected session used ``concatenate`` artifact removal.

    ``concatenate`` re-times the kept samples onto a fresh timeline (see GuPPy issue #354),
    which is incompatible with NWB export's need to align processed traces to the acquisition
    clock. Reads each session's ``GuPPyParamtersUsed.json`` (the configuration the data was
    actually processed with) and raises before any file is written.

    Parameters
    ----------
    pairs : list of (str, str)
        ``(session_path, run_name)`` pairs selected for export.

    Raises
    ------
    ValueError
        If any selected session was processed with ``removeArtifacts=True`` and
        ``artifactsRemovalMethod="concatenate"``.
    """
    offending = []
    for session_path, run_name in pairs:
        guppy_folder_path = run_folder_for_run(session_path, run_name)
        with open(os.path.join(guppy_folder_path, "GuPPyParamtersUsed.json")) as parameters_file:
            parameters = json.load(parameters_file)
        if parameters.get("removeArtifacts") and (
            parameters.get("artifactsRemovalMethod") == _UNSUPPORTED_ARTIFACT_REMOVAL_METHOD
        ):
            offending.append(f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})")

    if offending:
        raise ValueError(
            f"NWB export does not support the '{_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD}' artifact-removal "
            f"method because it re-times the kept samples onto a fresh timeline, breaking alignment to the "
            f"acquisition clock. The following session(s) were processed this way: {', '.join(offending)}. "
            f"Re-run Step 3 (Preprocess and Remove Artifacts) with artifactsRemovalMethod='replace with NaN', "
            f"which preserves the original timeline, then export again. "
            f"If you need '{_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD}' support "
            f"for NWB export, please raise an issue at {RAISE_ISSUE_URL}."
        )


def _validate_local_mode(inputParameters: dict[str, object]) -> None:
    """Abort the NWB export batch if the pipeline was run against DANDI-streamed sessions.

    DANDI mode reads each session from a remote NWB asset rather than from raw acquisition files, so
    there is no raw source on disk to bundle with the GuPPy outputs. Raised before any file is
    written.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Raises
    ------
    ValueError
        If ``inputParameters["mode"]`` is ``"dandi"``.
    """
    if inputParameters.get("mode") == "dandi":
        raise ValueError(
            "NWB export does not support sessions read from DANDI. The export bundles a session's raw "
            "acquisition with its GuPPy outputs, and a DANDI session's source is an NWB file that was "
            "already written. If you need NWB export for DANDI-streamed sessions, please raise an issue "
            f"at {RAISE_ISSUE_URL}."
        )


def export_session_to_nwb(
    *,
    session_folder_path: str,
    acquisition_format: str,
    guppy_folder_path: str,
    metadata_yaml_path: str | None,
    nwbfile_path: str,
) -> str:
    """Convert one GuPPy session/run to NWB.

    Parameters
    ----------
    session_folder_path : str
        Path to the raw session folder. It holds both the fiber-photometry traces and the
        behavioral events.
    acquisition_format : str
        The format the session was recorded in, selecting which interfaces read the raw folder.
        One of :data:`~guppy.utils.acquisition_format.SUPPORTED_ACQUISITION_FORMATS`, as returned by
        :func:`~guppy.utils.acquisition_format.resolve_acquisition_format`.
    guppy_folder_path : str
        Path to the GuPPy ``<session>_output_<run>`` directory.
    metadata_yaml_path : str or None
        The session's metadata overlay (``nwb_metadata.yaml``). Applied, when
        present, on top of the converter's auto-filled metadata.
    nwbfile_path : str
        Output path for the written ``.nwb`` file.

    Returns
    -------
    str
        The path of the written ``.nwb`` file.

    Raises
    ------
    ValueError
        If neither the raw acquisition nor the metadata overlay supplies a session start time.
    """
    # Imported here rather than at module scope so the pure-Python prerequisite checks in this
    # module stay importable (and unit-testable) without the heavyweight neuroconv dependency.
    from neuroconv.converters import GuppyConverter
    from neuroconv.utils import dict_deep_update, load_dict_from_file

    converter = GuppyConverter(
        fiber_photometry_folder_path=session_folder_path,
        events_folder_path=session_folder_path,
        guppy_folder_path=guppy_folder_path,
        acquisition_format=acquisition_format,
    )

    metadata = converter.get_metadata()
    if metadata_yaml_path and os.path.exists(metadata_yaml_path):
        metadata = dict_deep_update(metadata, load_dict_from_file(metadata_yaml_path))

    # Only a TDT tank's header always records one, so for every other format the metadata form is the
    # only source. Checked here because pynwb's own failure names neither the session nor the step
    # that would fix it.
    if not metadata["NWBFile"].get("session_start_time"):
        raise ValueError(
            f"No session start time is available for '{session_folder_path}': the "
            f"'{acquisition_format}' raw files do not record one, so it must be supplied in "
            f"Step 6 (Input Metadata) before exporting."
        )

    converter.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=True,
    )

    logger.info(f"Wrote NWB file to {nwbfile_path}")
    return nwbfile_path


def orchestrate_export_nwb(inputParameters: dict[str, object]) -> None:
    """Export every selected ``(session, run)`` to NWB, reporting progress per session.

    One failed session is skipped without aborting the rest of the batch; if any session
    failed, the collected failures are reported through the progress channel once the batch
    ends.
    """
    pairs = _selected_session_runs(inputParameters)
    _validate_local_mode(inputParameters)
    _validate_artifact_removal_methods(pairs)
    progress.start(len(pairs))

    failures = []
    for session_path, run_name in pairs:
        guppy_folder_path = run_folder_for_run(session_path, run_name)
        session_basename = os.path.basename(session_path.rstrip(os.sep))
        output_dir_name = os.path.basename(guppy_folder_path.rstrip(os.sep))
        metadata_yaml_path = os.path.join(guppy_folder_path, METADATA_FILENAME)
        # Name the file after the full output directory so exports from multiple runs/sessions
        # stay distinct and can be aggregated into one folder without renaming.
        nwbfile_path = os.path.join(guppy_folder_path, f"{output_dir_name}.nwb")

        try:
            export_session_to_nwb(
                session_folder_path=session_path,
                acquisition_format=resolve_acquisition_format(session_path),
                guppy_folder_path=guppy_folder_path,
                metadata_yaml_path=metadata_yaml_path,
                nwbfile_path=nwbfile_path,
            )
            logger.info(f"Exported {session_basename} ({run_name}) to NWB.")
        except Exception as exception:
            logger.error(f"NWB export failed for {session_basename} ({run_name}): {exception}")
            failures.append(f"{session_basename} ({run_name}): {exception}")
        finally:
            progress.advance()

    # Reported rather than raised: the surviving sessions were exported successfully, so the
    # step is a partial success. The progress channel surfaces this on the server IOLoop,
    # which a notification from this background thread could not do.
    if failures:
        progress.fail(f"NWB export failed for {len(failures)} of {len(pairs)} session(s): " + "; ".join(failures))


@step_error_handler
def run_export_nwb_step(input_parameters: dict[str, object]) -> None:
    """Run step-7 NWB export with failure reporting attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    orchestrate_export_nwb(input_parameters)
