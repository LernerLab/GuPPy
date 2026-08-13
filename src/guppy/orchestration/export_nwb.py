"""Step 7 orchestration: export selected GuPPy sessions to NWB.

Each selected ``(session, run)`` pair is written to one NWB file in its output directory, by
whichever of two routes the session's source calls for. A session recorded in a raw acquisition
format is bundled with its GuPPy outputs by the neuroconv :class:`GuppyConverter`, with the
converter's auto-filled metadata merged with the session-level YAML overlay produced by Step 6.
A session GuPPy processed out of an NWB file — a local file or a DANDI asset — is instead handed
to the standalone :class:`GuppyInterface`, which adds the GuPPy outputs to the file it came from.
"""

import logging
import os
from contextlib import closing

from pynwb import NWBFile
from pynwb.device import DeviceModel

from .metadata import METADATA_FILENAME
from .save_parameters import read_artifact_provenance
from ..extractors.dandi_nwb_recording_extractor import (
    _stream_nwb,
    is_dandi_uri,
    parse_dandi_uri,
)
from ..utils import progress
from ..utils.acquisition_format import resolve_session_source
from ..utils.nwb_io import open_nwbfile_io, write_nwbfile_from_source
from ..utils.progress import step_error_handler
from ..utils.utils import RAISE_ISSUE_URL, run_folder_for_run, selected_session_runs
from ..utils.validation import validate_data_not_combined

logger = logging.getLogger(__name__)

# The artifact-removal method that re-stamps surviving samples onto a fresh, continuous
# timeline, destroying the anchor to the raw acquisition clock. NWB export needs that anchor
# to align processed traces back to the source streams, so this method is unsupported here.
_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD = "concatenate"


def _validate_artifact_removal_methods(*, pairs: list[tuple[str, str]]) -> None:
    """Abort the NWB export batch if any selected run had its artifacts removed by ``concatenate``.

    ``concatenate`` re-times the kept samples onto a fresh timeline (see GuPPy issue #354),
    which is incompatible with NWB export's need to align processed traces to the acquisition
    clock. Reads each run's recorded artifact provenance (the state the data on disk was
    actually produced with) and raises before any file is written.

    Parameters
    ----------
    pairs : list of (str, str)
        ``(session_path, run_name)`` pairs selected for export.

    Raises
    ------
    ValueError
        If any selected run recorded ``removeArtifacts=True`` with
        ``artifactsRemovalMethod="concatenate"``.
    """
    offending = []
    for session_path, run_name in pairs:
        guppy_folder_path = run_folder_for_run(session_path, run_name)
        artifacts_removed, removal_method = read_artifact_provenance(destination=guppy_folder_path)
        if artifacts_removed and removal_method == _UNSUPPORTED_ARTIFACT_REMOVAL_METHOD:
            offending.append(f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})")

    if offending:
        raise ValueError(
            f"NWB export does not support the '{_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD}' artifact-removal "
            f"method because it re-times the kept samples onto a fresh timeline, breaking alignment to the "
            f"acquisition clock. The following session(s) were processed this way: {', '.join(offending)}. "
            f"Re-run Remove Artifacts after choosing 'replace with NaN' on the Select Artifact Windows page, "
            f"which preserves the original timeline; or re-run Step 3 (Preprocess) to drop the artifact "
            f"removal entirely. Then export again. "
            f"If you need '{_UNSUPPORTED_ARTIFACT_REMOVAL_METHOD}' support "
            f"for NWB export, please raise an issue at {RAISE_ISSUE_URL}."
        )


def _overlay_metadata_yaml(*, metadata: dict, metadata_yaml_path: str | None) -> dict:
    """Apply the session's ``nwb_metadata.yaml`` on top of auto-filled metadata, when it exists."""
    from neuroconv.utils import dict_deep_update, load_dict_from_file

    if metadata_yaml_path and os.path.exists(metadata_yaml_path):
        return dict_deep_update(metadata, load_dict_from_file(metadata_yaml_path))
    return metadata


def _adopt_orphan_device_models(*, nwbfile: NWBFile) -> None:
    """Give every device model read off a legacy file a place in ``nwbfile``.

    NWB 2.9 turned ``Device.model`` from a text attribute into a link to a ``DeviceModel``. Reading a
    file written before that, pynwb turns each ``model`` string into a ``DeviceModel`` container but
    never adds it to the file, so it has no parent and no path to link to; writing then fails. Each
    such model is added here, so the model strings survive the export.

    Devices agreeing on every model field share one container. Where a model name covers more than
    one set of fields, every model under that name is renamed after its device, since a name can
    only be claimed once and no device's metadata may stand in for another's.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The file to repair, in place. Devices whose model already has a parent are left alone, so
        this does nothing to a file written against a current schema.
    """

    def fields_of(model: DeviceModel) -> tuple:
        return (model.name, model.description, model.manufacturer, model.model_number)

    orphans = {
        device.name: device.model
        for device in nwbfile.devices.values()
        if device.model is not None and device.model.parent is None
    }
    fields_by_name = {}
    for model in orphans.values():
        fields_by_name.setdefault(model.name, set()).add(fields_of(model))
    ambiguous_names = {name for name, fields in fields_by_name.items() if len(fields) > 1}

    adopted = {}
    for device_name, model in orphans.items():
        model_fields = fields_of(model)
        device = nwbfile.devices[device_name]
        if model_fields in adopted:
            # Assigned through ``fields`` because ``Device.model`` is set once and rejects a second
            # assignment, and ``DeviceModel.name`` is read-only.
            device.fields["model"] = adopted[model_fields]
            continue
        if model.name in ambiguous_names:
            model = DeviceModel(
                name=f"{model.name} ({device_name})",
                description=model.description,
                manufacturer=model.manufacturer,
                model_number=model.model_number,
            )
            device.fields["model"] = model
        adopted[model_fields] = model
        nwbfile.add_device_model(model)


def _export_session_from_nwb_source(
    *,
    nwb_source: str,
    guppy_folder_path: str,
    metadata_yaml_path: str | None,
    nwbfile_path: str,
) -> str:
    """Add one GuPPy session/run's outputs to the NWB file it was processed out of.

    The source is read, the GuPPy outputs are added to it, and the result is written to
    ``nwbfile_path``; writing to a new path leaves the source untouched. GuPPy's store ids were
    derived from the source's own response series, so the recording sites registry links into the
    ``FiberPhotometryTable`` already there.

    The output stays on the extension versions the source was written with, so a session recorded
    against an older ndx-fiber-photometry exports as a file on that same version rather than one
    claiming a schema its contents do not match.
    """
    from neuroconv.datainterfaces import GuppyInterface

    interface = GuppyInterface(folder_path=guppy_folder_path)

    if is_dandi_uri(nwb_source):
        dandiset_id, asset_path = parse_dandi_uri(nwb_source)
        nwbfile, source_io, _counter = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
    else:
        source_io = open_nwbfile_io(path=nwb_source)
        nwbfile = source_io.read()

    _adopt_orphan_device_models(nwbfile=nwbfile)

    with closing(source_io):
        interface.add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=_overlay_metadata_yaml(metadata=interface.get_metadata(), metadata_yaml_path=metadata_yaml_path),
        )
        write_nwbfile_from_source(
            nwbfile=nwbfile,
            source_io=source_io,
            nwbfile_path=nwbfile_path,
            added_namespaces=("ndx-guppy",),
        )

    logger.info(f"Wrote NWB file to {nwbfile_path}")
    return nwbfile_path


def export_session_to_nwb(
    *,
    session_folder_path: str,
    acquisition_format: str,
    guppy_folder_path: str,
    metadata_yaml_path: str | None,
    nwbfile_path: str,
    nwb_source: str | None = None,
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
        present, on top of the auto-filled metadata.
    nwbfile_path : str
        Output path for the written ``.nwb`` file.
    nwb_source : str or None, optional
        The NWB file this session was processed out of — a local path or a ``dandi://`` URI. When
        given, the GuPPy outputs are added to that file rather than bundled with raw acquisition
        files, and ``acquisition_format`` is not used.

    Returns
    -------
    str
        The path of the written ``.nwb`` file.

    Raises
    ------
    ValueError
        If neither the raw acquisition nor the metadata overlay supplies a session start time.
    """
    if nwb_source is not None:
        return _export_session_from_nwb_source(
            nwb_source=nwb_source,
            guppy_folder_path=guppy_folder_path,
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=nwbfile_path,
        )

    # Imported here rather than at module scope because neuroconv is slow to import, and the GUI
    # imports this module at startup to wire up the Export button.
    from neuroconv.converters import GuppyConverter

    converter = GuppyConverter(
        fiber_photometry_folder_path=session_folder_path,
        events_folder_path=session_folder_path,
        guppy_folder_path=guppy_folder_path,
        acquisition_format=acquisition_format,
    )

    metadata = _overlay_metadata_yaml(metadata=converter.get_metadata(), metadata_yaml_path=metadata_yaml_path)

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
    pairs = selected_session_runs(inputParameters=inputParameters)
    validate_data_not_combined(combine_data=inputParameters["combine_data"])
    _validate_artifact_removal_methods(pairs=pairs)
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
            acquisition_format, nwb_source = resolve_session_source(
                session_folder_path=session_path, inputParameters=inputParameters
            )
            export_session_to_nwb(
                session_folder_path=session_path,
                acquisition_format=acquisition_format,
                guppy_folder_path=guppy_folder_path,
                metadata_yaml_path=metadata_yaml_path,
                nwbfile_path=nwbfile_path,
                nwb_source=nwb_source,
            )
            logger.info(f"Exported {session_basename} ({run_name}) to NWB.")
        except Exception as exception:
            # logger.exception so the traceback survives: the user-facing summary below is only the
            # message, and a partial-success batch is exactly the case where the log is all there is.
            logger.exception(f"NWB export failed for {session_basename} ({run_name})")
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
