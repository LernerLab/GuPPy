"""Step 7 orchestration: export selected GuPPy sessions to NWB.

Drives the neuroconv :class:`TDTFiberPhotometryGuppyConverter` for each selected
``(session, run)`` pair, merging the converter's auto-filled metadata with the
session-level YAML overlay produced by Step 6, then writing one NWB file per
output directory.
"""

import logging
import os

import panel as pn
from neuroconv.converters import TDTFiberPhotometryGuppyConverter
from neuroconv.utils import dict_deep_update, load_dict_from_file

from .metadata import METADATA_FILENAME, _selected_session_runs
from ..utils.utils import output_dir_for_run

logger = logging.getLogger(__name__)


def _prune_absent_commanded_voltage(metadata: dict, available_streams: set[str]) -> None:
    """Drop CommandedVoltageSeries (and table refs) whose TDT stream is absent.

    Some tanks expose a fiber-photometry stream the metadata template assumes
    (e.g. ``Fi1d``) under a different name (e.g. ``Fi1r``). Rather than fail, drop
    the CommandedVoltageSeries that reference unavailable streams, mirroring the
    converter test's robustness guard but keyed to the streams actually present.
    """
    fiber_photometry = metadata.get("Ophys", {}).get("FiberPhotometry", {})
    commanded_voltage_series = fiber_photometry.get("CommandedVoltageSeries")
    if not commanded_voltage_series:
        return

    kept = [series for series in commanded_voltage_series if series.get("stream_name") in available_streams]
    kept_names = {series["name"] for series in kept}
    if kept:
        fiber_photometry["CommandedVoltageSeries"] = kept
    else:
        fiber_photometry.pop("CommandedVoltageSeries", None)

    table = fiber_photometry.get("FiberPhotometryTable", {})
    for row in table.get("rows", []):
        if row.get("commanded_voltage_series") not in kept_names:
            row.pop("commanded_voltage_series", None)


def export_session_to_nwb(
    *,
    tdt_folder_path: str,
    guppy_folder_path: str,
    metadata_yaml_path: str | None,
    nwbfile_path: str,
) -> str:
    """Convert one GuPPy session/run to NWB.

    Parameters
    ----------
    tdt_folder_path : str
        Path to the raw TDT tank (the session folder).
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
    """
    converter = TDTFiberPhotometryGuppyConverter(
        tdt_folder_path=tdt_folder_path,
        guppy_folder_path=guppy_folder_path,
    )

    metadata = converter.get_metadata()
    if metadata_yaml_path and os.path.exists(metadata_yaml_path):
        metadata = dict_deep_update(metadata, load_dict_from_file(metadata_yaml_path))

    tdt_interface = converter.data_interface_objects["TDTFiberPhotometry"]
    available_streams = set(tdt_interface.load().streams.keys())
    _prune_absent_commanded_voltage(metadata, available_streams)

    converter.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=True,
    )

    logger.info(f"Wrote NWB file to {nwbfile_path}")
    return nwbfile_path


def orchestrate_export_nwb_page(inputParameters: dict[str, object], progress_bar: object = None) -> None:
    """Export every selected ``(session, run)`` to NWB, updating the progress bar.

    Runs synchronously in the caller's thread (like the visualization step), so
    the progress bar and notifications can be updated directly. One failed session
    is reported and skipped without aborting the rest of the batch.
    """
    pairs = _selected_session_runs(inputParameters)
    if progress_bar is not None:
        progress_bar.max = len(pairs)
        progress_bar.value = 0

    for index, (session_path, run_name) in enumerate(pairs, start=1):
        guppy_folder_path = output_dir_for_run(session_path, run_name)
        session_basename = os.path.basename(session_path.rstrip(os.sep))
        output_dir_name = os.path.basename(guppy_folder_path.rstrip(os.sep))
        metadata_yaml_path = os.path.join(guppy_folder_path, METADATA_FILENAME)
        # Name the file after the full output directory so exports from multiple runs/sessions
        # stay distinct and can be aggregated into one folder without renaming.
        nwbfile_path = os.path.join(guppy_folder_path, f"{output_dir_name}.nwb")

        try:
            export_session_to_nwb(
                tdt_folder_path=session_path,
                guppy_folder_path=guppy_folder_path,
                metadata_yaml_path=metadata_yaml_path,
                nwbfile_path=nwbfile_path,
            )
            if pn.state.notifications:
                pn.state.notifications.success(f"Exported {session_basename} ({run_name}) to NWB.")
        except Exception as exception:
            logger.error(f"NWB export failed for {session_basename} ({run_name}): {exception}")
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"NWB export failed for {session_basename} ({run_name}): {exception}", duration=0
                )
        finally:
            if progress_bar is not None:
                progress_bar.value = index
