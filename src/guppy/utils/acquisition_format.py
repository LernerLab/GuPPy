"""What acquisition format a session holds, and what that implies for NWB export.

Shared by Step 6 (which needs to know whether the user must supply a session start time) and
Step 7 (which needs to tell the converter how to read the raw folder), so it lives apart from
either step's orchestration.
"""

from .utils import RAISE_ISSUE_URL
from ..extractors.detect_acquisition_formats import detect_trace_formats
from ..extractors.nwb_recording_extractor import _find_nwb_file

# The acquisition formats NWB export can read: every format GuPPy reads raw acquisition files for.
# ``"nwb"`` is not a converter format -- a session already in NWB is exported by adding the GuPPy
# outputs to the file it came from, rather than by bundling raw acquisition files.
SUPPORTED_ACQUISITION_FORMATS = ("tdt", "doric", "npm", "csv", "nwb")

# The formats whose raw files record the session's start time. Only a TDT tank's header always does;
# a session already in NWB carries one because NWB requires it.
_FORMATS_RECORDING_SESSION_START_TIME = ("tdt", "nwb")


def resolve_acquisition_format(session_folder_path: str) -> str:
    """Return the acquisition format to read a session's raw files with.

    Resolved from the formats supplying the session's photometry traces, so a session exports as
    whatever GuPPy processed it as. Event stores GuPPy's custom-event import wrote single-column
    CSVs for are read from those CSVs by the converter itself and do not enter this decision.

    Parameters
    ----------
    session_folder_path : str
        Path to the raw session folder.

    Returns
    -------
    str
        One of :data:`SUPPORTED_ACQUISITION_FORMATS`.

    Raises
    ------
    ValueError
        If the folder holds no acquisition data, or traces from more than one acquisition format.
    """
    detected_formats = detect_trace_formats(session_folder_path)

    if not detected_formats:
        raise ValueError(
            f"No acquisition data was found in '{session_folder_path}', so there is nothing to export "
            f"the GuPPy outputs alongside. NWB export reads the raw session folder the pipeline was run "
            f"on, which holds the traces and the events GuPPy processed."
        )

    if len(detected_formats) > 1:
        raise ValueError(
            f"NWB export does not support sessions whose traces come from more than one acquisition "
            f"system, and '{session_folder_path}' holds {sorted(detected_formats)}. A recording site's "
            f"stores are written onto a single shared timestamps vector, which two acquisition systems "
            f"do not share. If you need NWB export for such sessions, please raise an issue at "
            f"{RAISE_ISSUE_URL}."
        )

    return detected_formats.pop()


def resolve_session_source(*, session_folder_path: str, inputParameters: dict[str, object]) -> tuple[str, str | None]:
    """Return how a session's source must be read for NWB export.

    Parameters
    ----------
    session_folder_path : str
        Path to the session folder.
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    acquisition_format : str
        One of :data:`SUPPORTED_ACQUISITION_FORMATS`.
    nwb_source : str or None
        The NWB file the session was processed out of — a local path or a ``dandi://`` URI — or
        ``None`` for a session recorded in a raw acquisition format.

    Raises
    ------
    ValueError
        If the folder holds no acquisition data, or traces from more than one acquisition format.
    """
    # A DANDI session's folder is a local stand-in holding only GuPPy's outputs, so the source is the
    # remote asset the pipeline streamed rather than anything the folder can be inspected for.
    if inputParameters.get("mode") == "dandi":
        return "nwb", inputParameters["dandi_uri_map"][session_folder_path]

    acquisition_format = resolve_acquisition_format(session_folder_path)
    if acquisition_format == "nwb":
        return acquisition_format, _find_nwb_file(session_folder_path)
    return acquisition_format, None


def acquisition_supplies_session_start_time(*, session_folder_path: str, acquisition_format: str) -> bool:
    """Report whether the raw acquisition is guaranteed to record the session's start time.

    NWB requires a session start time. Where this returns ``False`` the user must supply the start
    time through the Step 6 metadata form, which overrides whatever the acquisition may have read.

    Parameters
    ----------
    session_folder_path : str
        Path to the raw session folder.
    acquisition_format : str
        The format the session was recorded in, as returned by :func:`resolve_acquisition_format`.

    Returns
    -------
    bool
        ``True`` for TDT tanks and for sessions already in NWB, ``False`` otherwise. A ``.doric``
        HDF5 export carries a creation timestamp only when the acquisition software wrote one, so
        Doric is not counted here.
    """
    return acquisition_format in _FORMATS_RECORDING_SESSION_START_TIME
