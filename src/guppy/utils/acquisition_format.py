"""What acquisition format a session holds, and what that implies for NWB export.

Shared by Step 6 (which needs to know whether the user must supply a session start time) and
Step 7 (which needs to tell the converter how to read the raw folder), so it lives apart from
either step's orchestration.
"""

from .utils import RAISE_ISSUE_URL
from ..extractors.detect_acquisition_formats import detect_acquisition_formats

# The acquisition formats NWB export can read.
SUPPORTED_ACQUISITION_FORMATS = ("tdt", "doric", "csv")

# The rest of what GuPPy reads, each with why the export cannot take it yet. Naming the reason beats a
# bare "unsupported": one of these is a data-correctness hold rather than missing plumbing.
_UNSUPPORTED_ACQUISITION_FORMATS = {
    "npm": (
        "GuPPy records a timestamp unit for a Neurophotometrics run that does not match the one it "
        "applied, so the exported timestamps would disagree with GuPPy's own by a factor of 1000. "
        "Tracked in https://github.com/LernerLab/GuPPy/issues/411."
    ),
    "nwb": (
        "A session read from an NWB file has no raw acquisition to bundle with the GuPPy outputs, and "
        "the converter has no interface for reading one back out."
    ),
}


def resolve_acquisition_format(session_folder_path: str) -> str:
    """Return the acquisition format to read a session's raw files with.

    Uses the same folder detection the rest of the pipeline dispatches on, so a session exports as
    whatever GuPPy processed it as.

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
        If the folder holds no acquisition data, more than one acquisition format, or a format NWB
        export cannot read.
    """
    detected_formats = detect_acquisition_formats(session_folder_path)

    if not detected_formats:
        raise ValueError(
            f"No acquisition data was found in '{session_folder_path}', so there is nothing to export "
            f"the GuPPy outputs alongside. NWB export reads the raw session folder the pipeline was run "
            f"on, which holds the traces and the events GuPPy processed."
        )

    if len(detected_formats) > 1:
        raise ValueError(
            f"NWB export does not support sessions holding more than one acquisition format, and "
            f"'{session_folder_path}' holds {sorted(detected_formats)}. The export reads a session's "
            f"traces and its events through a single format. Note that custom events are written as "
            f"single-column CSVs into the session folder, which makes the session a 'csv' source as "
            f"well. If you need NWB export for mixed-format sessions, please raise an issue at "
            f"{RAISE_ISSUE_URL}."
        )

    acquisition_format = detected_formats.pop()
    if acquisition_format not in SUPPORTED_ACQUISITION_FORMATS:
        raise ValueError(
            f"NWB export does not support the '{acquisition_format}' acquisition format, which is what "
            f"'{session_folder_path}' holds. {_UNSUPPORTED_ACQUISITION_FORMATS[acquisition_format]} "
            f"Supported formats: {list(SUPPORTED_ACQUISITION_FORMATS)}."
        )
    return acquisition_format


def acquisition_supplies_session_start_time(*, session_folder_path: str, acquisition_format: str) -> bool:
    """Report whether the raw acquisition is guaranteed to record the session's start time.

    NWB requires a session start time, and only a TDT tank's header always carries one. Where this
    returns ``False`` the user must supply the start time through the Step 6 metadata form, which
    overrides whatever the acquisition may have read.

    Parameters
    ----------
    session_folder_path : str
        Path to the raw session folder.
    acquisition_format : str
        The format the session was recorded in, as returned by :func:`resolve_acquisition_format`.

    Returns
    -------
    bool
        ``True`` for TDT tanks, ``False`` otherwise. A ``.doric`` HDF5 export carries a creation
        timestamp only when the acquisition software wrote one, so Doric is not counted here.
    """
    return acquisition_format == "tdt"
