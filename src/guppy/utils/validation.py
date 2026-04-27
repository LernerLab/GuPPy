"""Shared input-validation helpers for the GuPPy pipeline.

This module is the canonical home for reusable validation logic that several
layers of the pipeline (frontend, orchestration, analysis) need. Validation
that is genuinely one-off — i.e. coupled to a specific extractor or step —
should stay where it is used; only patterns that already repeat (or that this
module's introduction makes repeatable without straining the abstraction) live
here.

Conventions
-----------
- **Exception type**: validation helpers raise ``ValueError``. The Panel UI in
  ``orchestration/home.py`` catches exceptions from input-parameter parsing and
  step orchestration and surfaces them as persistent notifications via
  ``pn.state.notifications.error(str(e), duration=0)``. Using ``ValueError``
  consistently — instead of the generic ``Exception`` left over from older code
  — lets callers distinguish input problems from genuine bugs.
- **Layer responsibilities**:

  * Frontend (``src/guppy/frontend/``): required-field and format checks that
    can be evaluated from the form alone (folder selected, DANDI URI present).
  * Orchestration (``src/guppy/orchestration/``): pre-execution prerequisite
    checks that depend on the cross-product of multiple parameters or on
    on-disk state (storenames consistency, peak-window ordering, metric
    availability against step-5 outputs).
  * Analysis (``src/guppy/analysis/``): parameter-vs-data checks that need a
    loaded signal (baseline window inside signal timespan).

- **Error message style**: name the offending value, state the rule, and tell
  the user the valid range or fix. See PR #283 for the established template.
"""

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return not np.isnan(float(value))


def validate_window_bounds(*, start, end, ts_min, ts_max, start_name, end_name, range_label="signal timespan"):
    """Validate a ``[start, end]`` window against an outer ``[ts_min, ts_max]`` range.

    Parameters
    ----------
    start, end : numeric
        The candidate window bounds, in the same units as ``ts_min`` / ``ts_max``.
    ts_min, ts_max : numeric
        The valid outer range the window must lie inside.
    start_name, end_name : str
        Parameter names used in error messages (e.g. ``"baselineWindowStart"``).
    range_label : str, optional
        Short label for the outer range used in error messages
        (e.g. ``"signal timespan"`` or ``"PSTH window"``).

    Raises
    ------
    ValueError
        If either bound is non-numeric / NaN, if ``start >= end``, or if either
        bound falls outside ``[ts_min, ts_max]``.
    """
    for name, value in ((start_name, start), (end_name, end)):
        if not _is_finite_number(value):
            message = f"{name}={value!r} is not a valid number; provide a numeric value in seconds."
            logger.error(message)
            raise ValueError(message)

    if start >= end:
        message = f"{start_name}={start} must be strictly less than {end_name}={end}; " f"choose start < end."
        logger.error(message)
        raise ValueError(message)

    if start < ts_min or end > ts_max:
        offending = []
        if start < ts_min:
            offending.append(f"{start_name}={start} is before the signal start {ts_min:.4g}s")
        if end > ts_max:
            offending.append(f"{end_name}={end} exceeds signal duration {ts_max:.4g}s")
        message = (
            f"{'; '.join(offending)}; "
            f"{range_label} is [{ts_min:.4g}, {ts_max:.4g}]s — "
            f"choose values within this range."
        )
        logger.error(message)
        raise ValueError(message)


def validate_peak_windows(*, peak_starts, peak_ends):
    """Validate paired peak-window arrays and return them with NaN padding stripped.

    The GUI exposes ten peak-window slots, each padded with ``NaN`` when unused,
    so valid input has equal numbers of non-NaN start and end values.

    Parameters
    ----------
    peak_starts, peak_ends : sequence of float
        Per-pair start and end times (in seconds, relative to the PSTH event).

    Returns
    -------
    starts, ends : np.ndarray
        Cleaned arrays with NaN slots removed.

    Raises
    ------
    ValueError
        If the number of non-NaN starts and ends differ, or if any
        ``end <= start`` after stripping.
    """
    starts = np.asarray(peak_starts, dtype=float)
    ends = np.asarray(peak_ends, dtype=float)
    starts = starts[~np.isnan(starts)]
    ends = ends[~np.isnan(ends)]

    if starts.shape[0] != ends.shape[0]:
        message = (
            f"Number of Peak Start Time and Peak End Time are unequal "
            f"(start: {starts.shape[0]}, end: {ends.shape[0]}). "
            f"Each peak window needs both a start and an end value."
        )
        logger.error(message)
        raise ValueError(message)

    if starts.shape[0] > 0 and np.less_equal(ends, starts).any():
        offending = [f"(start={starts[i]}, end={ends[i]})" for i in range(starts.shape[0]) if ends[i] <= starts[i]]
        message = (
            "Peak End Time is less than or equal to Peak Start Time for "
            f"{len(offending)} window(s): {', '.join(offending)}. "
            "Please check the Peak parameters window."
        )
        logger.error(message)
        raise ValueError(message)

    return starts, ends


def validate_required_folder_selection(*, file_selectors: Sequence) -> None:
    """Validate that at least one folder is selected across the given file selectors.

    Parameters
    ----------
    file_selectors : sequence
        Iterable of Panel file-selector widgets (or any object with a ``.value``
        attribute that returns a list of selected paths).

    Raises
    ------
    ValueError
        If every selector is empty.
    """
    if all(len(selector.value) == 0 for selector in file_selectors):
        message = (
            "No folder is selected for analysis. Pick at least one session folder in the "
            "file selector(s) before running this step."
        )
        logger.error(message)
        raise ValueError(message)


def validate_same_parent_directory(*, paths: Sequence[str]) -> np.ndarray:
    """Validate that every path shares the same parent directory.

    Parameters
    ----------
    paths : sequence of str
        Absolute paths to selected session folders.

    Returns
    -------
    np.ndarray
        A length-1 array containing the shared parent directory.

    Raises
    ------
    ValueError
        If the paths span more than one parent directory.
    """
    parents = np.unique(np.asarray([os.path.dirname(p) for p in paths]))
    if len(parents) > 1:
        path_to_parent = "\n".join(f"  - {p} (parent: {os.path.dirname(p)})" for p in paths)
        message = (
            "All the folders selected should be at the same location, but the selected folders "
            f"span {len(parents)} parent directories:\n{path_to_parent}"
        )
        logger.error(message)
        raise ValueError(message)
    return parents
