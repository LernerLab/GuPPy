"""Window algebra for artifact removal.

The Select Artifact Windows page asks the user to mark the periods where
*artifacts occurred*, but ``coordsForPreProcessing_<recording_site>.npy`` stores
the periods to *keep*. The conversion between the two lives here so it is
testable independently of the Panel page.
"""

import numpy as np


def merge_windows(*, windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Fuse overlapping and touching windows into a minimal sorted set.

    Parameters
    ----------
    windows : list of tuple of float
        ``(start, end)`` windows in any order.

    Returns
    -------
    list of tuple of float
        Sorted, non-overlapping windows covering the same span as the input.
    """
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def complement_windows(
    *, windows: list[tuple[float, float]], span_start: float, span_end: float
) -> list[tuple[float, float]]:
    """
    Invert artifact windows into the keep-windows that surround them.

    Parameters
    ----------
    windows : list of tuple of float
        ``(start, end)`` windows marking where artifacts occurred.
    span_start, span_end : float
        Bounds of the full recording. Pass the margined span
        (``timestamps[0] - dt``, ``timestamps[-1] + dt``) so the first and last
        samples are never clipped.

    Returns
    -------
    list of tuple of float
        The ``(start, end)`` windows to keep. An empty ``windows`` yields the single
        full-span window; windows spanning the whole range yield an empty list.
    """
    keep: list[tuple[float, float]] = []
    cursor = span_start
    for start, end in merge_windows(windows=windows):
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < span_end:
        keep.append((cursor, span_end))
    return keep


def windows_to_coords(*, windows: list[tuple[float, float]]) -> np.ndarray:
    """
    Flatten keep-windows into the interleaved ``(2M, 2)`` array stored on disk.

    Column 0 holds ``[s0, e0, s1, e1, …]``, which ``fetchCoords`` reshapes back
    into ``[[s0, e0], …]``; column 1 is an unused placeholder.

    Parameters
    ----------
    windows : list of tuple of float
        ``(start, end)`` windows to keep.

    Returns
    -------
    np.ndarray
        Shape ``(2M, 2)`` array of coordinates.
    """
    rows = []
    for start, end in windows:
        rows.append([start, 0.0])
        rows.append([end, 0.0])
    return np.array(rows)


def coords_to_windows(*, coords: np.ndarray) -> list[tuple[float, float]]:
    """
    Convert fetched keep-window coordinates into ``(start, end)`` tuples.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds, as returned by
        ``fetchCoords``.

    Returns
    -------
    list of tuple of float
        The same windows as Python floats.
    """
    return [(float(start), float(end)) for start, end in coords]
