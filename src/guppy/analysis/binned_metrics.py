"""Whole-session time-binned metrics.

Some experiments score a continuous behavioral variable at a coarse cadence --
akinesia severity rated once every two minutes, say -- and ask how the photometry
signal tracks it. That question needs the session reduced to one value per fixed
time bin rather than to event-triggered averages. This module tiles a recording
into fixed-width bins and reduces each bin to a mean z-score, a mean dF/F and a
count of detected transients.

The tiling is derived from the recording alone: bins start at the first timestamp
and run in ``bin_width`` steps to the last. The final bin is kept even when it is
shorter than the others, and its true bounds are reported so a caller can drop it.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A final bin narrower than this fraction of bin_width is a floating-point
# artifact of snapping the last edge, not a real bin.
_SLIVER_TOLERANCE = 1e-9


def compute_bin_edges(*, start: float, end: float, bin_width: float) -> np.ndarray:
    """Tile ``[start, end]`` with bins of ``bin_width``, keeping a short final bin.

    Parameters
    ----------
    start, end : float
        First and last timestamps (s) of the recording.
    bin_width : float
        Bin width in seconds.

    Returns
    -------
    np.ndarray
        Monotonically increasing bin edges, ``n_bins + 1`` of them. The first is
        ``start`` and the last is exactly ``end``.

    Raises
    ------
    ValueError
        If ``bin_width`` is not positive, or ``end`` is not after ``start``.
    """
    if not bin_width > 0:
        message = f"bin_width={bin_width} must be positive; choose a bin width greater than 0 seconds."
        logger.error(message)
        raise ValueError(message)
    if not end > start:
        message = (
            f"The recording spans [{start:.4g}, {end:.4g}]s, which has no duration to bin; "
            "check that preprocessing produced a corrected time axis."
        )
        logger.error(message)
        raise ValueError(message)

    n_bins = int(np.ceil((end - start) / bin_width))
    edges = start + bin_width * np.arange(n_bins + 1, dtype=float)
    # The last edge lands on or past `end`; snap it back so no bin claims time
    # the recording does not cover.
    edges[-1] = end

    # Snapping can leave a zero-width final bin when the duration is an exact
    # multiple of bin_width but rounds up, which np.histogram rejects.
    if n_bins > 1 and edges[-1] - edges[-2] < _SLIVER_TOLERANCE * bin_width:
        edges = np.delete(edges, -2)

    return edges


def _bin_sums_and_counts(*, trace: np.ndarray, bin_index: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``trace`` and count its finite samples within each bin."""
    finite = ~np.isnan(trace)
    sums = np.bincount(bin_index[finite], weights=trace[finite], minlength=n_bins)
    counts = np.bincount(bin_index[finite], minlength=n_bins)
    return sums, counts


def _bin_mean(*, trace: np.ndarray, bin_index: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``trace`` within each bin, NaN where the bin holds no finite sample."""
    sums, counts = _bin_sums_and_counts(trace=trace, bin_index=bin_index, n_bins=n_bins)
    # np.maximum guards the division only; the where() picks NaN for empty bins.
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def compute_binned_metrics(
    *,
    z_score: np.ndarray,
    dff: np.ndarray,
    timestamps: np.ndarray,
    transient_timestamps: dict[str, np.ndarray],
    bin_width: float,
) -> pd.DataFrame:
    """Reduce a whole session to one row per fixed-width time bin.

    Bins are half-open -- a sample at an interior edge belongs to the bin that
    edge opens -- except the final bin, which includes its own end. Transient
    counts follow the same convention, so a transient and the samples around it
    are always attributed to the same bin.

    Parameters
    ----------
    z_score, dff : np.ndarray
        Full-session preprocessed traces for a single recording site, sampled on
        ``timestamps``.
    timestamps : np.ndarray
        Corrected time axis (s) shared by both traces, monotonically increasing.
    transient_timestamps : dict of str to np.ndarray
        Detected transient occurrence times (s), keyed by the metric they were
        detected on (``"z_score"`` and/or ``"dff"``). One
        ``transient_count_<metric>`` column is produced per entry.
    bin_width : float
        Bin width in seconds.

    Returns
    -------
    pd.DataFrame
        One row per bin, indexed ``0..n_bins-1`` (index name ``"bin"``), with
        columns ``bin_start``, ``bin_end``, ``n_samples``, ``mean_zscore``,
        ``mean_dff``, and one ``transient_count_<metric>`` per entry in
        ``transient_timestamps``. ``n_samples`` counts the finite samples behind
        the means, so a bin lost to artifact removal reads ``0`` with NaN means.

    Raises
    ------
    ValueError
        If ``bin_width`` is not positive, or the recording has no duration.
    """
    z_score = np.asarray(z_score, dtype=float).ravel()
    dff = np.asarray(dff, dtype=float).ravel()
    timestamps = np.asarray(timestamps, dtype=float).ravel()

    edges = compute_bin_edges(start=float(timestamps[0]), end=float(timestamps[-1]), bin_width=bin_width)
    n_bins = edges.shape[0] - 1

    # side="right" gives half-open [edge_k, edge_k+1); clipping folds the sample
    # sitting exactly on the final edge into the last bin.
    bin_index = np.searchsorted(edges, timestamps, side="right") - 1
    np.clip(bin_index, 0, n_bins - 1, out=bin_index)

    mean_zscore, n_samples = _bin_mean(trace=z_score, bin_index=bin_index, n_bins=n_bins)
    mean_dff, _ = _bin_mean(trace=dff, bin_index=bin_index, n_bins=n_bins)

    binned = pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "n_samples": n_samples,
            "mean_zscore": mean_zscore,
            "mean_dff": mean_dff,
        },
        index=pd.RangeIndex(n_bins, name="bin"),
    )

    for metric in sorted(transient_timestamps):
        occurrences = np.asarray(transient_timestamps[metric], dtype=float).ravel()
        binned["transient_count_" + metric] = np.histogram(occurrences, bins=edges)[0]

    return binned
