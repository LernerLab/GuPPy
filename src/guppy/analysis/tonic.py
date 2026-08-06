"""Tonic / basal fluorescence analysis.

Pharmacological experiments care about slow shifts in the overall fluorescence
level after a drug injection rather than event-triggered transients. This module
averages the already-preprocessed z-score and dF/F traces over user-defined
absolute-time epoch windows, producing one scalar mean per (epoch, signal). The
difference of each epoch from a baseline epoch is a viewing-time choice and is
intentionally not computed or stored here.

The epoch windows are defined per recording site (different sites may see the
drug at different times, e.g. an ICV injection), mirroring the per-site
granularity of the artifact-removal coordinates.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TONIC_EPOCH_COLUMNS = ["label", "start", "end"]


def validate_tonic_epochs(epochs: pd.DataFrame, ts_min: float, ts_max: float) -> None:
    """Validate tonic epoch windows against a recording's timespan.

    Unlike the PSTH peak windows, tonic windows are lenient at the recording
    boundaries: a window that runs to the nominal recording duration (a hair past
    the last sample) or starts at 0 (a hair before the first sample) is accepted
    and simply clamped to the available samples when averaged. Only genuinely
    unusable windows are rejected: non-numeric bounds, ``start >= end``, or a
    window that does not overlap the recording at all.

    Parameters
    ----------
    epochs : pd.DataFrame
        Epoch definitions with columns ``label``, ``start``, ``end``.
    ts_min, ts_max : float
        First and last timestamps (s) of the recording.

    Raises
    ------
    ValueError
        If any window is non-numeric, has ``start >= end``, or falls entirely
        outside ``[ts_min, ts_max]``.
    """
    labels = list(epochs["label"])
    starts = np.asarray(epochs["start"], dtype=float)
    ends = np.asarray(epochs["end"], dtype=float)

    for label, start, end in zip(labels, starts, ends):
        if not (np.isfinite(start) and np.isfinite(end)):
            message = f"epoch {label!r} has a non-numeric start/end ({start}, {end}); provide numeric seconds."
            logger.error(message)
            raise ValueError(message)
        if start >= end:
            message = f"epoch {label!r} start={start} must be strictly less than end={end}; choose start < end."
            logger.error(message)
            raise ValueError(message)
        if end <= ts_min or start >= ts_max:
            message = (
                f"epoch {label!r} window [{start}, {end}]s does not overlap the recording "
                f"[{ts_min:.4g}, {ts_max:.4g}]s; choose a window inside the recording."
            )
            logger.error(message)
            raise ValueError(message)


def compute_tonic_means(
    z_score: np.ndarray,
    dff: np.ndarray,
    timestamps: np.ndarray,
    epochs: pd.DataFrame,
) -> pd.DataFrame:
    """Average the z-score and dF/F traces within each tonic epoch window.

    Windows are clamped to the recording's timespan, so an epoch running to the
    nominal recording duration simply averages up to the last sample.

    Parameters
    ----------
    z_score, dff : np.ndarray
        Full-session preprocessed traces for a single recording site, sampled on
        ``timestamps``.
    timestamps : np.ndarray
        Corrected time axis (s) shared by both traces.
    epochs : pd.DataFrame
        Epoch definitions with columns ``label``, ``start``, ``end`` (start/end
        in seconds of absolute session time).

    Returns
    -------
    pd.DataFrame
        Indexed by epoch label (index name ``"epoch"``) with columns
        ``mean_zscore`` and ``mean_dff`` — the window mean of each trace.

    Raises
    ------
    ValueError
        If any epoch window is non-numeric, has ``start >= end``, or does not
        overlap the recording's timespan.
    """
    z_score = np.asarray(z_score, dtype=float).ravel()
    dff = np.asarray(dff, dtype=float).ravel()
    timestamps = np.asarray(timestamps, dtype=float).ravel()

    ts_min = float(timestamps[0])
    ts_max = float(timestamps[-1])

    validate_tonic_epochs(epochs, ts_min, ts_max)

    labels = list(epochs["label"])
    starts = np.asarray(epochs["start"], dtype=float)
    ends = np.asarray(epochs["end"], dtype=float)

    mean_zscore = []
    mean_dff = []
    for start, end in zip(starts, ends):
        mask = (timestamps >= start) & (timestamps <= end)
        mean_zscore.append(np.nanmean(z_score[mask]))
        mean_dff.append(np.nanmean(dff[mask]))

    return pd.DataFrame(
        {"mean_zscore": mean_zscore, "mean_dff": mean_dff},
        index=pd.Index(labels, name="epoch"),
    )
