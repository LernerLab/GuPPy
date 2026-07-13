import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def compute_cross_correlation(
    signal_a_trials: list[np.ndarray], signal_b_trials: list[np.ndarray], sample_rate: float
) -> np.ndarray:
    """
    Compute normalized cross-correlations between paired trial arrays.

    Parameters
    ----------
    signal_a_trials : array-like of np.ndarray
        Sequence of 1-D arrays for the first signal (one per trial).
    signal_b_trials : array-like of np.ndarray
        Sequence of 1-D arrays for the second signal (one per trial).
    sample_rate : float
        Sampling rate in Hz; used to convert lag indices to milliseconds.

    Returns
    -------
    cross_correlations : np.ndarray
        Shape ``(n_trials + 1, n_lags)`` array where the last row contains
        lag values in milliseconds.

    Raises
    ------
    ValueError
        If either region has no trials to correlate.

    Notes
    -----
    Independent artifact removal can leave the two regions with a different
    number of trials, or with trials of different sample lengths. The two
    regions are aligned to their common trial count and common per-trial sample
    length so the correlation is always well-defined and rectangular; a warning
    is logged whenever an alignment discards data.
    """
    a_trials = list(signal_a_trials)
    b_trials = list(signal_b_trials)

    n_trials = min(len(a_trials), len(b_trials))
    if n_trials == 0:
        message = (
            "Cross-correlation requires at least one trial in each region, but received "
            f"{len(a_trials)} trial(s) for the first region and {len(b_trials)} for the second."
        )
        logger.error(message)
        raise ValueError(message)
    if len(a_trials) != len(b_trials):
        logger.warning(
            f"Cross-correlation received {len(a_trials)} trials for the first region and "
            f"{len(b_trials)} for the second (uneven artifact removal across regions); "
            f"correlating the first {n_trials} matched trials."
        )
    a_trials = a_trials[:n_trials]
    b_trials = b_trials[:n_trials]

    # Align every trial to a common sample length so each correlation has the same
    # length and the stacked result is rectangular (regions may differ in sample count).
    common_length = min(trial.shape[0] for trial in (*a_trials, *b_trials))

    cross_corr = list()
    for trial_a, trial_b in zip(a_trials, b_trials):
        trial_a = trial_a[:common_length]
        trial_b = trial_b[:common_length]
        if np.isnan(trial_a).any() or np.isnan(trial_b).any():
            corr = signal.correlate(trial_a, trial_b, method="direct")
        else:
            corr = signal.correlate(trial_a, trial_b)
        corr_norm = corr / np.max(np.abs(corr))
        cross_corr.append(corr_norm)

    lag = signal.correlation_lags(common_length, common_length)
    lag_msec = np.array(lag / sample_rate, dtype="float32").reshape(1, -1)

    cross_correlations = np.array(cross_corr, dtype="float32")
    cross_correlations = np.concatenate((cross_correlations, lag_msec), axis=0)
    return cross_correlations
