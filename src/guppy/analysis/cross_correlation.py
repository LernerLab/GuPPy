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
    """
    cross_corr = list()
    for trial_a, trial_b in zip(signal_a_trials, signal_b_trials):
        if np.isnan(trial_a).any() or np.isnan(trial_b).any():
            corr = signal.correlate(trial_a, trial_b, method="direct")
        else:
            corr = signal.correlate(trial_a, trial_b)
        corr_norm = corr / np.max(np.abs(corr))
        cross_corr.append(corr_norm)
        lag = signal.correlation_lags(len(trial_a), len(trial_b))
        lag_msec = np.array(lag / sample_rate, dtype="float32")

    cross_correlations = np.array(cross_corr, dtype="float32")
    lag_msec = lag_msec.reshape(1, -1)
    cross_correlations = np.concatenate((cross_correlations, lag_msec), axis=0)
    return cross_correlations
