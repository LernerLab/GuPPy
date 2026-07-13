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
        If there are no trials to correlate.

    Notes
    -----
    The two trial sequences must already be paired: ``signal_a_trials[k]`` is
    correlated with ``signal_b_trials[k]``. The caller is responsible for
    reconciling the two regions' trials (see
    :func:`guppy.analysis.psth_utils.match_trials_by_timestamp`) so that uneven
    artifact removal does not misalign them.
    """
    a_trials = list(signal_a_trials)
    b_trials = list(signal_b_trials)
    if len(a_trials) == 0 or len(b_trials) == 0:
        message = (
            "Cross-correlation requires at least one trial in each region, but received "
            f"{len(a_trials)} trial(s) for the first region and {len(b_trials)} for the second."
        )
        logger.error(message)
        raise ValueError(message)

    cross_corr = list()
    for trial_a, trial_b in zip(a_trials, b_trials):
        if np.isnan(trial_a).any() or np.isnan(trial_b).any():
            corr = signal.correlate(trial_a, trial_b, method="direct")
        else:
            corr = signal.correlate(trial_a, trial_b)
        corr_norm = corr / np.max(np.abs(corr))
        cross_corr.append(corr_norm)

    lag = signal.correlation_lags(len(a_trials[0]), len(b_trials[0]))
    lag_msec = np.array(lag / sample_rate, dtype="float32").reshape(1, -1)

    cross_correlations = np.array(cross_corr, dtype="float32")
    cross_correlations = np.concatenate((cross_correlations, lag_msec), axis=0)
    return cross_correlations
