"""Bootstrap significance testing for PSTH curves.

Computes a bias-corrected and accelerated (BCa) bootstrap confidence interval on the mean
PSTH, or on the difference between two mean PSTHs, at every timepoint, and marks the
timepoints whose interval excludes zero. Isolated significant timepoints are discarded so
that only stretches longer than the moving-average filter's own resolution survive.

The resampling axis is whichever axis the caller supplies: PSTH trials in a run folder,
session means in a group folder.
"""

import logging

import numpy as np
from scipy import ndimage, stats

logger = logging.getLogger(__name__)

DEFAULT_NUM_RESAMPLES = 1000
DEFAULT_SIGNIFICANCE_LEVEL = 0.05

# Fixed so that re-running an analysis reproduces its significance masks exactly.
BOOTSTRAP_SEED = 20260828

# The bootstrap needs at least three samples to have a non-degenerate resampling
# distribution, and stays unreliable well above that: below this many samples the interval
# is driven by a handful of distinct resamples.
MINIMUM_SAMPLES = 3
SMALL_SAMPLE_WARNING_THRESHOLD = 5

# scipy resamples every replicate at once unless told otherwise, which for a full-length
# PSTH means an intermediate of (num_resamples, num_samples, num_timepoints) -- tens of
# gigabytes. The batch size caps that intermediate at roughly this many bytes.
_MAXIMUM_RESAMPLE_BYTES = 5e8


def minimum_consecutive_samples_for(*, filter_window: int) -> int:
    """
    Shortest significant stretch that is not discarded as noise.

    Parameters
    ----------
    filter_window : int
        Moving-average window, in samples, applied during preprocessing.

    Returns
    -------
    minimum_consecutive_samples : int
        Number of consecutive samples a significant stretch must exceed.
    """
    # A moving-average filter cannot produce features shorter than its own window, so
    # anything briefer than one window period is noise by construction.
    return 2 * filter_window


def _resample_batch_size(*, num_samples: int, num_timepoints: int) -> int:
    """
    Number of bootstrap replicates to resample at once.

    Parameters
    ----------
    num_samples : int
        Number of trials or session means being resampled.
    num_timepoints : int
        Length of the PSTH time axis.

    Returns
    -------
    batch : int
        Replicates per vectorized call, at least 1.
    """
    return max(1, int(_MAXIMUM_RESAMPLE_BYTES / (num_samples * num_timepoints * 8)))


def _difference_of_means(samples_a: np.ndarray, samples_b: np.ndarray, axis: int) -> np.ndarray:
    """
    Difference between two sets of samples' means, ignoring NaN.

    Parameters
    ----------
    samples_a, samples_b : np.ndarray
        Sample arrays sharing a time axis.
    axis : int
        Axis along which to average.

    Returns
    -------
    difference : np.ndarray
        ``nanmean(samples_a) - nanmean(samples_b)`` along ``axis``.
    """
    return np.nanmean(samples_a, axis=axis) - np.nanmean(samples_b, axis=axis)


def bootstrap_mean_confidence_interval(
    *,
    samples: np.ndarray,
    rng: np.random.Generator,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BCa bootstrap confidence interval on the mean, at every timepoint.

    Parameters
    ----------
    samples : np.ndarray
        2-D array of shape ``(num_samples, num_timepoints)``. Resampling is over the first
        axis. NaN entries are ignored when averaging.
    rng : np.random.Generator
        Random generator driving the resampling.
    num_resamples : int
        Number of bootstrap replicates.
    significance_level : float
        Two-sided alpha, e.g. ``0.05`` for a 95% interval.

    Returns
    -------
    lower, upper : np.ndarray
        Confidence bounds, each of shape ``(num_timepoints,)``.

    Raises
    ------
    ValueError
        If fewer than three samples are supplied.
    """
    num_samples, num_timepoints = samples.shape
    if num_samples < MINIMUM_SAMPLES:
        raise ValueError(f"Bootstrap needs at least {MINIMUM_SAMPLES} samples to resample; got {num_samples}.")

    result = stats.bootstrap(
        (samples,),
        np.nanmean,
        axis=0,
        n_resamples=num_resamples,
        confidence_level=1 - significance_level,
        method="BCa",
        batch=_resample_batch_size(num_samples=num_samples, num_timepoints=num_timepoints),
        rng=rng,
    )

    return result.confidence_interval.low, result.confidence_interval.high


def bootstrap_difference_confidence_interval(
    *,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rng: np.random.Generator,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BCa bootstrap confidence interval on the difference of two means, at every timepoint.

    The two sets are resampled independently, so this is an unpaired comparison. Their
    sample counts may differ.

    Parameters
    ----------
    samples_a, samples_b : np.ndarray
        2-D arrays of shape ``(num_samples, num_timepoints)`` sharing a time axis.
        Resampling is over the first axis. NaN entries are ignored when averaging.
    rng : np.random.Generator
        Random generator driving the resampling.
    num_resamples : int
        Number of bootstrap replicates.
    significance_level : float
        Two-sided alpha, e.g. ``0.05`` for a 95% interval.

    Returns
    -------
    lower, upper : np.ndarray
        Confidence bounds on ``mean(samples_a) - mean(samples_b)``, each of shape
        ``(num_timepoints,)``.

    Raises
    ------
    ValueError
        If either set has fewer than three samples, or the two disagree on timepoint count.
    """
    num_samples_a, num_timepoints = samples_a.shape
    num_samples_b, num_timepoints_b = samples_b.shape
    if num_timepoints != num_timepoints_b:
        raise ValueError(
            f"The two PSTHs must share a time axis; got {num_timepoints} and {num_timepoints_b} timepoints."
        )
    if num_samples_a < MINIMUM_SAMPLES or num_samples_b < MINIMUM_SAMPLES:
        raise ValueError(
            f"Bootstrap needs at least {MINIMUM_SAMPLES} samples in each set to resample; "
            f"got {num_samples_a} and {num_samples_b}."
        )

    result = stats.bootstrap(
        (samples_a, samples_b),
        _difference_of_means,
        axis=0,
        paired=False,
        n_resamples=num_resamples,
        confidence_level=1 - significance_level,
        method="BCa",
        batch=_resample_batch_size(num_samples=max(num_samples_a, num_samples_b), num_timepoints=num_timepoints),
        rng=rng,
    )

    return result.confidence_interval.low, result.confidence_interval.high


def significant_sample_mask(*, lower: np.ndarray, upper: np.ndarray, minimum_consecutive_samples: int) -> np.ndarray:
    """
    Mark timepoints belonging to a long enough run of intervals that exclude zero.

    Parameters
    ----------
    lower, upper : np.ndarray
        Confidence bounds over the time axis.
    minimum_consecutive_samples : int
        A run of consecutive significant timepoints is kept only if it is strictly longer
        than this.

    Returns
    -------
    mask : np.ndarray
        Boolean array over the time axis, True inside a surviving run.
    """
    excludes_zero = (lower > 0) | (upper < 0)

    mask = np.zeros_like(excludes_zero, dtype=bool)
    labeled_runs, _ = ndimage.label(excludes_zero)
    for run_slice in ndimage.find_objects(labeled_runs):
        run_length = run_slice[0].stop - run_slice[0].start
        if run_length > minimum_consecutive_samples:
            mask[run_slice] = True

    return mask
