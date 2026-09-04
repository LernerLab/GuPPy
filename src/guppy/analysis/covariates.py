"""Behavioral covariates aligned to whole-session time bins.

A covariate is a continuous variable measured outside the rig -- an akinesia
severity rated every couple of minutes, say -- ingested as an ordinary GuPPy-CSV
store and labeled ``covariate_<name>`` in step 1. This module averages such a
variable within the bins :mod:`guppy.analysis.binned_metrics` already produced
and correlates it against each per-bin photometry metric.

Pearson r and Spearman rho are reported as **descriptive** statistics. No p-value
is computed: per-bin photometry and a slowly varying behavioral score are both
strongly autocorrelated across bins, and the standard significance tests assume
independent samples, so any p-value derived from these numbers would be
uninterpretable.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# Columns of a binned-metrics table that describe the bins rather than the signal.
_BIN_GEOMETRY_COLUMNS = ("bin_start", "bin_end", "n_samples")

# Per-covariate sample-count columns are derived, not variables in their own right.
_COUNT_PREFIX = "n_samples_"

# Below this many paired bins a correlation coefficient is not worth reporting.
_MINIMUM_PAIRS = 3


def bin_edges_from_binned_metrics(*, binned_metrics: pd.DataFrame) -> np.ndarray:
    """Recover the bin edges behind a binned-metrics table.

    Reading the edges back off the table keeps a covariate on exactly the bins its
    metrics were reduced to, including a short final bin.

    Parameters
    ----------
    binned_metrics : pd.DataFrame
        A table written by :func:`guppy.analysis.binned_metrics.compute_binned_metrics`,
        with ``bin_start`` and ``bin_end`` columns.

    Returns
    -------
    np.ndarray
        Monotonically increasing bin edges, ``n_bins + 1`` of them.
    """
    bin_starts = binned_metrics["bin_start"].to_numpy(dtype=float)
    return np.append(bin_starts, float(binned_metrics["bin_end"].iloc[-1]))


def _bin_mean(*, values: np.ndarray, bin_index: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` within each bin, NaN where the bin holds no finite sample."""
    finite = ~np.isnan(values)
    sums = np.bincount(bin_index[finite], weights=values[finite], minlength=n_bins)
    counts = np.bincount(bin_index[finite], minlength=n_bins)
    # np.maximum guards the division only; the where() picks NaN for empty bins.
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def _validate_covariate_coverage(
    *, name: str, counts: np.ndarray, sample_timestamps: np.ndarray, bin_edges: np.ndarray
) -> None:
    """Raise when a covariate lands in too few bins to correlate.

    Parameters
    ----------
    name : str
        Covariate name, used in the error message.
    counts : np.ndarray
        Number of samples that landed in each bin.
    sample_timestamps : np.ndarray
        The covariate's own timestamps (s), before binning.
    bin_edges : np.ndarray
        Bin edges (s) the covariate was binned onto.

    Raises
    ------
    ValueError
        If fewer than three bins hold a sample.
    """
    occupied_bins = int(np.count_nonzero(counts))
    if occupied_bins >= _MINIMUM_PAIRS:
        return

    covariate_span = (
        f"[{sample_timestamps.min():.4g}, {sample_timestamps.max():.4g}]s" if sample_timestamps.size else "empty"
    )
    message = (
        f"Covariate '{name}' lands in only {occupied_bins} bin(s), which is too few to correlate; "
        f"at least {_MINIMUM_PAIRS} are needed. The covariate spans {covariate_span} but the session "
        f"spans [{bin_edges[0]:.4g}, {bin_edges[-1]:.4g}]s — check that its timestamps are in seconds "
        "on the same clock the session was recorded on."
    )
    logger.error(message)
    raise ValueError(message)


def compute_binned_covariates(
    *, covariate_series: dict[str, tuple[np.ndarray, np.ndarray]], bin_edges: np.ndarray
) -> pd.DataFrame:
    """Average each behavioral covariate within the bins its metrics used.

    Bins are half-open -- a sample on an interior edge belongs to the bin that edge
    opens -- except the final bin, which includes its own end. Samples outside the
    binned span are dropped rather than folded into the first or last bin.

    Parameters
    ----------
    covariate_series : dict of str to tuple of np.ndarray
        Covariate name → ``(timestamps, values)``, both 1-D and the same length.
        Timestamps are in seconds on the session's own clock.
    bin_edges : np.ndarray
        Bin edges (s), ``n_bins + 1`` of them.

    Returns
    -------
    pd.DataFrame
        One row per bin, indexed ``0..n_bins-1`` (index name ``"bin"``), with columns
        ``bin_start``, ``bin_end``, one ``<name>`` per covariate and one
        ``n_samples_<name>`` per covariate. A bin holding no sample of a covariate
        reads NaN with a count of 0.

    Raises
    ------
    ValueError
        If any covariate lands in fewer than three bins.
    """
    bin_edges = np.asarray(bin_edges, dtype=float).ravel()
    n_bins = bin_edges.shape[0] - 1

    means = {}
    sample_counts = {}
    for name in sorted(covariate_series):
        timestamps, values = covariate_series[name]
        timestamps = np.asarray(timestamps, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()

        # Unlike metric timestamps, a covariate's are not guaranteed to sit inside the
        # recording, and clipping alone would fold a stray sample into an end bin.
        inside = (timestamps >= bin_edges[0]) & (timestamps <= bin_edges[-1])
        bin_index = np.searchsorted(bin_edges, timestamps[inside], side="right") - 1
        np.clip(bin_index, 0, n_bins - 1, out=bin_index)

        mean, counts = _bin_mean(values=values[inside], bin_index=bin_index, n_bins=n_bins)
        _validate_covariate_coverage(name=name, counts=counts, sample_timestamps=timestamps, bin_edges=bin_edges)
        means[name] = mean
        sample_counts[_COUNT_PREFIX + name] = counts

    binned = pd.DataFrame(
        {"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], **means, **sample_counts},
        index=pd.RangeIndex(n_bins, name="bin"),
    )
    return binned


def _pearson_r(*, x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two paired arrays, NaN when either is constant."""
    # np.corrcoef rather than scipy.stats.pearsonr: pearsonr returns a p-value in the
    # same object, and no valid p-value can be computed from autocorrelated bins.
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_rho(*, x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation of two paired arrays, NaN when either is constant."""
    return _pearson_r(x=rankdata(x), y=rankdata(y))


def compute_covariate_correlations(*, binned_metrics: pd.DataFrame, binned_covariates: pd.DataFrame) -> pd.DataFrame:
    """Correlate every per-bin photometry metric against every behavioral covariate.

    Bins where either series is missing are dropped pairwise, so ``n_bins`` is the
    number of bins actually behind each pair of coefficients.

    Parameters
    ----------
    binned_metrics : pd.DataFrame
        Per-bin photometry metrics; every column that is not bin geometry is treated
        as a metric.
    binned_covariates : pd.DataFrame
        Per-bin covariate means on the same bins, as returned by
        :func:`compute_binned_covariates`.

    Returns
    -------
    pd.DataFrame
        One row per (metric, covariate) pair, indexed ``0..n_pairs-1`` (index name
        ``"pair"``), with columns ``metric``, ``covariate``, ``pearson_r``,
        ``spearman_rho`` and ``n_bins``. Coefficients are NaN when a series is
        constant or fewer than three bins pair up.
    """
    metric_names = sorted(name for name in binned_metrics.columns if name not in _BIN_GEOMETRY_COLUMNS)
    covariate_names = sorted(
        name
        for name in binned_covariates.columns
        if name not in ("bin_start", "bin_end") and not name.startswith(_COUNT_PREFIX)
    )

    rows = []
    for metric in metric_names:
        metric_values = binned_metrics[metric].to_numpy(dtype=float)
        for covariate in covariate_names:
            covariate_values = binned_covariates[covariate].to_numpy(dtype=float)
            usable = ~np.isnan(metric_values) & ~np.isnan(covariate_values)
            n_bins = int(np.count_nonzero(usable))

            if n_bins < _MINIMUM_PAIRS:
                logger.warning(
                    "Only %s bin(s) pair '%s' with '%s'; reporting no coefficients for this pair, which needs at least "
                    "%s.",
                    n_bins,
                    covariate,
                    metric,
                    _MINIMUM_PAIRS,
                )
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = _pearson_r(x=covariate_values[usable], y=metric_values[usable])
                spearman = _spearman_rho(x=covariate_values[usable], y=metric_values[usable])
                if np.isnan(pearson):
                    logger.warning(
                        "'%s' or '%s' is constant across the %s paired bins, so their correlation is undefined.",
                        covariate,
                        metric,
                        n_bins,
                    )

            rows.append(
                {
                    "metric": metric,
                    "covariate": covariate,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "n_bins": n_bins,
                }
            )

    return pd.DataFrame(rows, index=pd.RangeIndex(len(rows), name="pair"))
