"""PSTH significance testing over an output directory's PSTH results.

Runs inside step 4 for each session run folder and inside Group Analysis for each group
folder. Both hold PSTH files with the same layout, so the same code serves both; only the
meaning of a resampled column differs, a trial in a run folder and a session mean in a
group folder.

Every event is tested against zero automatically. Pairs of events are compared only when
the user names them, since which contrast is worth drawing is a scientific judgement.
"""

import logging
import multiprocessing as mp
import os
from itertools import repeat

import numpy as np
import pandas as pd

from ..analysis.io_utils import (
    is_continuous_label,
    make_dir_for_psth_significance,
    recording_sites_for_output_directory,
)
from ..analysis.psth_significance import (
    BOOTSTRAP_SEED,
    MINIMUM_SAMPLES,
    SMALL_SAMPLE_WARNING_THRESHOLD,
    bootstrap_difference_confidence_interval,
    bootstrap_mean_confidence_interval,
    minimum_consecutive_samples_for,
    significant_sample_mask,
)
from ..analysis.standard_io import (
    write_psth_significance_to_csv,
    write_psth_significance_to_hdf5,
)
from ..utils.utils import event_labels_for_analysis, read_Df
from ..utils.validation import (
    validate_comparison_events_available,
    validate_psth_comparisons,
)

logger = logging.getLogger(__name__)

# Columns of a PSTH table that are not resampled: the time axis and the summary columns
# appended by create_Df_for_psth. Binned-trial columns are excluded separately by prefix.
_NON_SAMPLE_COLUMNS = ("timestamps", "mean", "err")
_BIN_COLUMN_PREFIX = "bin_"

# Fraction of a window that may come back without a computable interval before the run is
# worth flagging to the user.
UNCOMPUTABLE_FRACTION_WARNING_THRESHOLD = 0.1


def psth_metrics(inputParameters: dict[str, object]) -> list[str]:
    """Return the preprocessed metrics PSTHs were computed for.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``selectForComputePsth``.

    Returns
    -------
    list of str
        Metric prefixes, some of ``"z_score"`` and ``"dff"``.
    """
    selectForComputePsth = inputParameters["selectForComputePsth"]
    if selectForComputePsth == "z_score":
        return ["z_score"]
    if selectForComputePsth == "dff":
        return ["dff"]
    return ["z_score", "dff"]


def read_psth_samples(
    *, filepath: str, event: str, recording_site: str, basename: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one PSTH file's resampling columns and its time axis.

    Parameters
    ----------
    filepath : str
        Output directory holding the PSTH file.
    event : str
        Event label, already sanitized.
    recording_site : str
        Recording site the PSTH belongs to.
    basename : str
        Preprocessed trace name, ``<metric>_<recording_site>``.

    Returns
    -------
    tuple of np.ndarray, or None
        ``(samples, timestamps)`` where ``samples`` has shape
        ``(num_samples, num_timepoints)``, or None when the directory holds no PSTH for
        this combination.
    """
    psth_path = os.path.join(filepath, f"{event}_{recording_site}_{basename}.h5")
    if not os.path.exists(psth_path):
        return None

    psth = read_Df(filepath, f"{event}_{recording_site}", basename)
    # Selected by name rather than position: binned-trial columns sit between the trials
    # and the time axis whenever bin_psth_trials is set.
    sample_columns = [
        column
        for column in psth.columns
        if column not in _NON_SAMPLE_COLUMNS and not str(column).startswith(_BIN_COLUMN_PREFIX)
    ]
    samples = np.asarray(psth[sample_columns], dtype=float).T
    timestamps = np.asarray(psth["timestamps"], dtype=float)

    return samples, timestamps


def compute_comparison(
    *,
    samples_a: np.ndarray,
    samples_b: np.ndarray | None,
    timestamps: np.ndarray,
    minimum_consecutive_samples: int,
    significance_level: float,
    num_resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap one comparison and mark its significant stretches.

    Parameters
    ----------
    samples_a : np.ndarray
        Resampling columns of the first PSTH, shape ``(num_samples, num_timepoints)``.
    samples_b : np.ndarray or None
        Resampling columns of the second PSTH for a two-sample comparison, or None to
        test ``samples_a`` against zero.
    timestamps : np.ndarray
        PSTH time axis.
    minimum_consecutive_samples : int
        A significant stretch is kept only if strictly longer than this.
    significance_level : float
        Two-sided alpha the interval is computed at.
    num_resamples : int
        Number of bootstrap resamples each interval is built from.
    rng : np.random.Generator
        Random generator driving the resampling.

    Returns
    -------
    pd.DataFrame
        One row per timepoint, with columns ``timestamps``, ``estimate``, ``ci_lower``,
        ``ci_upper``, ``significant``, ``alpha`` and ``n`` (plus ``n_b`` when two-sample).
    """
    if samples_b is None:
        estimate = np.nanmean(samples_a, axis=0)
        lower, upper = bootstrap_mean_confidence_interval(
            samples=samples_a, rng=rng, significance_level=significance_level, num_resamples=num_resamples
        )
    else:
        estimate = np.nanmean(samples_a, axis=0) - np.nanmean(samples_b, axis=0)
        lower, upper = bootstrap_difference_confidence_interval(
            samples_a=samples_a,
            samples_b=samples_b,
            rng=rng,
            significance_level=significance_level,
            num_resamples=num_resamples,
        )

    mask = significant_sample_mask(lower=lower, upper=upper, minimum_consecutive_samples=minimum_consecutive_samples)

    significance = pd.DataFrame(
        {
            "timestamps": timestamps,
            "estimate": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "significant": mask.astype(int),
            "alpha": significance_level,
            "n": samples_a.shape[0],
        }
    )
    if samples_b is not None:
        significance["n_b"] = samples_b.shape[0]

    return significance


def comparison_name(*, event_a: str, event_b: str | None, recording_site: str, basename: str) -> str:
    """Build the filename stem identifying a comparison.

    Parameters
    ----------
    event_a : str
        First event label.
    event_b : str or None
        Second event label for a two-sample comparison, or None for a test against zero.
    recording_site : str
        Recording site the comparison was computed for.
    basename : str
        Preprocessed trace name, ``<metric>_<recording_site>``.

    Returns
    -------
    str
        Name passed to the significance writers.
    """
    events = event_a if event_b is None else f"{event_a}_vs_{event_b}"
    return f"{events}_{recording_site}_{basename}"


def execute_one_comparison(
    filepath: str, comparison: tuple[str, str | None, str, str], inputParameters: dict[str, object]
) -> None:
    """Compute and write a single PSTH significance comparison.

    Parameters
    ----------
    filepath : str
        Output directory holding the PSTHs, and the parent of the results subdirectory.
    comparison : tuple
        ``(event_a, event_b, recording_site, basename)``; ``event_b`` is None for a test
        against zero.
    inputParameters : dict
        Full pipeline input parameters; uses ``filter_window``, ``psthSignificanceAlpha``
        and ``psthBootstrapResamples``.
    """
    event_a, event_b, recording_site, basename = comparison

    loaded_a = read_psth_samples(filepath=filepath, event=event_a, recording_site=recording_site, basename=basename)
    if loaded_a is None:
        return
    samples_a, timestamps = loaded_a

    samples_b = None
    if event_b is not None:
        loaded_b = read_psth_samples(filepath=filepath, event=event_b, recording_site=recording_site, basename=basename)
        if loaded_b is None:
            return
        samples_b, _ = loaded_b

    name = comparison_name(event_a=event_a, event_b=event_b, recording_site=recording_site, basename=basename)

    sample_counts = [samples_a.shape[0]] + ([samples_b.shape[0]] if samples_b is not None else [])
    if min(sample_counts) < MINIMUM_SAMPLES:
        logger.warning(
            f"Skipping significance for {name}: needs at least {MINIMUM_SAMPLES} trials or sessions "
            f"to resample, found {min(sample_counts)}."
        )
        return
    if min(sample_counts) < SMALL_SAMPLE_WARNING_THRESHOLD:
        logger.warning(
            f"Significance for {name} is computed from only {min(sample_counts)} trials or sessions; "
            f"the confidence interval is unreliable at this sample size."
        )

    significance = compute_comparison(
        samples_a=samples_a,
        samples_b=samples_b,
        timestamps=timestamps,
        minimum_consecutive_samples=minimum_consecutive_samples_for(filter_window=inputParameters["filter_window"]),
        significance_level=inputParameters["psthSignificanceAlpha"],
        num_resamples=inputParameters["psthBootstrapResamples"],
        rng=np.random.default_rng(BOOTSTRAP_SEED),
    )

    # The sample counts above are per file, but NaN padding at the window edges and
    # artifact removal can leave individual timepoints with too few usable trials to
    # bootstrap. Those come back as a NaN interval and fall out as non-significant, so a
    # largely uncomputable comparison would otherwise look like a confidently blank one.
    uncomputable = float(np.mean(~np.isfinite(significance["ci_lower"].to_numpy())))
    if uncomputable > UNCOMPUTABLE_FRACTION_WARNING_THRESHOLD:
        logger.warning(
            f"No confidence interval could be computed for {uncomputable:.0%} of the window in {name}, "
            f"where too few trials or sessions overlap. Those timepoints are reported as "
            f"not significant."
        )

    output_path = make_dir_for_psth_significance(filepath)
    write_psth_significance_to_hdf5(filepath=output_path, significance=significance, name=name)
    write_psth_significance_to_csv(filepath=output_path, significance=significance, name=name)
    logger.info(f"Significance for {name} computed.")


def plan_comparisons(filepath: str, inputParameters: dict[str, object]) -> list[tuple[str, str | None, str, str]]:
    """Enumerate every comparison to run in one output directory.

    Each event is tested against zero, and each user-named event pair is compared, once
    per recording site and metric present. Pairs are never resolved across recording sites
    or across metrics.

    Parameters
    ----------
    filepath : str
        Output directory: a session run folder or a group folder.
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    list of tuple
        ``(event_a, event_b, recording_site, basename)`` entries; ``event_b`` is None for
        a test against zero.

    Raises
    ------
    ValueError
        If a named comparison event has no results in this directory.
    """
    store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
    events = [
        event.replace("\\", "_").replace("/", "_")
        for event in event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters)
        if not is_continuous_label(event)
    ]

    comparisons = validate_psth_comparisons(
        comparisons_a=inputParameters["psthComparisonsA"],
        comparisons_b=inputParameters["psthComparisonsB"],
    )
    validate_comparison_events_available(comparisons=comparisons, available_events=events)

    planned = []
    for recording_site in recording_sites_for_output_directory(filepath):
        for metric in psth_metrics(inputParameters):
            basename = f"{metric}_{recording_site}"
            for event in events:
                planned.append((event, None, recording_site, basename))
            for event_a, event_b in comparisons:
                planned.append((event_a, event_b, recording_site, basename))

    return planned


def execute_compute_psth_significance(filepath: str, inputParameters: dict[str, object]) -> None:
    """Run every PSTH significance comparison for one output directory.

    Parameters
    ----------
    filepath : str
        Output directory: a session run folder or a group folder.
    inputParameters : dict
        Full pipeline input parameters; uses ``computePsthSignificance``,
        ``psthComparisonsA``, ``psthComparisonsB``, ``selectForComputePsth``,
        ``psthSignificanceAlpha``, ``filter_window`` and ``numberOfCores``.
    """
    if not inputParameters["computePsthSignificance"]:
        return

    planned = plan_comparisons(filepath, inputParameters)
    if not planned:
        logger.info(f"No PSTH results to test for significance in {filepath}.")
        return

    # Reported rather than refused: how many resamples to spend is the user's call, but
    # below two per tail the requested quantile falls past the most extreme resample, so
    # the interval comes back narrower than the alpha asked for.
    significance_level = inputParameters["psthSignificanceAlpha"]
    num_resamples = inputParameters["psthBootstrapResamples"]
    resamples_per_tail = num_resamples * significance_level / 2
    if resamples_per_tail < 1:
        logger.warning(
            f"psthBootstrapResamples={num_resamples} cannot resolve a {significance_level:g} "
            f"two-sided interval: that needs at least {int(np.ceil(2 / significance_level))} resamples. "
            f"The interval will be narrower than requested."
        )

    logger.info(f"Computing significance for {len(planned)} comparison(s) in {filepath}...")
    # Pinned rather than inherited, for the same reason as the step-4 pools: forking a
    # process holding other live threads can leave a logging or HDF5 lock held forever.
    spawn_context = mp.get_context("spawn")
    # Closed and joined before leaving the block, as the step-4 pools are: the context
    # manager's __exit__ calls terminate(), which blocks in waitpid() until every worker
    # is gone and hangs on one that is slow to exit. starmap has already returned, so
    # there is nothing to abort -- close() lets each worker exit on its own.
    with spawn_context.Pool(inputParameters["numberOfCores"]) as significance_pool:
        significance_pool.starmap(execute_one_comparison, zip(repeat(filepath), planned, repeat(inputParameters)))
        significance_pool.close()
        significance_pool.join()
