import glob
import logging
import math
import os
import re

import numpy as np
import pandas as pd

from .io_utils import recording_site_from_preprocessed_label

logger = logging.getLogger(__name__)


def create_Df_for_psth(filepath: str, event: str, name: str, psth: np.ndarray, columns: list[object]) -> None:
    """
    Build a PSTH DataFrame (with mean/error columns) and save it as an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory where the ``.h5`` file is written.
    event : str
        Event label; backslashes and forward-slashes are replaced with underscores.
    name : str
        Channel name suffix appended to the filename.
    psth : np.ndarray
        2-D PSTH matrix (trials × time-points).
    columns : list, optional
        Column labels for the trials axis. Default is an empty list.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        output_path = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        output_path = os.path.join(filepath, event + ".h5")

    # removing psth binned trials
    columns = np.array(columns, dtype="str")
    regex = re.compile("bin_*")
    single_trials = columns[[i for i in range(len(columns)) if not regex.match(columns[i])]]
    single_trials_index = [i for i in range(len(single_trials)) if single_trials[i] != "timestamps"]

    psth = psth.T
    if psth.ndim > 1:
        mean = np.nanmean(psth[:, single_trials_index], axis=1).reshape(-1, 1)
        standard_error = np.nanstd(psth[:, single_trials_index], axis=1) / math.sqrt(
            psth[:, single_trials_index].shape[1]
        )
        standard_error = standard_error.reshape(-1, 1)
        psth = np.hstack((psth, mean))
        psth = np.hstack((psth, standard_error))

    columns = np.asarray(columns)
    columns = np.append(columns, ["mean", "err"])
    df = pd.DataFrame(psth, index=None, columns=list(columns), dtype="float32")

    df.to_hdf(output_path, key="df", mode="w")


def match_trials_by_timestamp(
    labels_a: list[object], labels_b: list[object]
) -> tuple[np.ndarray, np.ndarray, list[object]]:
    """Pair two recording sites' PSTH trial columns by the event they came from.

    Cross-correlation runs only with the ``replace with NaN`` artifact method, so
    both recording sites keep full-length signals sampled on the same time base; the
    same event therefore has near-identical timestamps in each recording site,
    differing only by sub-sample jitter. Independent artifact removal can still leave
    one recording site without a boundary trial the other kept, so trials are matched
    by nearest timestamp within half the smallest inter-event gap. Burst rejection
    keeps events well separated, so a genuine match is unambiguous and an unmatched
    trial (present in one recording site only) is dropped from both. Non-numeric
    columns (e.g. binned-trial aggregates) are matched by exact label.

    Parameters
    ----------
    labels_a, labels_b : list
        Trial-column labels for the two recording sites. Trial columns are event
        timestamps (numeric, possibly as strings when read back from HDF5);
        other columns (binned aggregates) are matched by exact label.

    Returns
    -------
    indices_a, indices_b : np.ndarray
        Integer index arrays into ``labels_a`` / ``labels_b`` selecting the
        matched trials, in ascending-timestamp order (numeric matches first,
        then exact-label matches in ``labels_a`` order).
    matched_labels : list
        The label from ``labels_a`` for each matched pair.
    """

    def split_numeric(labels: list[object]) -> tuple[list[tuple[int, float]], list[tuple[int, object]]]:
        numeric, other = [], []
        for index, label in enumerate(labels):
            try:
                numeric.append((index, float(label)))
            except (TypeError, ValueError):
                other.append((index, label))
        return numeric, other

    numeric_a, other_a = split_numeric(labels_a)
    numeric_b, other_b = split_numeric(labels_b)

    indices_a: list[int] = []
    indices_b: list[int] = []
    matched_labels: list[object] = []

    if numeric_a and numeric_b:
        timestamps_a = np.array([timestamp for _, timestamp in numeric_a])
        timestamps_b = np.array([timestamp for _, timestamp in numeric_b])
        column_indices_a = np.array([index for index, _ in numeric_a])
        column_indices_b = np.array([index for index, _ in numeric_b])

        inter_event_gaps = [np.diff(np.sort(ts)) for ts in (timestamps_a, timestamps_b) if ts.size > 1]
        tolerance = 0.5 * min(np.min(gaps) for gaps in inter_event_gaps) if inter_event_gaps else np.inf

        used_b: set[int] = set()
        for order in np.argsort(timestamps_a):
            distances = np.abs(timestamps_b - timestamps_a[order])
            nearest = int(np.argmin(distances))
            if distances[nearest] <= tolerance and column_indices_b[nearest] not in used_b:
                used_b.add(int(column_indices_b[nearest]))
                indices_a.append(int(column_indices_a[order]))
                indices_b.append(int(column_indices_b[nearest]))
                matched_labels.append(labels_a[int(column_indices_a[order])])

    label_b_to_index = {label: index for index, label in other_b}
    for index, label in other_a:
        if label in label_b_to_index:
            indices_a.append(index)
            indices_b.append(label_b_to_index[label])
            matched_labels.append(label)

    return np.array(indices_a, dtype=int), np.array(indices_b, dtype=int), matched_labels


def create_Df_for_cross_correlation(
    filepath: str, event: str, name: str, psth: np.ndarray, columns: list[object]
) -> None:
    """
    Build a cross-correlation DataFrame (with mean/error columns) and save it as an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory where the ``.h5`` file is written.
    event : str
        Event label used to build the filename.
    name : str
        Channel name suffix appended to the filename.
    psth : np.ndarray
        2-D cross-correlation matrix (trials × lags).
    columns : list, optional
        Column labels for the trials axis. Default is an empty list.
    """
    if name:
        output_path = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        output_path = os.path.join(filepath, event + ".h5")

    # removing psth binned trials
    columns = list(np.array(columns, dtype="str"))
    regex = re.compile("bin_*")
    single_trials_index = [i for i in range(len(columns)) if not regex.match(columns[i])]
    single_trials_index = [i for i in range(len(columns)) if columns[i] != "timestamps"]

    psth = psth.T
    if psth.ndim > 1:
        mean = np.nanmean(psth[:, single_trials_index], axis=1).reshape(-1, 1)
        standard_error = np.nanstd(psth[:, single_trials_index], axis=1) / math.sqrt(
            psth[:, single_trials_index].shape[1]
        )
        standard_error = standard_error.reshape(-1, 1)
        psth = np.hstack((psth, mean))
        psth = np.hstack((psth, standard_error))

    columns = np.asarray(columns)
    columns = np.append(columns, ["mean", "err"])
    df = pd.DataFrame(psth, index=None, columns=columns, dtype="float32")

    df.to_hdf(output_path, key="df", mode="w")


def getCorrCombinations(filepath: str, inputParameters: dict[str, object]) -> tuple[list[str], list[str]]:
    """
    Determine which channel pairs to cross-correlate in a session directory.

    Parameters
    ----------
    filepath : str
        Session output directory containing z-score or dff HDF5 files.
    inputParameters : dict
        Analysis configuration; must include ``'selectForComputePsth'``.

    Returns
    -------
    corr_info : list of str
        Ordered list of channel suffixes to correlate; circular when more than two.
    type : list of str
        Unique channel-type prefixes (e.g. ``['z_score']``).
    """
    selectForComputePsth = inputParameters["selectForComputePsth"]
    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    names = list()
    type = list()
    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        names.append(recording_site_from_preprocessed_label(basename))
        type.append((os.path.basename(path[i])).split(".")[0].split("_" + names[-1], 1)[0])

    names = list(np.unique(np.array(names)))
    type = list(np.unique(np.array(type)))

    corr_info = list()
    if len(names) <= 1:
        logger.info("Cross-correlation cannot be computed because only one signal is present.")
        return names, type
    elif len(names) == 2:
        corr_info = names
    else:
        corr_info = names
        corr_info.append(names[0])

    return corr_info, type
