import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import (
    make_dir_for_cross_correlation,
    recording_site_from_preprocessed_label,
)
from .psth_utils import create_Df_for_psth, getCorrCombinations
from ..utils.utils import read_Df

logger = logging.getLogger(__name__)


def average_psth_for_group(
    *, member_run_folders: list[str], event: str, group_folder: str, inputParameters: dict[str, object]
) -> bool:
    """
    Average PSTH, peak/AUC, and cross-correlation results across a group's member runs.

    Parameters
    ----------
    member_run_folders : list of str
        Output (run) directories holding the precomputed PSTH files to average.
    event : str
        Event label to average across the member runs.
    group_folder : str
        Group output directory the averaged results are written into.
    inputParameters : dict
        Analysis configuration dictionary; must include ``'selectForComputePsth'``.

    Returns
    -------
    bool
        Whether a PSTH average was written for ``event``. False when no member run
        holds a PSTH for it.
    """

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    logger.debug("Averaging group of data...")
    path = []
    selectForComputePsth = inputParameters["selectForComputePsth"]
    run_folder = group_folder

    # combining paths to all the selected folders for doing average
    for i in range(len(member_run_folders)):
        if selectForComputePsth == "z_score":
            matched_paths = list(Path(member_run_folders[i]).glob("z_score_*"))
        elif selectForComputePsth == "dff":
            matched_paths = list(Path(member_run_folders[i]).glob("dff_*"))
        else:
            matched_paths = list(Path(member_run_folders[i]).glob("z_score_*")) + list(
                Path(member_run_folders[i]).glob("dff_*")
            )

        for j in range(len(matched_paths)):
            basename = matched_paths[j].name.split(".")[0]
            name_1 = recording_site_from_preprocessed_label(basename)
            entry = [member_run_folders[i], event + "_" + name_1, basename]
            path.append(entry)

    # processing of all the paths
    naming = []
    for i in range(len(path)):
        naming.append(path[i][2])
    naming = np.unique(np.asarray(naming))

    # Size by the number of unique basenames across all folders so that mismatched
    # or non-overlapping store_ids across sessions do not cause an IndexError.
    new_path = [[] for _ in range(len(naming))]
    for i in range(len(path)):
        index = np.where(naming == path[i][2])[0][0]
        new_path[index].append(path[i])

    # read PSTH for each event and make the average of it. Save the final output to the group folder.
    wrote_psth = False
    for i in range(len(new_path)):
        psth, psth_bins = [], []
        columns = []
        bin_columns = []
        session_entries = new_path[i]
        for j in range(len(session_entries)):
            psth_path = Path(session_entries[j][0]) / (session_entries[j][1] + f"_{session_entries[j][2]}.h5")
            if not psth_path.exists():
                continue
            else:
                # read_Df arguments are filepath, event, name
                df = read_Df(session_entries[j][0], session_entries[j][1], session_entries[j][2])
                column_names = list(df.columns)
                regex = re.compile("bin_[(]")
                bin_columns = [column_names[i] for i in range(len(column_names)) if regex.match(column_names[i])]
                psth.append(np.asarray(df["mean"]))
                columns.append(Path(session_entries[j][0]).name)
                if len(bin_columns) > 0:
                    psth_bins.append(df[bin_columns])

        if len(psth) == 0:
            logger.warning(
                "No PSTH files found for event %r (basename %r, selectForComputePsth=%r) across the selected folders; "
                "skipping average for this event.",
                event,
                session_entries[0][2],
                selectForComputePsth,
            )
            continue

        if len(bin_columns) > 0:
            df_bins = pd.concat(psth_bins, axis=1)
            df_bins_mean = df_bins.T.groupby(df_bins.columns).mean().T
            df_bins_err = df_bins.T.groupby(df_bins.columns).std().T / math.sqrt(df_bins.shape[1])
            error_column_names = list(df_bins_err.columns)
            error_rename_map = {}
            for column_name in error_column_names:
                name_parts = column_name.split("_")
                error_rename_map[column_name] = f"{name_parts[0]}_err_{name_parts[1]}"
            df_bins_err = df_bins_err.rename(columns=error_rename_map)
            columns = columns + list(df_bins_mean.columns) + list(df_bins_err.columns)
            df_bins_mean_err = pd.concat([df_bins_mean, df_bins_err], axis=1).T
            psth, df_bins_mean_err = np.asarray(psth), np.asarray(df_bins_mean_err)
            psth = np.concatenate((psth, df_bins_mean_err), axis=0)
        else:
            psth = psth_shape_check(psth)
            psth = np.asarray(psth)

        timestamps = np.asarray(df["timestamps"]).reshape(1, -1)
        psth = np.concatenate((psth, timestamps), axis=0)
        columns = columns + ["timestamps"]
        create_Df_for_psth(run_folder, session_entries[j][1], session_entries[j][2], psth, columns=columns)
        wrote_psth = True

    # read PSTH peak and area for each event and combine them. Save the final output to an average folder
    for i in range(len(new_path)):
        peak_area_frames = []
        row_indices = []
        session_entries = new_path[i]
        for j in range(len(session_entries)):
            peak_auc_path = Path(session_entries[j][0]) / (
                "peak_AUC_" + session_entries[j][1] + "_" + session_entries[j][2] + ".h5"
            )
            if not peak_auc_path.exists():
                continue
            else:
                df = read_Df_area_peak(session_entries[j][0], session_entries[j][1] + "_" + session_entries[j][2])
                peak_area_frames.append(df)
                row_indices.append(list(df.index))

        if len(peak_area_frames) == 0:
            logger.warning(
                "No peak/AUC files found for event %r (basename %r) across the selected folders; skipping peak/AUC "
                "average for this event.",
                event,
                session_entries[0][2],
            )
            continue
        row_indices = list(np.concatenate(row_indices))
        new_df = pd.concat(peak_area_frames, axis=0)
        new_df.to_csv(
            Path(run_folder) / f"peak_AUC_{session_entries[j][1]}_{session_entries[j][2]}.csv",
            index=row_indices,
        )
        new_df.to_hdf(
            Path(run_folder) / f"peak_AUC_{session_entries[j][1]}_{session_entries[j][2]}.h5",
            key="df",
            mode="w",
            index=row_indices,
        )

    # read cross-correlation files and combine them. Save the final output to an average folder
    type = []
    for i in range(len(member_run_folders)):
        _, session_types = getCorrCombinations(member_run_folders[i], inputParameters)
        type.append(session_types)

    type = np.unique(np.array(type))
    for i in range(len(type)):
        corr = []
        columns = []
        df = None
        for j in range(len(member_run_folders)):
            corr_info, _ = getCorrCombinations(member_run_folders[j], inputParameters)
            for k in range(1, len(corr_info)):
                correlation_folder = Path(member_run_folders[j]) / "cross_correlation_output"
                name = type[i] + "_" + corr_info[k - 1] + "_" + corr_info[k]
                if not (correlation_folder / ("corr_" + event + "_" + name + ".h5")).exists():
                    continue
                else:
                    df = read_Df(correlation_folder, "corr_" + event, name)
                    corr.append(df["mean"])
                    columns.append(Path(member_run_folders[j]).name)

        if not isinstance(df, pd.DataFrame):
            break

        corr = np.array(corr)
        timestamps = np.array(df["timestamps"]).reshape(1, -1)
        corr = np.concatenate((corr, timestamps), axis=0)
        columns.append("timestamps")
        create_Df_for_psth(
            make_dir_for_cross_correlation(run_folder),
            "corr_" + event,
            type[i] + "_" + corr_info[k - 1] + "_" + corr_info[k],
            corr,
            columns=columns,
        )

    logger.info("Group of data averaged.")

    return wrote_psth


def psth_shape_check(psth: list[np.ndarray]) -> list[np.ndarray]:
    """
    Pad or truncate PSTH trial arrays so they all share the same length.

    Parameters
    ----------
    psth : list of np.ndarray
        List of 1-D PSTH mean arrays, potentially with differing lengths.

    Returns
    -------
    psth : list of np.ndarray
        List of arrays all truncated or NaN-padded to the length of the last element.
    """

    each_ln = []
    for i in range(len(psth)):
        each_ln.append(psth[i].shape[0])

    each_ln = np.asarray(each_ln)
    keep_ln = each_ln[-1]

    for i in range(len(psth)):
        if psth[i].shape[0] > keep_ln:
            psth[i] = psth[i][:keep_ln]
        elif psth[i].shape[0] < keep_ln:
            psth[i] = np.append(psth[i], np.full(keep_ln - len(psth[i]), np.nan))
        else:
            psth[i] = psth[i]

    return psth


def read_Df_area_peak(filepath: str, name: str) -> pd.DataFrame:
    """
    Read a peak/AUC HDF5 file and return its DataFrame.

    Parameters
    ----------
    filepath : str
        Directory containing the ``peak_AUC_<name>.h5`` file.
    name : str
        Filename stem (without the ``peak_AUC_`` prefix or ``.h5`` suffix).

    Returns
    -------
    df : pd.DataFrame
        DataFrame of peak and area-under-curve metrics.
    """
    output_path = Path(filepath) / ("peak_AUC_" + name + ".h5")
    df = pd.read_hdf(output_path, key="df", mode="r")

    return df
