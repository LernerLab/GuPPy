import glob
import logging
import math
import os
import re

import numpy as np
import pandas as pd

from .io_utils import (
    make_dir_for_cross_correlation,
    makeAverageDir,
    write_hdf5,
)
from .psth_utils import create_Df_for_psth, getCorrCombinations
from ..utils.utils import read_Df

logger = logging.getLogger(__name__)


# function to compute average of group of recordings
def averageForGroup(folderNames, event, inputParameters):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    logger.debug("Averaging group of data...")
    path = []
    abspath = inputParameters["abspath"]
    selectForComputePsth = inputParameters["selectForComputePsth"]
    path_temp_len = []
    op = makeAverageDir(abspath)

    # combining paths to all the selected folders for doing average
    for i in range(len(folderNames)):
        if selectForComputePsth == "z_score":
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*"))
        elif selectForComputePsth == "dff":
            path_temp = glob.glob(os.path.join(folderNames[i], "dff_*"))
        else:
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*")) + glob.glob(
                os.path.join(folderNames[i], "dff_*")
            )

        path_temp_len.append(len(path_temp))
        # path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*'))
        for j in range(len(path_temp)):
            basename = (os.path.basename(path_temp[j])).split(".")[0]
            write_hdf5(np.array([]), basename, op, "data")
            name_1 = basename.split("_")[-1]
            temp = [folderNames[i], event + "_" + name_1, basename]
            path.append(temp)

    # processing of all the paths
    path_temp_len = np.asarray(path_temp_len)
    max_len = np.argmax(path_temp_len)

    naming = []
    for i in range(len(path)):
        naming.append(path[i][2])
    naming = np.unique(np.asarray(naming))

    new_path = [[] for _ in range(path_temp_len[max_len])]
    for i in range(len(path)):
        idx = np.where(naming == path[i][2])[0][0]
        new_path[idx].append(path[i])

    # read PSTH for each event and make the average of it. Save the final output to an average folder.
    for i in range(len(new_path)):
        psth, psth_bins = [], []
        columns = []
        bins_cols = []
        temp_path = new_path[i]
        for j in range(len(temp_path)):
            # logger.info(os.path.join(temp_path[j][0], temp_path[j][1]+'_{}.h5'.format(temp_path[j][2])))
            if not os.path.exists(os.path.join(temp_path[j][0], temp_path[j][1] + "_{}.h5".format(temp_path[j][2]))):
                continue
            else:
                df = read_Df(temp_path[j][0], temp_path[j][1], temp_path[j][2])  # filepath, event, name
                cols = list(df.columns)
                regex = re.compile("bin_[(]")
                bins_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
                psth.append(np.asarray(df["mean"]))
                columns.append(os.path.basename(temp_path[j][0]))
                if len(bins_cols) > 0:
                    psth_bins.append(df[bins_cols])

        if len(psth) == 0:
            logger.warning("Something is wrong with the file search pattern.")
            continue

        if len(bins_cols) > 0:
            df_bins = pd.concat(psth_bins, axis=1)
            df_bins_mean = df_bins.groupby(by=df_bins.columns, axis=1).mean()
            df_bins_err = df_bins.groupby(by=df_bins.columns, axis=1).std() / math.sqrt(df_bins.shape[1])
            cols_err = list(df_bins_err.columns)
            dict_err = {}
            for i in cols_err:
                split = i.split("_")
                dict_err[i] = "{}_err_{}".format(split[0], split[1])
            df_bins_err = df_bins_err.rename(columns=dict_err)
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
        create_Df_for_psth(op, temp_path[j][1], temp_path[j][2], psth, columns=columns)

    # read PSTH peak and area for each event and combine them. Save the final output to an average folder
    for i in range(len(new_path)):
        arr = []
        index = []
        temp_path = new_path[i]
        for j in range(len(temp_path)):
            if not os.path.exists(
                os.path.join(temp_path[j][0], "peak_AUC_" + temp_path[j][1] + "_" + temp_path[j][2] + ".h5")
            ):
                continue
            else:
                df = read_Df_area_peak(temp_path[j][0], temp_path[j][1] + "_" + temp_path[j][2])
                arr.append(df)
                index.append(list(df.index))

        if len(arr) == 0:
            logger.warning("Something is wrong with the file search pattern.")
            continue
        index = list(np.concatenate(index))
        new_df = pd.concat(arr, axis=0)  # os.path.join(filepath, 'peak_AUC_'+name+'.csv')
        new_df.to_csv(os.path.join(op, "peak_AUC_{}_{}.csv".format(temp_path[j][1], temp_path[j][2])), index=index)
        new_df.to_hdf(
            os.path.join(op, "peak_AUC_{}_{}.h5".format(temp_path[j][1], temp_path[j][2])),
            key="df",
            mode="w",
            index=index,
        )

    # read cross-correlation files and combine them. Save the final output to an average folder
    type = []
    for i in range(len(folderNames)):
        _, temp_type = getCorrCombinations(folderNames[i], inputParameters)
        type.append(temp_type)

    type = np.unique(np.array(type))
    for i in range(len(type)):
        corr = []
        columns = []
        df = None
        for j in range(len(folderNames)):
            corr_info, _ = getCorrCombinations(folderNames[j], inputParameters)
            for k in range(1, len(corr_info)):
                path = os.path.join(
                    folderNames[j],
                    "cross_correlation_output",
                    "corr_" + event + "_" + type[i] + "_" + corr_info[k - 1] + "_" + corr_info[k],
                )
                if not os.path.exists(path + ".h5"):
                    continue
                else:
                    df = read_Df(
                        os.path.join(folderNames[j], "cross_correlation_output"),
                        "corr_" + event,
                        type[i] + "_" + corr_info[k - 1] + "_" + corr_info[k],
                    )
                    corr.append(df["mean"])
                    columns.append(os.path.basename(folderNames[j]))

        if not isinstance(df, pd.DataFrame):
            break

        corr = np.array(corr)
        timestamps = np.array(df["timestamps"]).reshape(1, -1)
        corr = np.concatenate((corr, timestamps), axis=0)
        columns.append("timestamps")
        create_Df_for_psth(
            make_dir_for_cross_correlation(op),
            "corr_" + event,
            type[i] + "_" + corr_info[k - 1] + "_" + corr_info[k],
            corr,
            columns=columns,
        )

    logger.info("Group of data averaged.")


def psth_shape_check(psth):

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


def read_Df_area_peak(filepath, name):
    op = os.path.join(filepath, "peak_AUC_" + name + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df
