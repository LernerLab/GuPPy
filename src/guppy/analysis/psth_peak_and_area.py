import glob
import logging
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from .io_utils import read_Df, read_hdf5

logger = logging.getLogger(__name__)


# function to compute PSTH peak and area using the function helperPSTHPeakAndArea save the values to h5 and csv files.
def findPSTHPeakAndArea(filepath, event, inputParameters):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    # sampling_rate = read_hdf5(storesList[0,0], filepath, 'sampling_rate')
    peak_startPoint = inputParameters["peak_startPoint"]
    peak_endPoint = inputParameters["peak_endPoint"]
    selectForComputePsth = inputParameters["selectForComputePsth"]

    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    if "control" in event.lower() or "signal" in event.lower():
        return 0
    else:
        for i in range(len(path)):
            logger.info(f"Computing peak and area for PSTH mean signal for event {event}...")
            basename = (os.path.basename(path[i])).split(".")[0]
            name_1 = basename.split("_")[-1]
            sampling_rate = read_hdf5("timeCorrection_" + name_1, filepath, "sampling_rate")[0]
            psth = read_Df(filepath, event + "_" + name_1, basename)
            cols = list(psth.columns)
            regex = re.compile("bin_[(]")
            bin_names = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
            regex_trials = re.compile("[+-]?([0-9]*[.])?[0-9]+")
            trials_names = [cols[i] for i in range(len(cols)) if regex_trials.match(cols[i])]
            psth_mean_bin_names = trials_names + bin_names + ["mean"]
            psth_mean_bin_mean = np.asarray(psth[psth_mean_bin_names])
            timestamps = np.asarray(psth["timestamps"]).ravel()  # np.asarray(read_Df(filepath, 'ts_psth', '')).ravel()
            peak_area = helperPSTHPeakAndArea(
                psth_mean_bin_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint
            )  # peak, area =
            # arr = np.array([[peak, area]])
            fileName = [os.path.basename(os.path.dirname(filepath))]
            index = [fileName[0] + "_" + s for s in psth_mean_bin_names]
            create_Df_area_peak(
                filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index
            )  # columns=['peak', 'area']
            create_csv_area_peak(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
            logger.info(f"Peak and Area for PSTH mean signal for event {event} computed.")


def helperPSTHPeakAndArea(psth_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint):

    peak_startPoint = np.asarray(peak_startPoint)
    peak_endPoint = np.asarray(peak_endPoint)

    peak_startPoint = peak_startPoint[~np.isnan(peak_startPoint)]
    peak_endPoint = peak_endPoint[~np.isnan(peak_endPoint)]

    if peak_startPoint.shape[0] != peak_endPoint.shape[0]:
        logger.error("Number of Peak Start Time and Peak End Time are unequal.")
        raise Exception("Number of Peak Start Time and Peak End Time are unequal.")

    if np.less_equal(peak_endPoint, peak_startPoint).any() == True:
        logger.error(
            "Peak End Time is lesser than or equal to Peak Start Time. Please check the Peak parameters window."
        )
        raise Exception(
            "Peak End Time is lesser than or equal to Peak Start Time. Please check the Peak parameters window."
        )

    peak_area = OrderedDict()

    if peak_startPoint.shape[0] == 0 or peak_endPoint.shape[0] == 0:
        peak_area["peak"] = np.nan
        peak_area["area"] = np.nan

    for i in range(peak_startPoint.shape[0]):
        startPtForPeak = np.where(timestamps >= peak_startPoint[i])[0]
        endPtForPeak = np.where(timestamps >= peak_endPoint[i])[0]
        if len(startPtForPeak) >= 1 and len(endPtForPeak) >= 1:
            peakPoint_pos = startPtForPeak[0] + np.argmax(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peakPoint_neg = startPtForPeak[0] + np.argmin(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peak_area["peak_pos_" + str(i + 1)] = np.amax(psth_mean[peakPoint_pos], axis=0)
            peak_area["peak_neg_" + str(i + 1)] = np.amin(psth_mean[peakPoint_neg], axis=0)
            peak_area["area_" + str(i + 1)] = np.trapz(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
        else:
            peak_area["peak_" + str(i + 1)] = np.nan
            peak_area["area_" + str(i + 1)] = np.nan

    return peak_area


def create_Df_area_peak(filepath, arr, name, index=[]):

    op = os.path.join(filepath, "peak_AUC_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index)

    df.to_hdf(op, key="df", mode="w")


def create_csv_area_peak(filepath, arr, name, index=[]):
    op = os.path.join(filepath, "peak_AUC_" + name + ".csv")
    df = pd.DataFrame(arr, index=index)

    df.to_csv(op)
