import glob
import logging
import math
import os

import numpy as np
from scipy import signal as ss

from .io_utils import read_hdf5, write_hdf5
from .psth_utils import create_Df

logger = logging.getLogger(__name__)


# function to create PSTH for each event using function helper_psth and save the PSTH to h5 file
def storenamePsth(filepath, event, inputParameters):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    selectForComputePsth = inputParameters["selectForComputePsth"]
    bin_psth_trials = inputParameters["bin_psth_trials"]
    use_time_or_trials = inputParameters["use_time_or_trials"]

    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    b = np.divide(np.ones((100,)), 100)
    a = 1

    # storesList = storesList
    # sampling_rate = read_hdf5(storesList[0,0], filepath, 'sampling_rate')
    nSecPrev, nSecPost = inputParameters["nSecPrev"], inputParameters["nSecPost"]
    baselineStart, baselineEnd = inputParameters["baselineCorrectionStart"], inputParameters["baselineCorrectionEnd"]
    timeInterval = inputParameters["timeInterval"]

    if "control" in event.lower() or "signal" in event.lower():
        return 0
    else:
        for i in range(len(path)):
            logger.info(f"Computing PSTH for event {event}...")
            basename = (os.path.basename(path[i])).split(".")[0]
            name_1 = basename.split("_")[-1]
            control = read_hdf5("control_" + name_1, os.path.dirname(path[i]), "data")
            if (control == 0).all() == True:
                signal = read_hdf5("signal_" + name_1, os.path.dirname(path[i]), "data")
                z_score = ss.filtfilt(b, a, signal)
                just_use_signal = True
            else:
                z_score = read_hdf5("", path[i], "data")
                just_use_signal = False
            psth, psth_baselineUncorrected, cols = helper_psth(
                z_score,
                event,
                filepath,
                nSecPrev,
                nSecPost,
                timeInterval,
                bin_psth_trials,
                use_time_or_trials,
                baselineStart,
                baselineEnd,
                name_1,
                just_use_signal,
            )

            create_Df(
                filepath,
                event + "_" + name_1 + "_baselineUncorrected",
                basename,
                psth_baselineUncorrected,
                columns=cols,
            )  # extra
            create_Df(filepath, event + "_" + name_1, basename, psth, columns=cols)
            logger.info(f"PSTH for event {event} computed.")


# *********************************** Functions used by storenamePsth *********************************** #


# helper function to make PSTH for each event
def helper_psth(
    z_score,
    event,
    filepath,
    nSecPrev,
    nSecPost,
    timeInterval,
    bin_psth_trials,
    use_time_or_trials,
    baselineStart,
    baselineEnd,
    naming,
    just_use_signal,
):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    sampling_rate = read_hdf5("timeCorrection_" + naming, filepath, "sampling_rate")[0]

    # calculate time before event timestamp and time after event timestamp
    nTsPrev = int(round(nSecPrev * sampling_rate))
    nTsPost = int(round(nSecPost * sampling_rate))

    totalTs = (-1 * nTsPrev) + nTsPost
    increment = ((-1 * nSecPrev) + nSecPost) / totalTs
    timeAxis = np.linspace(nSecPrev, nSecPost + increment, totalTs + 1)
    timeAxisNew = np.concatenate((timeAxis, timeAxis[::-1]))

    # avoid writing same data to same file in multi-processing
    # if not os.path.exists(os.path.join(filepath, 'ts_psth.h5')):
    # 	logger.info('file not exists')
    # 	create_Df(filepath, 'ts_psth', '', timeAxis)
    # 	time.sleep(2)

    ts = read_hdf5(event + "_" + naming, filepath, "ts")

    # reject timestamps for which baseline cannot be calculated because of nan values
    new_ts = []
    for i in range(ts.shape[0]):
        thisTime = ts[i]  # -1 not needed anymore
        if thisTime < abs(baselineStart):
            continue
        else:
            new_ts.append(ts[i])

    # reject burst of timestamps
    ts = np.asarray(new_ts)
    # skip the event if there are no TTLs
    if len(ts) == 0:
        new_ts = np.array([])
        logger.info(f"Warning : No TTLs present for {event}. This will cause an error in Visualization step")
    else:
        new_ts = [ts[0]]
        for i in range(1, ts.shape[0]):
            thisTime = ts[i]
            prevTime = new_ts[-1]
            diff = thisTime - prevTime
            if diff < timeInterval:
                continue
            else:
                new_ts.append(ts[i])

    # final timestamps
    ts = np.asarray(new_ts)
    nTs = ts.shape[0]

    # initialize PSTH vector
    psth = np.full((nTs, totalTs + 1), np.nan)
    psth_baselineUncorrected = np.full((nTs, totalTs + 1), np.nan)  # extra

    # for each timestamp, create trial which will be saved in a PSTH vector
    for i in range(nTs):
        thisTime = ts[i]  # -timeForLightsTurnOn
        thisIndex = int(round(thisTime * sampling_rate))
        arr = rowFormation(z_score, thisIndex, -1 * nTsPrev, nTsPost)
        if just_use_signal == True:
            res = np.subtract(arr, np.nanmean(arr))
            z_score_arr = np.divide(res, np.nanstd(arr))
            arr = z_score_arr
        else:
            arr = arr

        psth_baselineUncorrected[i, :] = arr  # extra
        psth[i, :] = baselineCorrection(filepath, arr, timeAxis, baselineStart, baselineEnd)

    write_hdf5(ts, event + "_" + naming, filepath, "ts")
    columns = list(ts)

    if use_time_or_trials == "Time (min)" and bin_psth_trials > 0:
        timestamps = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
        timestamps = np.divide(timestamps, 60)
        ts_min = np.divide(ts, 60)
        bin_steps = np.arange(timestamps[0], timestamps[-1] + bin_psth_trials, bin_psth_trials)
        indices_each_step = dict()
        for i in range(1, bin_steps.shape[0]):
            indices_each_step[f"{np.around(bin_steps[i-1],0)}-{np.around(bin_steps[i],0)}"] = np.where(
                (ts_min >= bin_steps[i - 1]) & (ts_min <= bin_steps[i])
            )[0]
    elif use_time_or_trials == "# of trials" and bin_psth_trials > 0:
        bin_steps = np.arange(0, ts.shape[0], bin_psth_trials)
        if bin_steps[-1] < ts.shape[0]:
            bin_steps = np.concatenate((bin_steps, [ts.shape[0]]), axis=0)
        indices_each_step = dict()
        for i in range(1, bin_steps.shape[0]):
            indices_each_step[f"{bin_steps[i-1]}-{bin_steps[i]}"] = np.arange(bin_steps[i - 1], bin_steps[i])
    else:
        indices_each_step = dict()

    psth_bin, psth_bin_baselineUncorrected = [], []
    if indices_each_step:
        keys = list(indices_each_step.keys())
        for k in keys:
            # no trials in a given bin window, just put all the nan values
            if indices_each_step[k].shape[0] == 0:
                psth_bin.append(np.full(psth.shape[1], np.nan))
                psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
                psth_bin.append(np.full(psth.shape[1], np.nan))
                psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
            else:
                index = indices_each_step[k]
                arr = psth[index, :]
                #  mean of bins
                psth_bin.append(np.nanmean(psth[index, :], axis=0))
                psth_bin_baselineUncorrected.append(np.nanmean(psth_baselineUncorrected[index, :], axis=0))
                psth_bin.append(np.nanstd(psth[index, :], axis=0) / math.sqrt(psth[index, :].shape[0]))
                # error of bins
                psth_bin_baselineUncorrected.append(
                    np.nanstd(psth_baselineUncorrected[index, :], axis=0)
                    / math.sqrt(psth_baselineUncorrected[index, :].shape[0])
                )

            # adding column names
            columns.append(f"bin_({k})")
            columns.append(f"bin_err_({k})")

        psth = np.concatenate((psth, psth_bin), axis=0)
        psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, psth_bin_baselineUncorrected), axis=0)

    timeAxis = timeAxis.reshape(1, -1)
    psth = np.concatenate((psth, timeAxis), axis=0)
    psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, timeAxis), axis=0)
    columns.append("timestamps")

    return psth, psth_baselineUncorrected, columns


# ***************************** Functions used by helper_psth ***************************** #


# function to create PSTH trials corresponding to each event timestamp
def rowFormation(z_score, thisIndex, nTsPrev, nTsPost):

    if nTsPrev < thisIndex and z_score.shape[0] > (thisIndex + nTsPost):
        res = z_score[thisIndex - nTsPrev - 1 : thisIndex + nTsPost]
    elif nTsPrev >= thisIndex and z_score.shape[0] > (thisIndex + nTsPost):
        mismatch = nTsPrev - thisIndex + 1
        res = np.zeros(nTsPrev + nTsPost + 1)
        res[:mismatch] = np.nan
        res[mismatch:] = z_score[: thisIndex + nTsPost]
    elif nTsPrev >= thisIndex and z_score.shape[0] < (thisIndex + nTsPost):
        mismatch1 = nTsPrev - thisIndex + 1
        mismatch2 = (thisIndex + nTsPost) - z_score.shape[0]
        res1 = np.full(mismatch1, np.nan)
        res2 = z_score
        res3 = np.full(mismatch2, np.nan)
        res = np.concatenate((res1, np.concatenate((res2, res3))))
    else:
        mismatch = (thisIndex + nTsPost) - z_score.shape[0]
        res1 = np.zeros(mismatch)
        res1[:] = np.nan
        res2 = z_score[thisIndex - nTsPrev - 1 : z_score.shape[0]]
        res = np.concatenate((res2, res1))

    return res


# function to calculate baseline for each PSTH trial and do baseline correction
def baselineCorrection(filepath, arr, timeAxis, baselineStart, baselineEnd):

    # timeAxis = read_Df(filepath, 'ts_psth', '')
    # timeAxis = np.asarray(timeAxis).reshape(-1)
    baselineStrtPt = np.where(timeAxis >= baselineStart)[0]
    baselineEndPt = np.where(timeAxis >= baselineEnd)[0]

    # logger.info(baselineStrtPt[0], baselineEndPt[0])
    if baselineStart == 0 and baselineEnd == 0:
        return arr

    baseline = np.nanmean(arr[baselineStrtPt[0] : baselineEndPt[0]])
    baselineSub = np.subtract(arr, baseline)

    return baselineSub
