import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def compute_psth(
    z_score: np.ndarray,
    event: str,
    filepath: str,
    nSecPrev: float,
    nSecPost: float,
    timeInterval: float,
    bin_psth_trials: float,
    use_time_or_trials: str,
    baselineStart: float,
    baselineEnd: float,
    naming: str,
    just_use_signal: bool,
    sampling_rate: float,
    ts: np.ndarray,
    corrected_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[object], np.ndarray]:
    """
    Build a peri-stimulus time histogram (PSTH) matrix for one event.

    Parameters
    ----------
    z_score : np.ndarray
        1-D z-scored signal array.
    event : str
        Event label used for file naming.
    filepath : str
        Path to the session output directory.
    nSecPrev : float
        Seconds before each event timestamp to include in each trial.
    nSecPost : float
        Seconds after each event timestamp to include in each trial.
    timeInterval : float
        Minimum inter-event interval (s); events closer than this are skipped.
    bin_psth_trials : int or float
        Bin width for trial averaging; 0 disables binning.
    use_time_or_trials : str
        ``'Time (min)'`` or ``'# of trials'`` — selects binning mode.
    baselineStart : float
        Start of the baseline window (s) for baseline correction.
    baselineEnd : float
        End of the baseline window (s) for baseline correction.
    naming : str
        Channel name suffix used to look up timestamps in HDF5.
    just_use_signal : bool
        When True, z-score each trial independently rather than using the pre-computed z-score.
    sampling_rate : float
        Sampling rate in Hz.
    ts : np.ndarray
        Event timestamp array (s).
    corrected_timestamps : np.ndarray
        Full corrected photometry timestamp array (recording-start basis). Its first
        value is the signal start used to map event times to z-score sample indices,
        and the array is also used for time-based binning.

    Returns
    -------
    psth : np.ndarray
        PSTH matrix (trials × time-points), with optional bin rows appended.
    psth_baselineUncorrected : np.ndarray
        Same shape as ``psth`` but without baseline correction.
    columns : list
        Column labels corresponding to ``psth``; last entry is ``'timestamps'``.
    ts : np.ndarray
        Filtered event timestamps actually used to build the PSTH.
    """

    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    # calculate time before event timestamp and time after event timestamp
    nTsPrev = int(round(nSecPrev * sampling_rate))
    nTsPost = int(round(nSecPost * sampling_rate))

    totalTs = (-1 * nTsPrev) + nTsPost
    increment = ((-1 * nSecPrev) + nSecPost) / totalTs
    timeAxis = np.linspace(nSecPrev, nSecPost + increment, totalTs + 1)
    timeAxisNew = np.concatenate((timeAxis, timeAxis[::-1]))

    # reject timestamps for which baseline cannot be calculated because of nan values.
    # Events are on the recording-start basis (same as corrected_timestamps), so the
    # time available before an event is measured from the signal start, corrected_timestamps[0].
    stream_start = corrected_timestamps[0]
    new_ts = []
    for i in range(ts.shape[0]):
        thisTime = ts[i]
        if (thisTime - stream_start) < abs(baselineStart):
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
        thisTime = ts[i]
        # Events share the recording-start basis with the continuous stream, so map an
        # event time to its positional index in z_score relative to the signal start
        # (z_score[0] corresponds to corrected_timestamps[0]).
        thisIndex = int(round((thisTime - stream_start) * sampling_rate))
        # nSecPrev (and therefore nTsPrev) is negative by convention; flip to a positive
        # sample count for rowFormation, which expects nTsPrev as a positive lookback length.
        arr = rowFormation(z_score, thisIndex, -1 * nTsPrev, nTsPost)
        if just_use_signal == True:
            res = np.subtract(arr, np.nanmean(arr))
            z_score_arr = np.divide(res, np.nanstd(arr))
            arr = z_score_arr
        else:
            arr = arr

        psth_baselineUncorrected[i, :] = arr  # extra
        psth[i, :] = baselineCorrection(arr, timeAxis, baselineStart, baselineEnd)

    columns = list(ts)

    if use_time_or_trials == "Time (min)" and bin_psth_trials > 0:
        corrected_timestamps = np.divide(corrected_timestamps, 60)
        ts_min = np.divide(ts, 60)
        bin_steps = np.arange(corrected_timestamps[0], corrected_timestamps[-1] + bin_psth_trials, bin_psth_trials)
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

    return psth, psth_baselineUncorrected, columns, ts


def rowFormation(z_score: np.ndarray, thisIndex: int, nTsPrev: int, nTsPost: int) -> np.ndarray:
    """
    Extract one PSTH trial from the z-score array, padding with NaN at boundaries.

    Parameters
    ----------
    z_score : np.ndarray
        1-D z-scored signal array.
    thisIndex : int
        Sample index of the event timestamp in ``z_score``.
    nTsPrev : int
        Number of samples before the event to include.
    nTsPost : int
        Number of samples after the event to include.

    Returns
    -------
    res : np.ndarray
        1-D trial array of length ``nTsPrev + nTsPost + 1``, NaN-padded where the
        signal does not exist.
    """

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


def baselineCorrection(arr: np.ndarray, timeAxis: np.ndarray, baselineStart: float, baselineEnd: float) -> np.ndarray:
    """
    Subtract the mean baseline from a single PSTH trial.

    Parameters
    ----------
    arr : np.ndarray
        1-D trial array aligned with ``timeAxis``.
    timeAxis : np.ndarray
        1-D time axis (s) for the trial window.
    baselineStart : float
        Start of the baseline window (s).
    baselineEnd : float
        End of the baseline window (s).

    Returns
    -------
    baselineSub : np.ndarray
        Trial array with the mean baseline subtracted; ``arr`` unchanged when
        both ``baselineStart`` and ``baselineEnd`` are zero.
    """
    baselineStrtPt = np.where(timeAxis >= baselineStart)[0]
    baselineEndPt = np.where(timeAxis >= baselineEnd)[0]

    if baselineStart == 0 and baselineEnd == 0:
        return arr

    baseline = np.nanmean(arr[baselineStrtPt[0] : baselineEndPt[0]])
    baselineSub = np.subtract(arr, baseline)

    return baselineSub
