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
    event_timestamps: np.ndarray,
    corrected_timestamps: np.ndarray,
    timeForLightsTurnOn: float,
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
    event_timestamps : np.ndarray
        Event timestamp array (s).
    corrected_timestamps : np.ndarray
        Full corrected photometry timestamp array (recording-start basis), used for
        time-based binning.
    timeForLightsTurnOn : float
        Lights-on offset (s). Events are stored on the recording-start basis while
        ``z_score[0]`` corresponds to the lights-on instant, so event times are mapped
        to z-score sample indices relative to ``timeForLightsTurnOn``.

    Returns
    -------
    psth : np.ndarray
        PSTH matrix (trials × time-points), with optional bin rows appended.
    psth_baselineUncorrected : np.ndarray
        Same shape as ``psth`` but without baseline correction.
    columns : list
        Column labels corresponding to ``psth``; last entry is ``'timestamps'``.
    event_timestamps : np.ndarray
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
    # Events are on the recording-start basis; z_score[0] is the lights-on instant, so the
    # time available before an event is measured relative to timeForLightsTurnOn.
    kept_timestamps = []
    for i in range(event_timestamps.shape[0]):
        thisTime = event_timestamps[i]
        if (thisTime - timeForLightsTurnOn) < abs(baselineStart):
            continue
        else:
            kept_timestamps.append(event_timestamps[i])

    # reject burst of timestamps
    event_timestamps = np.asarray(kept_timestamps)
    # skip the event if there are no TTLs
    if len(event_timestamps) == 0:
        kept_timestamps = np.array([])
        logger.info(f"Warning : No TTLs present for {event}. This will cause an error in Visualization step")
    else:
        kept_timestamps = [event_timestamps[0]]
        for i in range(1, event_timestamps.shape[0]):
            thisTime = event_timestamps[i]
            prevTime = kept_timestamps[-1]
            diff = thisTime - prevTime
            if diff < timeInterval:
                continue
            else:
                kept_timestamps.append(event_timestamps[i])

    # final timestamps
    event_timestamps = np.asarray(kept_timestamps)
    nTs = event_timestamps.shape[0]

    # initialize PSTH vector
    psth = np.full((nTs, totalTs + 1), np.nan)
    psth_baselineUncorrected = np.full((nTs, totalTs + 1), np.nan)

    # for each timestamp, create trial which will be saved in a PSTH vector
    for i in range(nTs):
        thisTime = event_timestamps[i]
        # Events are on the recording-start basis; z_score[0] corresponds to the lights-on
        # instant, so subtract timeForLightsTurnOn to get the positional index into z_score.
        thisIndex = int(round((thisTime - timeForLightsTurnOn) * sampling_rate))
        # nSecPrev (and therefore nTsPrev) is negative by convention; flip to a positive
        # sample count for rowFormation, which expects nTsPrev as a positive lookback length.
        trial = rowFormation(z_score, thisIndex, -1 * nTsPrev, nTsPost)
        if just_use_signal == True:
            centered_trial = np.subtract(trial, np.nanmean(trial))
            trial_z_score = np.divide(centered_trial, np.nanstd(trial))
            trial = trial_z_score
        else:
            trial = trial

        psth_baselineUncorrected[i, :] = trial
        psth[i, :] = baselineCorrection(trial, timeAxis, baselineStart, baselineEnd)

    columns = list(event_timestamps)

    if use_time_or_trials == "Time (min)" and bin_psth_trials > 0:
        corrected_timestamps = np.divide(corrected_timestamps, 60)
        event_timestamps_in_mins = np.divide(event_timestamps, 60)
        bin_steps = np.arange(corrected_timestamps[0], corrected_timestamps[-1] + bin_psth_trials, bin_psth_trials)
        indices_each_step = dict()
        for i in range(1, bin_steps.shape[0]):
            indices_each_step[f"{np.around(bin_steps[i-1],0)}-{np.around(bin_steps[i],0)}"] = np.where(
                (event_timestamps_in_mins >= bin_steps[i - 1]) & (event_timestamps_in_mins <= bin_steps[i])
            )[0]
    elif use_time_or_trials == "# of trials" and bin_psth_trials > 0:
        bin_steps = np.arange(0, event_timestamps.shape[0], bin_psth_trials)
        if bin_steps[-1] < event_timestamps.shape[0]:
            bin_steps = np.concatenate((bin_steps, [event_timestamps.shape[0]]), axis=0)
        indices_each_step = dict()
        for i in range(1, bin_steps.shape[0]):
            indices_each_step[f"{bin_steps[i-1]}-{bin_steps[i]}"] = np.arange(bin_steps[i - 1], bin_steps[i])
    else:
        indices_each_step = dict()

    psth_bin, psth_bin_baselineUncorrected = [], []
    if indices_each_step:
        bin_labels = list(indices_each_step.keys())
        for bin_label in bin_labels:
            # no trials in a given bin window, just put all the nan values
            if indices_each_step[bin_label].shape[0] == 0:
                psth_bin.append(np.full(psth.shape[1], np.nan))
                psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
                psth_bin.append(np.full(psth.shape[1], np.nan))
                psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
            else:
                trial_indices = indices_each_step[bin_label]
                binned_trials = psth[trial_indices, :]
                #  mean of bins
                psth_bin.append(np.nanmean(psth[trial_indices, :], axis=0))
                psth_bin_baselineUncorrected.append(np.nanmean(psth_baselineUncorrected[trial_indices, :], axis=0))
                psth_bin.append(np.nanstd(psth[trial_indices, :], axis=0) / math.sqrt(psth[trial_indices, :].shape[0]))
                # error of bins
                psth_bin_baselineUncorrected.append(
                    np.nanstd(psth_baselineUncorrected[trial_indices, :], axis=0)
                    / math.sqrt(psth_baselineUncorrected[trial_indices, :].shape[0])
                )

            # adding column names
            columns.append(f"bin_({bin_label})")
            columns.append(f"bin_err_({bin_label})")

        psth = np.concatenate((psth, psth_bin), axis=0)
        psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, psth_bin_baselineUncorrected), axis=0)

    timeAxis = timeAxis.reshape(1, -1)
    psth = np.concatenate((psth, timeAxis), axis=0)
    psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, timeAxis), axis=0)
    columns.append("timestamps")

    return psth, psth_baselineUncorrected, columns, event_timestamps


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
    trial : np.ndarray
        1-D trial array of length ``nTsPrev + nTsPost + 1``, NaN-padded where the
        signal does not exist.
    """

    if nTsPrev < thisIndex and z_score.shape[0] > (thisIndex + nTsPost):
        trial = z_score[thisIndex - nTsPrev - 1 : thisIndex + nTsPost]
    elif nTsPrev >= thisIndex and z_score.shape[0] > (thisIndex + nTsPost):
        mismatch = nTsPrev - thisIndex + 1
        trial = np.zeros(nTsPrev + nTsPost + 1)
        trial[:mismatch] = np.nan
        trial[mismatch:] = z_score[: thisIndex + nTsPost]
    elif nTsPrev >= thisIndex and z_score.shape[0] < (thisIndex + nTsPost):
        mismatch1 = nTsPrev - thisIndex + 1
        mismatch2 = (thisIndex + nTsPost) - z_score.shape[0]
        nan_prefix = np.full(mismatch1, np.nan)
        existing_signal = z_score
        nan_suffix = np.full(mismatch2, np.nan)
        trial = np.concatenate((nan_prefix, np.concatenate((existing_signal, nan_suffix))))
    else:
        mismatch = (thisIndex + nTsPost) - z_score.shape[0]
        nan_suffix = np.zeros(mismatch)
        nan_suffix[:] = np.nan
        existing_signal = z_score[thisIndex - nTsPrev - 1 : z_score.shape[0]]
        trial = np.concatenate((existing_signal, nan_suffix))

    return trial


def baselineCorrection(trial: np.ndarray, timeAxis: np.ndarray, baselineStart: float, baselineEnd: float) -> np.ndarray:
    """
    Subtract the mean baseline from a single PSTH trial.

    Parameters
    ----------
    trial : np.ndarray
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
        Trial array with the mean baseline subtracted; ``trial`` unchanged when
        both ``baselineStart`` and ``baselineEnd`` are zero.
    """
    baselineStrtPt = np.where(timeAxis >= baselineStart)[0]
    baselineEndPt = np.where(timeAxis >= baselineEnd)[0]

    if baselineStart == 0 and baselineEnd == 0:
        return trial

    baseline = np.nanmean(trial[baselineStrtPt[0] : baselineEndPt[0]])
    baselineSub = np.subtract(trial, baseline)

    return baselineSub
