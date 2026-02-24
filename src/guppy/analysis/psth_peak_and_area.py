import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


def compute_psth_peak_and_area(psth_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint):

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

    peak_and_area = OrderedDict()

    if peak_startPoint.shape[0] == 0 or peak_endPoint.shape[0] == 0:
        peak_and_area["peak"] = np.nan
        peak_and_area["area"] = np.nan

    for i in range(peak_startPoint.shape[0]):
        startPtForPeak = np.where(timestamps >= peak_startPoint[i])[0]
        endPtForPeak = np.where(timestamps >= peak_endPoint[i])[0]
        if len(startPtForPeak) >= 1 and len(endPtForPeak) >= 1:
            peakPoint_pos = startPtForPeak[0] + np.argmax(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peakPoint_neg = startPtForPeak[0] + np.argmin(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peak_and_area["peak_pos_" + str(i + 1)] = np.amax(psth_mean[peakPoint_pos], axis=0)
            peak_and_area["peak_neg_" + str(i + 1)] = np.amin(psth_mean[peakPoint_neg], axis=0)
            peak_and_area["area_" + str(i + 1)] = np.trapezoid(
                psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0
            )
        else:
            peak_and_area["peak_" + str(i + 1)] = np.nan
            peak_and_area["area_" + str(i + 1)] = np.nan

    return peak_and_area
