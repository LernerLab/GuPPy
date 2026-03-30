import logging
import math
import multiprocessing as mp
from itertools import repeat

import numpy as np
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


def analyze_transients(ts, window, numProcesses, highAmpFilt, transientsThresh, sampling_rate, z_score):
    not_nan_indices = ~np.isnan(z_score)
    z_score = z_score[not_nan_indices]
    z_score_chunks, z_score_chunks_index = createChunks(z_score, sampling_rate, window)

    with mp.Pool(numProcesses) as p:
        result = p.starmap(
            processChunks, zip(z_score_chunks, z_score_chunks_index, repeat(highAmpFilt), repeat(transientsThresh))
        )

    result = np.asarray(result, dtype=object)
    ts = ts[not_nan_indices]
    freq, peaksAmp, peaksInd = calculate_freq_amp(result, z_score, z_score_chunks_index, ts)
    peaks_occurrences = np.array([ts[peaksInd], peaksAmp]).T
    arr = np.array([[freq, np.mean(peaksAmp)]])
    return z_score, ts, peaksInd, peaks_occurrences, arr


def processChunks(arrValues, arrIndexes, highAmpFilt, transientsThresh):

    arrValues = arrValues[~np.isnan(arrValues)]
    median = np.median(arrValues)

    mad = np.median(np.abs(arrValues - median))

    firstThreshold = median + (highAmpFilt * mad)

    greaterThanMad = np.where(arrValues > firstThreshold)[0]

    arr = np.arange(arrValues.shape[0])
    lowerThanMad = np.isin(arr, greaterThanMad, invert=True)
    filteredOut = arrValues[np.where(lowerThanMad == True)[0]]

    filteredOutMedian = np.median(filteredOut)
    filteredOutMad = np.median(np.abs(filteredOut - np.median(filteredOut)))
    secondThreshold = filteredOutMedian + (transientsThresh * filteredOutMad)

    greaterThanThreshIndex = np.where(arrValues > secondThreshold)[0]
    greaterThanThreshValues = arrValues[greaterThanThreshIndex]
    temp = np.zeros(arrValues.shape[0])
    temp[greaterThanThreshIndex] = greaterThanThreshValues
    peaks = argrelextrema(temp, np.greater)[0]

    firstThresholdY = np.full(arrValues.shape[0], firstThreshold)
    secondThresholdY = np.full(arrValues.shape[0], secondThreshold)

    newPeaks = np.full(arrValues.shape[0], np.nan)
    newPeaks[peaks] = peaks + arrIndexes[0]

    # madY = np.full(arrValues.shape[0], mad)
    medianY = np.full(arrValues.shape[0], median)
    filteredOutMedianY = np.full(arrValues.shape[0], filteredOutMedian)

    return peaks, mad, filteredOutMad, medianY, filteredOutMedianY, firstThresholdY, secondThresholdY


def createChunks(z_score, sampling_rate, window):

    logger.debug("Creating chunks for multiprocessing...")
    windowPoints = math.ceil(sampling_rate * window)
    remainderPoints = math.ceil((sampling_rate * window) - (z_score.shape[0] % windowPoints))

    if remainderPoints == windowPoints:
        padded_z_score = z_score
        z_score_index = np.arange(padded_z_score.shape[0])
    else:
        padding = np.full(remainderPoints, np.nan)
        padded_z_score = np.concatenate((z_score, padding))
        z_score_index = np.arange(padded_z_score.shape[0])

    reshape = padded_z_score.shape[0] / windowPoints

    if reshape.is_integer() == True:
        z_score_chunks = padded_z_score.reshape(int(reshape), -1)
        z_score_chunks_index = z_score_index.reshape(int(reshape), -1)
    else:
        logger.error("Reshaping values should be integer.")
        raise Exception("Reshaping values should be integer.")
    logger.info("Chunks are created for multiprocessing.")
    return z_score_chunks, z_score_chunks_index


def calculate_freq_amp(arr, z_score, z_score_chunks_index, timestamps):
    peaks = arr[:, 0]
    filteredOutMedian = arr[:, 4]
    count = 0
    peaksAmp = np.array([])
    peaksInd = np.array([])
    for i in range(z_score_chunks_index.shape[0]):
        count += peaks[i].shape[0]
        peaksIndexes = peaks[i] + z_score_chunks_index[i][0]
        peaksInd = np.concatenate((peaksInd, peaksIndexes))
        amps = z_score[peaksIndexes] - filteredOutMedian[i][0]
        peaksAmp = np.concatenate((peaksAmp, amps))

    peaksInd = peaksInd.ravel()
    peaksInd = peaksInd.astype(int)
    # logger.info(timestamps)
    freq = peaksAmp.shape[0] / ((timestamps[-1] - timestamps[0]) / 60)

    return freq, peaksAmp, peaksInd
