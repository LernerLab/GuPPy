import logging
import math
import multiprocessing as mp
from itertools import repeat

import numpy as np
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


def analyze_transients(
    ts: np.ndarray,
    window: float,
    numProcesses: int,
    highAmpFilt: float,
    transientsThresh: float,
    sampling_rate: float,
    z_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect transient events in a z-score signal using a two-threshold MAD approach.

    Parameters
    ----------
    ts : np.ndarray
        Timestamp array corresponding to ``z_score``.
    window : float
        Chunk duration in seconds for parallel processing.
    numProcesses : int
        Number of worker processes for the multiprocessing pool.
    highAmpFilt : float
        MAD multiplier used to identify and exclude high-amplitude samples before
        computing the second threshold.
    transientsThresh : float
        MAD multiplier applied to the filtered distribution to set the detection threshold.
    sampling_rate : float
        Sampling rate in Hz.
    z_score : np.ndarray
        1-D z-scored signal array.

    Returns
    -------
    z_score : np.ndarray
        NaN-free z-score array used for analysis.
    ts : np.ndarray
        NaN-free timestamp array aligned with the returned ``z_score``.
    peaksInd : np.ndarray
        Integer indices of detected transient peaks in ``z_score``.
    peaks_occurrences : np.ndarray
        Shape ``(n_peaks, 2)`` array of ``[timestamp, amplitude]`` for each peak.
    arr : np.ndarray
        Shape ``(1, 2)`` array of ``[frequency (events/min), mean amplitude]``.
    """
    not_nan_indices = ~np.isnan(z_score)
    z_score = z_score[not_nan_indices]
    z_score_chunks, z_score_chunks_index = createChunks(z_score, sampling_rate, window)

    with mp.Pool(numProcesses) as p:
        result = p.starmap(
            processChunks, zip(z_score_chunks, z_score_chunks_index, repeat(highAmpFilt), repeat(transientsThresh))
        )

    result = np.asarray(result, dtype=object)
    ts = ts[not_nan_indices]
    freq, peaksAmp, peaksInd = calculate_freq_amp(result, z_score, z_score_chunks_index, sampling_rate)
    peaks_occurrences = np.array([ts[peaksInd], peaksAmp]).T
    arr = np.array([[freq, np.mean(peaksAmp)]])
    return z_score, ts, peaksInd, peaks_occurrences, arr


def processChunks(
    arrValues: np.ndarray, arrIndexes: np.ndarray, highAmpFilt: float, transientsThresh: float
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the two-threshold MAD transient-detection algorithm to one chunk.

    Parameters
    ----------
    arrValues : np.ndarray
        Z-score values for this chunk (may contain NaN padding).
    arrIndexes : np.ndarray
        Global sample indices corresponding to ``arrValues``.
    highAmpFilt : float
        MAD multiplier for the first (high-amplitude exclusion) threshold.
    transientsThresh : float
        MAD multiplier for the second (detection) threshold.

    Returns
    -------
    tuple
        ``(peaks, mad, filteredOutMad, medianY, filteredOutMedianY,
        firstThresholdY, secondThresholdY)`` — detected peak indices and
        diagnostic arrays aligned with ``arrValues``.
    """

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


def createChunks(z_score: np.ndarray, sampling_rate: float, window: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad and reshape the z-score array into equal-length chunks for parallel processing.

    Parameters
    ----------
    z_score : np.ndarray
        1-D z-score array (NaN-free).
    sampling_rate : float
        Sampling rate in Hz.
    window : float
        Desired chunk duration in seconds.

    Returns
    -------
    z_score_chunks : np.ndarray
        Shape ``(n_chunks, chunk_length)`` array of z-score chunks.
    z_score_chunks_index : np.ndarray
        Shape ``(n_chunks, chunk_length)`` global sample indices aligned with ``z_score_chunks``.
    """

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
        message = (
            f"Cannot split z-score data of length {padded_z_score.shape[0]} into chunks of "
            f"{windowPoints} samples (sampling_rate={sampling_rate}, window={window}s): "
            f"the ratio {reshape} is non-integer. The padding logic may have a bug."
        )
        logger.error(message)
        raise ValueError(message)
    logger.info("Chunks are created for multiprocessing.")
    return z_score_chunks, z_score_chunks_index


def calculate_freq_amp(
    arr: np.ndarray,
    z_score: np.ndarray,
    z_score_chunks_index: np.ndarray,
    sampling_rate: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Aggregate per-chunk transient results into global frequency and amplitude statistics.

    Parameters
    ----------
    arr : np.ndarray
        Object array returned by ``processChunks``; each row is a chunk result tuple.
    z_score : np.ndarray
        Full 1-D z-score array.
    z_score_chunks_index : np.ndarray
        Shape ``(n_chunks, chunk_length)`` global sample indices from :func:`createChunks`.
    sampling_rate : float
        Sampling rate in Hz; used to derive the analyzed duration from the sample count.

    Returns
    -------
    freq : float
        Transient frequency in events per minute.
    peaksAmp : np.ndarray
        Amplitude (z-score minus local median) for each detected peak.
    peaksInd : np.ndarray
        Global integer indices of detected peaks in ``z_score``.
    """
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
    # Duration of the analyzed (kept) signal in minutes. Derived from the sample count so
    # that artifact-excised gaps are excluded from the denominator regardless of removal method.
    duration_minutes = (z_score.shape[0] / sampling_rate) / 60
    freq = peaksAmp.shape[0] / duration_minutes

    return freq, peaksAmp, peaksInd
