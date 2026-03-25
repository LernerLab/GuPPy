import math

import numpy as np
import pytest

from guppy.analysis.transients import createChunks, processChunks


def test_create_chunks_column_count_equals_window_points():
    sampling_rate = 100.0
    window = 60  # seconds
    z_score = np.random.default_rng(seed=0).standard_normal(7200)
    chunks, _ = createChunks(z_score, sampling_rate, window)
    expected_window_points = math.ceil(sampling_rate * window)
    assert chunks.shape[1] == expected_window_points


def test_create_chunks_total_elements_gte_original_length():
    z_score = np.ones(350)
    chunks, _ = createChunks(z_score, sampling_rate=100.0, window=2)
    assert chunks.size >= z_score.size


def test_create_chunks_padding_values_are_nan():
    # Use a length that is NOT a multiple of the window points so padding is added
    z_score = np.ones(350)
    sampling_rate = 100.0
    window = 2
    window_points = math.ceil(sampling_rate * window)
    remainder = z_score.shape[0] % window_points
    if remainder == 0:
        # Unlikely with these values, but guard anyway
        pytest.skip("No padding needed for this input")
    chunks, _ = createChunks(z_score, sampling_rate, window)
    flat = chunks.ravel()
    # Padding elements are at the end
    padding_count = chunks.size - z_score.size
    assert np.all(np.isnan(flat[-padding_count:]))


def test_create_chunks_no_padding_when_length_is_exact_multiple():
    sampling_rate = 100.0
    window = 2
    window_points = math.ceil(sampling_rate * window)
    z_score = np.ones(window_points * 5)  # exact multiple
    chunks, _ = createChunks(z_score, sampling_rate, window)
    assert not np.any(np.isnan(chunks))


def test_create_chunks_index_array_has_same_shape_as_chunks():
    z_score = np.arange(500, dtype=float)
    chunks, indices = createChunks(z_score, sampling_rate=100.0, window=1)
    assert chunks.shape == indices.shape


def test_process_chunks_flat_input_returns_zero_peaks():
    # Flat signal has zero MAD — no peaks above threshold
    values = np.ones(200)
    indices = np.arange(200)
    peaks, *_ = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=2.0)
    assert peaks.shape[0] == 0


def test_process_chunks_known_spikes_are_detected():
    rng = np.random.default_rng(seed=7)
    # Small baseline noise with two clear spikes
    values = rng.standard_normal(500) * 0.1
    values[100] = 50.0
    values[300] = 50.0
    indices = np.arange(500)
    peaks, *_ = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=2.0)
    # At least one of the spikes should be detected
    assert peaks.shape[0] >= 1


def test_process_chunks_returns_seven_element_tuple():
    values = np.random.default_rng(seed=8).standard_normal(200)
    indices = np.arange(200)
    result = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=2.0)
    assert len(result) == 7


def test_process_chunks_very_high_threshold_suppresses_peaks():
    rng = np.random.default_rng(seed=9)
    # Extreme transientsThresh ensures that no local maxima exceed the second threshold.
    # When transientsThresh is very large, secondThreshold is far above all values,
    # so greaterThanThreshIndex is empty and argrelextrema finds no peaks.
    values = rng.standard_normal(500)
    indices = np.arange(500)
    peaks, *_ = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=1000.0)
    assert peaks.shape[0] == 0
