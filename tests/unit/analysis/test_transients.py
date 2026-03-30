import numpy as np

from guppy.analysis.transients import (
    analyze_transients,
    calculate_freq_amp,
    createChunks,
    processChunks,
)


def test_create_chunks_column_count_equals_window_points():
    # sampling_rate=100, window=60 → window_points = ceil(100*60) = 6000
    z_score = np.random.default_rng(seed=0).standard_normal(7200)
    chunks, _ = createChunks(z_score, sampling_rate=100.0, window=60)
    assert chunks.shape[1] == 6000


def test_create_chunks_total_elements_equals_padded_length():
    # z_score length 350, window_points=200 → remainder=50 padding → padded length=400
    z_score = np.ones(350)
    chunks, _ = createChunks(z_score, sampling_rate=100.0, window=2)
    assert chunks.size == 400


def test_create_chunks_padding_values_are_nan():
    # Use a length that is NOT a multiple of the window points so padding is added
    z_score = np.ones(350)
    sampling_rate = 100.0
    window = 2
    window_points = int(sampling_rate * window)  # 200
    # 350 % 200 = 150, so padding is always added for these inputs
    chunks, _ = createChunks(z_score, sampling_rate, window)
    flat = chunks.ravel()
    # Padding elements are at the end
    padding_count = chunks.size - z_score.size
    assert np.all(np.isnan(flat[-padding_count:]))


def test_create_chunks_no_padding_when_length_is_exact_multiple():
    sampling_rate = 100.0
    window = 2
    window_points = int(sampling_rate * window)  # 200
    z_score = np.ones(window_points * 5)  # exact multiple: 1000
    chunks, _ = createChunks(z_score, sampling_rate, window)
    assert not np.any(np.isnan(chunks))


def test_create_chunks_index_array_contains_sequential_indices():
    # z_score length 500, window_points=100 → 5 chunks of 100; no padding
    z_score = np.arange(500, dtype=float)
    chunks, indices = createChunks(z_score, sampling_rate=100.0, window=1)
    assert indices.shape == (5, 100)
    np.testing.assert_array_equal(indices[0], np.arange(100))
    np.testing.assert_array_equal(indices[-1], np.arange(400, 500))


def test_process_chunks_flat_input_returns_zero_peaks():
    # Flat signal has zero MAD — no peaks above threshold
    values = np.ones(200)
    indices = np.arange(200)
    peaks, *_ = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=2.0)
    assert len(peaks) == 0


def test_process_chunks_known_spikes_are_detected():
    rng = np.random.default_rng(seed=7)
    # Small baseline noise with two clear spikes at indices 100 and 300
    values = rng.standard_normal(500) * 0.1
    values[100] = 50.0
    values[300] = 50.0
    indices = np.arange(500)
    peaks, *_ = processChunks(values, indices, highAmpFilt=3.0, transientsThresh=2.0)
    assert 100 in peaks
    assert 300 in peaks


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
    assert len(peaks) == 0


# ── calculate_freq_amp ────────────────────────────────────────────────────────


def test_calculate_freq_amp_known_single_peak_correct_frequency_and_amplitude():
    # One chunk of 5 samples; peak at local index 2 (global index 2)
    # filteredOutMedianY = 1.0; z_score[2] = 5.0 → amplitude = 4.0
    # timestamps span 0 to 240 seconds = 4 minutes → freq = 1 peak / 4 min = 0.25
    z_score = np.array([0.0, 1.0, 5.0, 1.0, 0.0])
    z_score_chunks_index = np.array([[0, 1, 2, 3, 4]])
    timestamps = np.array([0.0, 60.0, 120.0, 180.0, 240.0])

    arr = np.empty((1, 7), dtype=object)
    arr[0, 0] = np.array([2])  # peaks at local index 2
    arr[0, 1] = np.full(5, 0.5)  # mad
    arr[0, 2] = np.full(5, 0.3)  # filteredOutMad
    arr[0, 3] = np.full(5, 0.5)  # medianY
    arr[0, 4] = np.full(5, 1.0)  # filteredOutMedianY = 1.0
    arr[0, 5] = np.full(5, 2.0)  # firstThresholdY
    arr[0, 6] = np.full(5, 1.5)  # secondThresholdY

    freq, peaks_amp, peaks_ind = calculate_freq_amp(arr, z_score, z_score_chunks_index, timestamps)

    np.testing.assert_allclose(freq, 0.25, atol=1e-6)
    np.testing.assert_allclose(peaks_amp, np.array([4.0]), atol=1e-6)
    np.testing.assert_array_equal(peaks_ind, np.array([2]))


def test_calculate_freq_amp_two_chunks_two_peaks_correct_count():
    # Two chunks of 5, peak in each chunk at local index 2
    # Global peaks at index 2 and 7; timestamps span 0 to 120s = 2 min → freq = 2/2 = 1.0
    z_score = np.array([0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0])
    z_score_chunks_index = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    timestamps = np.array([0.0, 24.0, 48.0, 72.0, 96.0, 20.0, 44.0, 68.0, 92.0, 120.0])

    arr = np.empty((2, 7), dtype=object)
    for row in range(2):
        arr[row, 0] = np.array([2])  # peak at local index 2
        arr[row, 1] = np.full(5, 0.5)
        arr[row, 2] = np.full(5, 0.3)
        arr[row, 3] = np.full(5, 0.5)
        arr[row, 4] = np.full(5, 1.0)  # filteredOutMedianY = 1.0
        arr[row, 5] = np.full(5, 2.0)
        arr[row, 6] = np.full(5, 1.5)

    freq, peaks_amp, peaks_ind = calculate_freq_amp(arr, z_score, z_score_chunks_index, timestamps)

    assert peaks_amp.shape[0] == 2
    assert peaks_ind.shape[0] == 2
    np.testing.assert_allclose(freq, 1.0, atol=1e-6)


# ── analyze_transients ────────────────────────────────────────────────────────


def test_analyze_transients_output_shapes_are_correct():
    # 40 samples at 10 Hz over 4 seconds; window=2s → 2 chunks of 20
    rng = np.random.default_rng(seed=15)
    z_score = rng.standard_normal(40) * 0.1
    z_score[10] = 20.0  # clear spike
    timestamps = np.linspace(0, 4, 40)

    cleaned_z, cleaned_ts, peaks_ind, peaks_occurrences, arr = analyze_transients(
        ts=timestamps,
        window=2,
        numProcesses=1,
        highAmpFilt=3.0,
        transientsThresh=2.0,
        sampling_rate=10.0,
        z_score=z_score,
    )

    assert cleaned_z.shape == cleaned_ts.shape
    assert arr.shape == (1, 2)
    assert peaks_occurrences.ndim == 2
    assert peaks_occurrences.shape[1] == 2
