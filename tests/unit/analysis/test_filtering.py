import numpy as np
import pytest

from guppy.analysis.filtering import filterSignal, validate_chunk_lengths_for_filtering

# ── validate_chunk_lengths_for_filtering ──────────────────────────────────────


def test_validate_chunk_lengths_raises_on_short_chunk():
    # filter_window=100 -> padlen=300, so a chunk must have > 300 samples.
    tsNew = np.arange(250.0)  # 250 samples, all inside one chunk
    coords = np.array([[-0.5, 249.5]])
    with pytest.raises(ValueError) as exception_info:
        validate_chunk_lengths_for_filtering(tsNew, coords, 100)
    message = str(exception_info.value)
    assert "250 samples" in message
    assert "more than 300" in message


def test_validate_chunk_lengths_passes_when_all_chunks_long_enough():
    tsNew = np.arange(400.0)  # 400 samples > padlen 300
    coords = np.array([[-0.5, 399.5]])
    # Should not raise.
    validate_chunk_lengths_for_filtering(tsNew, coords, 100)


def test_validate_chunk_lengths_no_check_when_filtering_disabled():
    tsNew = np.arange(10.0)  # far shorter than any padlen
    coords = np.array([[-0.5, 9.5]])
    # filter_window=0 disables filtering, so no length requirement applies.
    validate_chunk_lengths_for_filtering(tsNew, coords, 0)


# ── filterSignal ──────────────────────────────────────────────────────────────


def test_filter_signal_window_zero_returns_original(uniform_signal):
    result = filterSignal(0, uniform_signal)
    np.testing.assert_array_equal(result, uniform_signal)


def test_filter_signal_window_greater_than_one_smooths(uniform_signal):
    result = filterSignal(11, uniform_signal)
    assert result.var() < uniform_signal.var()


def test_filter_signal_window_one_raises():
    signal = np.ones(100)
    with pytest.raises(ValueError) as exception_info:
        filterSignal(1, signal)
    message = str(exception_info.value)
    assert "filter_window=1" in message
    assert "Use 0 to disable" in message
