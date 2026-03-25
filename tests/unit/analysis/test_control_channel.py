import numpy as np

from guppy.analysis.control_channel import curveFitFn, helper_create_control_channel


def test_curve_fit_fn_known_params_at_x_zero():
    # a + b * exp(-(1/c) * 0) = a + b
    result = curveFitFn(np.array([0.0]), a=0.0, b=1.0, c=1.0)
    np.testing.assert_allclose(result, np.array([1.0]))


def test_curve_fit_fn_large_x_approaches_a():
    # At large x, exp term vanishes and result → a
    result = curveFitFn(np.array([1000.0]), a=5.0, b=10.0, c=1.0)
    np.testing.assert_allclose(result, np.array([5.0]), atol=1e-3)


def test_curve_fit_fn_vectorized_produces_same_shape():
    x = np.linspace(0, 10, 100)
    result = curveFitFn(x, a=1.0, b=2.0, c=3.0)
    assert result.shape == x.shape


def test_helper_create_control_channel_output_shape_matches_signal():
    rng = np.random.default_rng(seed=20)
    timestamps = np.linspace(0, 10, 500)
    # Well-conditioned exponential decay that curve_fit can reliably converge on
    signal = 3.0 + 10.0 * np.exp(-timestamps / 5.0) + 0.01 * rng.standard_normal(500)
    result = helper_create_control_channel(signal, timestamps, window=101)
    assert result.shape == signal.shape


def test_helper_create_control_channel_output_is_smooth():
    rng = np.random.default_rng(seed=21)
    timestamps = np.linspace(0, 10, 500)
    signal = 2.0 + 8.0 * np.exp(-timestamps / 4.0) + 0.01 * rng.standard_normal(500)
    result = helper_create_control_channel(signal, timestamps, window=101)
    # The fitted curve should have lower variance than the noisy signal
    assert result.var() < signal.var()


def test_helper_create_control_channel_large_window_adjusted_down():
    rng = np.random.default_rng(seed=22)
    timestamps = np.linspace(0, 10, 200)
    signal = 1.0 + 5.0 * np.exp(-timestamps / 3.0) + 0.01 * rng.standard_normal(200)
    result = helper_create_control_channel(signal, timestamps, window=9999)
    assert result.shape == signal.shape
