"""Unit tests for baseline-epoch control fitting (issue #200).

These tests exercise the estimate/apply split in the control fit and, crucially, the
interaction between baseline-epoch fitting and artifact-removal chunking: coefficients are
estimated once from the fit window intersected with the retained (non-artifact) data, then
applied across every chunk.

Inputs are built so the expected outputs are hand-computable and asserted as literals. All
signals use ``filter_window=0`` (no smoothing) so ``filterSignal`` is the identity and the
linear relationship is exact.
"""

import numpy as np
import pytest

from guppy.analysis.z_score import (
    apply_control_fit,
    compute_z_score,
    estimate_baseline_epoch_coefficients,
    estimate_control_fit_coefficients,
)


class TestEstimateAndApply:
    def test_estimate_recovers_exact_linear_coefficients(self):
        # signal = 2 * control + 1 exactly, so OLS returns (slope, intercept) = (2, 1).
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        signal = 2.0 * control + 1.0
        slope, intercept = estimate_control_fit_coefficients(control, signal, method="OLS")
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_apply_projects_control_onto_signal_scale(self):
        control = np.array([1.0, 2.0, 3.0])
        # 2 * [1, 2, 3] + 1 = [3, 5, 7]
        np.testing.assert_allclose(apply_control_fit(control, 2.0, 1.0), np.array([3.0, 5.0, 7.0]))


class TestBaselineEpochCoefficientEstimation:
    """The fit window is intersected with retained data before estimating coefficients."""

    def test_artifact_inside_window_is_excluded_from_the_fit(self):
        # Clean relationship signal = 2*control + 1, except a huge artifact spike at t=3.
        ts = np.arange(10.0)
        control = np.arange(1.0, 11.0)
        signal = 2.0 * control + 1.0
        signal[3] = 1000.0
        # Good chunks keep {0,1,2} and {4,...,9}; t=3 falls in the removed gap.
        coords = np.array([[-0.5, 2.5], [3.5, 9.5]])
        # Fit window spans the whole trace, but the artifact at t=3 is excluded by artifact removal.
        slope, intercept = estimate_baseline_epoch_coefficients(control, signal, ts, coords, 0, "OLS", 0, 9)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_empty_estimation_mask_raises(self):
        ts = np.arange(10.0)
        control = np.arange(1.0, 11.0)
        signal = 2.0 * control + 1.0
        # Retain only {4,...,9}; a fit window of [0, 3] intersects no retained data.
        coords = np.array([[3.5, 9.5]])
        with pytest.raises(ValueError, match="no data after artifact removal"):
            estimate_baseline_epoch_coefficients(control, signal, ts, coords, 0, "OLS", 0, 3)


class TestBaselineEpochComputeZScore:
    """End-to-end through compute_z_score, asserting fitted_control and dF/F literals.

    A step-change at t>=5 (like a drug injection) would corrupt a full-trace fit. Baseline-epoch
    mode estimates coefficients from the pre-step window and applies them everywhere, so the
    step survives in dF/F instead of being absorbed by the fit.
    """

    def _fixture(self):
        # Pre-step (t<5): signal = 2*control + 1. Post-step (t>=5): +100 sustained step.
        ts = np.arange(10.0)
        control = np.arange(1.0, 11.0)  # [1..10]
        signal = 2.0 * control + 1.0  # [3,5,7,9,11,13,15,17,19,21]
        signal[ts >= 5] += 100.0  # [.., 113,115,117,119,121]
        return ts, control, signal

    def _run(self, ts, control, signal, coords, window):
        return compute_z_score(
            control,
            signal,
            ts,
            coords,
            "replace with NaN",
            0,  # filter_window: no smoothing
            True,  # isosbestic_control
            "standard z-score",
            0,
            0,
            "OLS",
            "baseline epoch",
            window[0],
            window[1],
        )

    def test_step_survives_in_dff_single_chunk(self):
        ts, control, signal = self._fixture()
        coords = np.array([[-0.5, 9.5]])  # one chunk spanning the whole trace
        # Fit window [0, 4] -> pre-step points {0,1,2,3,4}, giving slope=2, intercept=1.
        _, dff, fitted_control, _ = self._run(ts, control, signal, coords, (0, 4))
        # fitted_control = 2*control + 1 across the whole trace.
        np.testing.assert_allclose(
            fitted_control, np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]), atol=1e-9
        )
        # Pre-step dF/F is exactly 0; post-step dF/F = 100 / fitted_control * 100.
        expected_dff = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0 / 13, 100.0 / 15, 100.0 / 17, 100.0 / 19, 100.0 / 21]) * 100.0
        )
        np.testing.assert_allclose(dff, expected_dff, atol=1e-9)

    def test_post_step_artifact_still_uses_frozen_baseline_coefficients(self):
        ts, control, signal = self._fixture()
        # Remove a post-step segment: keep {0..4} (pre) and {7,8,9} (post); {5,6} removed.
        coords = np.array([[-0.5, 4.5], [6.5, 9.5]])
        _, dff, fitted_control, _ = self._run(ts, control, signal, coords, (0, 4))
        # Frozen (2, 1) applies to the surviving post-step chunk; removed points stay NaN.
        np.testing.assert_allclose(
            fitted_control,
            np.array([3.0, 5.0, 7.0, 9.0, 11.0, np.nan, np.nan, 17.0, 19.0, 21.0]),
            atol=1e-9,
            equal_nan=True,
        )
        expected_dff = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 100.0 / 17 * 100, 100.0 / 19 * 100, 100.0 / 21 * 100]
        )
        np.testing.assert_allclose(dff, expected_dff, atol=1e-9, equal_nan=True)

    def test_fit_window_straddling_removed_region_uses_only_survivors(self):
        ts, control, signal = self._fixture()
        # Put an artifact spike at t=2 and remove it; keep {0,1} and {3,4,...}.
        signal[2] = 5000.0
        coords = np.array([[-0.5, 1.5], [2.5, 9.5]])
        # Fit window [0, 4] straddles the removed point t=2; only {0,1,3,4} feed the fit.
        _, dff, fitted_control, _ = self._run(ts, control, signal, coords, (0, 4))
        # Survivors {0,1,3,4} are all clean 2*control+1, so coefficients are still (2, 1).
        np.testing.assert_allclose(
            fitted_control,
            np.array([3.0, 5.0, np.nan, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]),
            atol=1e-9,
            equal_nan=True,
        )
        # t=2 removed (NaN); post-step points still show the true step.
        assert np.isnan(dff[2])
        np.testing.assert_allclose(
            dff[5:], np.array([100.0 / 13, 100.0 / 15, 100.0 / 17, 100.0 / 19, 100.0 / 21]) * 100.0
        )

    def test_full_trace_mode_absorbs_the_step(self):
        # Contrast: full-trace fit over the stepped signal does NOT recover the true (2, 1) fit,
        # so pre-step dF/F is not 0 — the step corrupts the whole-trace fit.
        ts, control, signal = self._fixture()
        coords = np.array([[-0.5, 9.5]])
        _, dff, _, _ = compute_z_score(
            control,
            signal,
            ts,
            coords,
            "replace with NaN",
            0,
            True,
            "standard z-score",
            0,
            0,
            "OLS",
            "full trace",
            0,
            0,
        )
        # A clean baseline-epoch fit gives pre-step dF/F == 0; full-trace does not.
        assert np.abs(dff[:5]).max() > 1.0
