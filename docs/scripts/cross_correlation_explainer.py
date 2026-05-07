# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Generate SVG figures for the cross-correlation explanation page.

The cross-correlation math (scipy.signal.correlate + per-trial peak
normalisation) mirrors src/guppy/analysis/cross_correlation.py exactly.
Only the input traces are synthetic and designed to make each point clean.

Run with:

    uv run docs/scripts/cross_correlation_explainer.py

Inline dependencies above let `uv run` resolve everything without needing
the surrounding project to be installed. Outputs are written directly to
docs/_static/images/cross_correlation_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "cross_correlation_explainer"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
        "axes.titlesize": 11,
    }
)

COLOR_A = "#1f77b4"
COLOR_B = "#d62728"
COLOR_CORR = "#333333"


def make_bump(t, center, width, amplitude=1.0):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))


def crosscorr(a, b, sample_rate):
    """Mirror of GuPPy's per-trial cross-correlation."""
    corr = signal.correlate(a, b)
    corr_norm = corr / np.max(np.abs(corr))
    lag = signal.correlation_lags(len(a), len(b)) / sample_rate
    return lag, corr_norm


def figure_1_what_is_cross_correlation():
    rng = np.random.default_rng(0)
    sample_rate = 100
    t = np.arange(-2, 2, 1 / sample_rate)
    a = make_bump(t, center=0.0, width=0.25) + 0.04 * rng.standard_normal(len(t))
    b = make_bump(t, center=0.4, width=0.25) + 0.04 * rng.standard_normal(len(t))
    lag, corr = crosscorr(a, b, sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.6))

    ax = axes[0]
    ax.plot(t, a, color=COLOR_A, label="region A", linewidth=1.6)
    ax.plot(t, b, color=COLOR_B, label="region B", linewidth=1.6)
    ax.set_xlabel("time within trial (s)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("two event-aligned traces", loc="left")

    ax = axes[1]
    ax.plot(lag, corr, color=COLOR_CORR, linewidth=1.6)
    peak_idx = int(np.argmax(corr))
    ax.axvline(lag[peak_idx], color=COLOR_CORR, linestyle=":", alpha=0.4)
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.annotate(
        f"peak at lag = {lag[peak_idx]:.2f} s\n(B trails A by ~0.4 s)",
        xy=(lag[peak_idx], corr[peak_idx]),
        xytext=(lag[peak_idx] + 0.7, 0.55),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLOR_CORR, "lw": 0.8},
    )
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("normalised correlation")
    ax.set_xlim(-2, 2)
    ax.set_title("cross-correlogram", loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_what_is_cross_correlation.svg")
    plt.close(fig)


def figure_2_lag_mapping():
    rng = np.random.default_rng(1)
    sample_rate = 100
    t = np.arange(-1.5, 1.5, 1 / sample_rate)

    scenarios = [
        ("B aligned with A", 0.0),
        ("B trails A by 0.3 s", 0.3),
        ("B leads A by 0.3 s", -0.3),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 5.4), gridspec_kw={"width_ratios": [1, 1]})

    for row, (label, shift) in enumerate(scenarios):
        a = make_bump(t, center=0.0, width=0.2) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=shift, width=0.2) + 0.04 * rng.standard_normal(len(t))
        lag, corr = crosscorr(a, b, sample_rate)

        ax_l, ax_r = axes[row]
        ax_l.plot(t, a, color=COLOR_A, linewidth=1.4)
        ax_l.plot(t, b, color=COLOR_B, linewidth=1.4)
        ax_l.set_xlim(-1.5, 1.5)
        ax_l.set_ylim(-0.2, 1.2)
        ax_l.set_ylabel(label, fontsize=9)
        if row == 0:
            ax_l.set_title("traces", loc="left")
        if row == 2:
            ax_l.set_xlabel("time within trial (s)")

        ax_r.plot(lag, corr, color=COLOR_CORR, linewidth=1.4)
        peak_lag = lag[int(np.argmax(corr))]
        ax_r.axvline(peak_lag, color=COLOR_CORR, linestyle=":", alpha=0.4)
        ax_r.axvline(0, color="#888888", linewidth=0.5, alpha=0.5)
        ax_r.set_xlim(-1.5, 1.5)
        ax_r.set_ylim(-0.4, 1.1)
        ax_r.text(
            0.97,
            0.92,
            f"peak at {peak_lag:+.2f} s",
            transform=ax_r.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        if row == 0:
            ax_r.set_title("cross-correlogram", loc="left")
        if row == 2:
            ax_r.set_xlabel("lag (s)")

    handles = [
        plt.Line2D([0], [0], color=COLOR_A, label="region A"),
        plt.Line2D([0], [0], color=COLOR_B, label="region B"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.27, 1.0),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig2_lag_mapping.svg")
    plt.close(fig)


def figure_3_normalisation_discards_amplitude():
    rng = np.random.default_rng(2)
    sample_rate = 100
    t = np.arange(-1.5, 1.5, 1 / sample_rate)

    scenarios = [
        ("both large (A=1.0, B=1.0)", 1.0, 1.0),
        ("both tiny (A=0.1, B=0.1)", 0.1, 0.1),
        ("asymmetric (A=1.0, B=0.1)", 1.0, 0.1),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 5.8))

    for row, (label, amp_a, amp_b) in enumerate(scenarios):
        noise_a = 0.01 * rng.standard_normal(len(t))
        noise_b = 0.01 * rng.standard_normal(len(t))
        a = amp_a * make_bump(t, center=0.0, width=0.2) + noise_a
        b = amp_b * make_bump(t, center=0.25, width=0.2) + noise_b
        lag, corr = crosscorr(a, b, sample_rate)

        ax_corr, ax_traces = axes[row]

        ax_corr.plot(lag, corr, color=COLOR_CORR, linewidth=1.4)
        ax_corr.axvline(0, color="#888888", linewidth=0.5, alpha=0.5)
        ax_corr.set_xlim(-1.5, 1.5)
        ax_corr.set_ylim(-0.4, 1.1)
        ax_corr.set_ylabel(label, fontsize=9)
        if row == 0:
            ax_corr.set_title("normalised correlogram (looks identical)", loc="left")
        if row == len(scenarios) - 1:
            ax_corr.set_xlabel("lag (s)")

        ax_traces.plot(t, a, color=COLOR_A, linewidth=1.4, label="region A")
        ax_traces.plot(t, b, color=COLOR_B, linewidth=1.4, label="region B")
        ax_traces.set_xlim(-1.5, 1.5)
        ax_traces.set_ylim(-0.2, 1.15)
        if row == 0:
            ax_traces.set_title("underlying traces (real amplitude)", loc="left")
            ax_traces.legend(loc="upper right", frameon=False, fontsize=8)
        if row == len(scenarios) - 1:
            ax_traces.set_xlabel("time within trial (s)")

    fig.tight_layout()
    fig.savefig(OUT / "fig3_normalisation_discards_amplitude.svg")
    plt.close(fig)


def _trial_panel_pair(ax_l, ax_r, t, trial_fn, n_trials, sample_rate, lag_xlim, ylim_l, label, with_titles, with_xlabels, ylim_r=(-1.1, 1.1)):
    """Plot n_trials overlays on ax_l and per-trial correlograms + trial average on ax_r."""
    per_trial_corrs = []
    lag = None
    for _ in range(n_trials):
        a, b = trial_fn()
        ax_l.plot(t, a, color=COLOR_A, linewidth=0.9, alpha=0.4)
        ax_l.plot(t, b, color=COLOR_B, linewidth=0.9, alpha=0.4)
        lag, corr = crosscorr(a, b, sample_rate)
        per_trial_corrs.append(corr)
        ax_r.plot(lag, corr, color="#888888", linewidth=0.7, alpha=0.4)
    avg_corr = np.mean(np.array(per_trial_corrs), axis=0)
    ax_r.plot(lag, avg_corr, color=COLOR_CORR, linewidth=1.8)

    ax_l.set_xlim(-2, 2)
    ax_l.set_ylim(*ylim_l)
    ax_l.set_ylabel(label, fontsize=9)
    ax_r.axvline(0, color="#888888", linewidth=0.5, alpha=0.5)
    ax_r.set_xlim(*lag_xlim)
    ax_r.set_ylim(*ylim_r)

    if with_titles:
        ax_l.set_title(f"{n_trials} trials overlaid", loc="left")
        ax_r.set_title("per-trial correlograms (pale) + trial average (bold)", loc="left")
    if with_xlabels:
        ax_l.set_xlabel("time within trial (s)")
        ax_r.set_xlabel("lag (s)")


def _add_trial_legend(fig):
    handles = [
        plt.Line2D([0], [0], color=COLOR_A, label="region A"),
        plt.Line2D([0], [0], color=COLOR_B, label="region B"),
        plt.Line2D([0], [0], color="#888888", alpha=0.6, label="single-trial correlogram"),
        plt.Line2D([0], [0], color=COLOR_CORR, label="trial average"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=9,
    )


def figure_5_timing_consistency():
    """Narrow vs broad trial-averaged peak from consistent vs jittered timing."""
    rng = np.random.default_rng(5)
    sample_rate = 100
    t = np.arange(-2, 2, 1 / sample_rate)
    lag_xlim = (-1.5, 1.5)
    n_trials = 8

    def consistent_timing_trial():
        jitter = rng.normal(0, 0.02)
        a = make_bump(t, center=0.0, width=0.15) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=0.2 + jitter, width=0.15) + 0.04 * rng.standard_normal(len(t))
        return a, b

    def jittered_timing_trial():
        jitter = rng.uniform(-0.45, 0.45)
        a = make_bump(t, center=0.0, width=0.15) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=jitter, width=0.15) + 0.04 * rng.standard_normal(len(t))
        return a, b

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.4))
    _trial_panel_pair(axes[0, 0], axes[0, 1], t, consistent_timing_trial, n_trials, sample_rate, lag_xlim, (-0.2, 1.15),
                      "consistent timing\n(narrow average peak)", with_titles=True, with_xlabels=False, ylim_r=(-0.2, 1.15))
    _trial_panel_pair(axes[1, 0], axes[1, 1], t, jittered_timing_trial, n_trials, sample_rate, lag_xlim, (-0.2, 1.15),
                      "jittered timing\n(broad average peak)", with_titles=False, with_xlabels=True, ylim_r=(-0.2, 1.15))
    _add_trial_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig5_timing_consistency.svg")
    plt.close(fig)


def figure_7_no_systematic_relationship():
    """Each region has a visible bump per trial, but bump positions are independent across regions."""
    rng = np.random.default_rng(7)
    sample_rate = 100
    t = np.arange(-2, 2, 1 / sample_rate)
    lag_xlim = (-1.5, 1.5)
    n_trials = 8

    def independent_bumps_trial():
        a_pos = rng.uniform(-1.0, 1.0)
        b_pos = rng.uniform(-1.0, 1.0)
        a = make_bump(t, center=a_pos, width=0.2) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=b_pos, width=0.2) + 0.04 * rng.standard_normal(len(t))
        return a, b

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.6))
    _trial_panel_pair(axes[0], axes[1], t, independent_bumps_trial, n_trials, sample_rate, lag_xlim, (-0.2, 1.15),
                      "no systematic relationship\n(no isolated peak in average)", with_titles=True, with_xlabels=True, ylim_r=(-0.2, 1.15))
    _add_trial_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig7_no_systematic_relationship.svg")
    plt.close(fig)


def figure_8_across_trials():
    """Two scenarios whose trial averages look similar but whose per-trial peak-lag distributions differ."""
    rng = np.random.default_rng(8)
    sample_rate = 100
    t = np.arange(-2, 2, 1 / sample_rate)
    lag_xlim = (-1.5, 1.5)
    n_trials = 80

    def jittered_timing_trial():
        # Peak lag uniformly distributed across a moderate range.
        jitter = rng.uniform(-0.35, 0.35)
        a = make_bump(t, center=0.0, width=0.2) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=jitter, width=0.2) + 0.04 * rng.standard_normal(len(t))
        return a, b

    def bimodal_timing_trial():
        # Half the trials peak at +0.2, the other half at -0.2; nothing in between.
        center = rng.choice([-0.2, 0.2])
        a = make_bump(t, center=0.0, width=0.2) + 0.04 * rng.standard_normal(len(t))
        b = make_bump(t, center=center, width=0.2) + 0.04 * rng.standard_normal(len(t))
        return a, b

    scenarios = [
        ("uniformly jittered timing", jittered_timing_trial),
        ("bimodal timing\n(half +0.2 s, half -0.2 s)", bimodal_timing_trial),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.6))

    for row, (label, trial_fn) in enumerate(scenarios):
        per_trial_peak_lags = []
        per_trial_corrs = []
        lag = None
        for _ in range(n_trials):
            a, b = trial_fn()
            lag, corr = crosscorr(a, b, sample_rate)
            per_trial_corrs.append(corr)
            per_trial_peak_lags.append(lag[int(np.argmax(corr))])
        avg_corr = np.mean(np.array(per_trial_corrs), axis=0)

        ax_avg, ax_hist = axes[row]

        ax_avg.plot(lag, avg_corr, color=COLOR_CORR, linewidth=1.6)
        ax_avg.axvline(0, color="#888888", linewidth=0.5, alpha=0.5)
        ax_avg.set_xlim(*lag_xlim)
        ax_avg.set_ylim(-0.2, 1.15)
        ax_avg.set_ylabel(label, fontsize=9)
        if row == 0:
            ax_avg.set_title("trial-averaged correlogram (what the GUI shows)", loc="left")
        if row == len(scenarios) - 1:
            ax_avg.set_xlabel("lag (s)")

        ax_hist.hist(per_trial_peak_lags, bins=24, range=lag_xlim, color="#888888", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax_hist.axvline(0, color="#888888", linewidth=0.5, alpha=0.5)
        ax_hist.set_xlim(*lag_xlim)
        if row == 0:
            ax_hist.set_title(f"per-trial peak lags ({n_trials} trials, computed from HDF5)", loc="left")
        if row == len(scenarios) - 1:
            ax_hist.set_xlabel("lag (s)")

    fig.tight_layout()
    fig.savefig(OUT / "fig8_across_trials.svg")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_what_is_cross_correlation()
    figure_2_lag_mapping()
    figure_3_normalisation_discards_amplitude()
    figure_5_timing_consistency()
    figure_7_no_systematic_relationship()
    figure_8_across_trials()
    print("Wrote SVGs to", OUT)
