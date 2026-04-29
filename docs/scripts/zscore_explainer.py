# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Generate SVG figures for the z-score normalisation explanation page.

The dF/F formula and z-score formulas mirror src/guppy/analysis/z_score.py:
    deltaFF:                 (signal - control_fit) / control_fit * 100
    standard z-score:        (dff - mean) / std
    baseline z-score:        (dff - mean[base]) / std[base]
    modified z-score:        0.6745 * (dff - median) / MAD

Run with:

    uv run docs/scripts/zscore_explainer.py

Inline dependencies above let `uv run` resolve everything without needing
the surrounding project to be installed. Outputs are written directly to
docs/_static/images/zscore_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "zscore_explainer"
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

COLOR_DFF = "#2ca02c"
COLOR_ZSTD = "#d62728"
COLOR_ZBASE = "#9467bd"
COLOR_ZMOD = "#ff7f0e"


def deltaFF(signal, control_fit):
    return (signal - control_fit) / control_fit * 100.0


def zscore_standard(dff):
    return (dff - np.nanmean(dff)) / np.nanstd(dff)


def zscore_baseline(dff, t, baseline_start=0.0, baseline_end=5.0):
    mask = (t >= baseline_start) & (t < baseline_end)
    base_mean = np.nanmean(dff[mask])
    base_std = np.nanstd(dff[mask])
    return (dff - base_mean) / base_std


def zscore_modified(dff):
    median = np.median(dff)
    mad = np.median(np.abs(dff - median))
    return 0.6745 * (dff - median) / mad


def figure_1_zscore_cross_session():
    """Two recordings with their own event structures and different absolute scales
    become comparable in z-score units. 2x2 hybrid: top row has per-session dF/F panels
    with a shared y-axis; bottom row spans both columns and overlays the z-scored versions
    on a single shared axis."""
    rng = np.random.default_rng(5)
    sample_rate = 100
    duration = 60.0
    t = np.arange(0, duration, 1 / sample_rate)

    transient_width = 0.5

    def make_session(centers, amps, noise_scale):
        drift = -8.0 * (t / t[-1])
        transients = sum(
            amp * np.exp(-((t - c) ** 2) / (2 * transient_width**2))
            for c, amp in zip(centers, amps)
        )
        noise = noise_scale * rng.standard_normal(len(t))
        baseline = 100.0
        signal = baseline + drift + transients + noise
        control_fit = baseline + drift
        return signal, control_fit

    # Session A: large transients in low noise.
    sig_a, ctrl_a = make_session(
        centers=[8.0, 20.0, 32.0, 48.0],
        amps=[10.0, 14.0, 8.0, 12.0],
        noise_scale=1.0,
    )
    # Session B: smaller transients in moderate noise; different event times.
    sig_b, ctrl_b = make_session(
        centers=[14.0, 26.0, 38.0, 54.0],
        amps=[4.0, 6.0, 3.5, 5.0],
        noise_scale=0.5,
    )

    dff_a = deltaFF(sig_a, ctrl_a)
    dff_b = deltaFF(sig_b, ctrl_b)
    z_a = zscore_standard(dff_a)
    z_b = zscore_standard(dff_b)

    color_a = "#1f77b4"
    color_b = "#ff7f0e"

    fig = plt.figure(figsize=(8.5, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.6, wspace=0.1)
    ax_dff_a = fig.add_subplot(gs[0, 0])
    ax_dff_b = fig.add_subplot(gs[0, 1], sharey=ax_dff_a)
    ax_z = fig.add_subplot(gs[1, :])

    ax_dff_a.plot(t, dff_a, color=color_a, linewidth=1.0)
    ax_dff_a.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff_a.set_ylabel("dF/F (%)", fontsize=9)
    ax_dff_a.set_xlabel("time (s)", fontsize=9)
    ax_dff_a.set_title("session A (dF/F): large transients, low noise", loc="left", fontsize=10)

    ax_dff_b.plot(t, dff_b, color=color_b, linewidth=1.0)
    ax_dff_b.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff_b.set_xlabel("time (s)", fontsize=9)
    ax_dff_b.set_title("session B (dF/F): small transients, moderate noise", loc="left", fontsize=10)
    plt.setp(ax_dff_b.get_yticklabels(), visible=False)

    ax_z.plot(t, z_a, color=color_a, linewidth=1.0, label="session A", alpha=0.9)
    ax_z.plot(t, z_b, color=color_b, linewidth=1.0, label="session B", alpha=0.9)
    ax_z.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_z.set_ylabel("standard z-score", fontsize=9)
    ax_z.set_xlabel("time (s)", fontsize=9)
    ax_z.legend(loc="upper right", frameon=False, fontsize=9)
    ax_z.set_ylim(-1.0, 7.0)
    ax_z.set_title("both sessions after standard z-scoring", loc="left", fontsize=10)

    fig.savefig(OUT / "fig1_zscore_cross_session.svg")
    plt.close(fig)


def figure_2_baseline_zscore_contrast():
    """Standard vs baseline z-score on the same dF/F. Short time window so the baseline
    region (0-5s) is a substantial fraction of the visible x-axis."""
    rng = np.random.default_rng(0)
    sample_rate = 100
    duration = 20.0
    t = np.arange(0, duration, 1 / sample_rate)

    baseline_level = 100.0
    drift = -3.0 * (t / t[-1])
    transient_centers = [9.0, 13.0, 17.0]
    transient_amps = [5.0, 7.5, 4.0]
    transient_width = 0.5
    transients = sum(
        amp * np.exp(-((t - c) ** 2) / (2 * transient_width**2))
        for c, amp in zip(transient_centers, transient_amps)
    )
    noise = 0.6 * rng.standard_normal(len(t))
    signal = baseline_level + drift + transients + noise
    control_fit = baseline_level + drift

    dff = deltaFF(signal, control_fit)
    z_std = zscore_standard(dff)
    z_base = zscore_baseline(dff, t, baseline_start=0.0, baseline_end=5.0)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True)
    ax_dff, ax_std, ax_base = axes

    ax_dff.axvspan(0, 5, color="#ffe699", alpha=0.55, linewidth=0)
    ax_dff.plot(t, dff, color=COLOR_DFF, linewidth=1.0)
    ax_dff.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff.set_ylabel("dF/F (%)", fontsize=9)
    ax_dff.set_title("dF/F: shaded baseline window contains only noise; transients fall outside it", loc="left")

    ax_std.plot(t, z_std, color=COLOR_ZSTD, linewidth=1.0)
    ax_std.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_std.set_ylabel("standard\nz-score", fontsize=9)
    ax_std.set_title("standard z-score: denominator is full-recording std (inflated by transients)", loc="left")

    ax_base.axvspan(0, 5, color="#ffe699", alpha=0.55, linewidth=0)
    ax_base.plot(t, z_base, color=COLOR_ZBASE, linewidth=1.0)
    ax_base.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_base.set_ylabel("baseline\nz-score", fontsize=9)
    ax_base.set_title("baseline z-score: denominator is baseline-window std only (much smaller)", loc="left")
    ax_base.set_xlabel("time (s)")

    ymin = min(z_std.min(), z_base.min()) - 0.5
    ymax = max(z_std.max(), z_base.max()) + 0.5
    for ax in (ax_std, ax_base):
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_baseline_zscore_contrast.svg")
    plt.close(fig)


def figure_3_modified_zscore_robustness():
    """Three-row stacked layout for peak-detection framing. Top: dF/F with three small
    real events plus two outliers of different sizes (40% and 80%). Middle: standard
    z-score with z = 3 peak threshold. Bottom: modified z-score with z = 3 peak threshold.
    The middle and bottom panels use independent y-scales so each can use its space well."""
    rng = np.random.default_rng(1)
    sample_rate = 30
    duration = 15.0
    t = np.arange(0, duration, 1 / sample_rate)

    small_centers = [2.0, 8.0, 13.0]
    small_amps = [5.0, 5.5, 4.5]
    huge_centers = [5.0, 11.0]
    huge_amps = [40.0, 80.0]
    transient_width = 0.4

    baseline = 100.0
    drift = -2.0 * (t / t[-1])

    small_t = sum(
        amp * np.exp(-((t - c) ** 2) / (2 * transient_width**2))
        for c, amp in zip(small_centers, small_amps)
    )
    huge_t = sum(
        amp * np.exp(-((t - c) ** 2) / (2 * transient_width**2))
        for c, amp in zip(huge_centers, huge_amps)
    )
    noise = 0.6 * rng.standard_normal(len(t))
    signal = baseline + drift + small_t + huge_t + noise
    control_fit = baseline + drift

    dff = deltaFF(signal, control_fit)
    z_std = zscore_standard(dff)
    z_mod = zscore_modified(dff)

    THRESHOLD = 3.0
    std_ymin, std_ymax = -1.5, 30.0
    mod_ymin, mod_ymax = -1.5, 30.0

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.6), sharex=True)
    ax_dff, ax_std, ax_mod = axes

    ax_dff.plot(t, dff, color=COLOR_DFF, linewidth=1.2)
    ax_dff.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff.set_ylabel("dF/F (%)", fontsize=9)
    ax_dff.set_title("dF/F: three small real events plus two outliers (40% at t = 5 s, 80% at t = 11 s)", loc="left", fontsize=10)

    def setup_zscore_panel(ax, z_trace, color, title, ymin, ymax, label_area=False):
        ax.axhspan(ymin, THRESHOLD, color="#cccccc", alpha=0.3, linewidth=0)
        ax.axhline(THRESHOLD, color="#666666", linewidth=0.9, linestyle="--")
        ax.plot(t, z_trace, color=color, linewidth=1.2)
        ax.axhline(0, color="#888888", linewidth=0.4, alpha=0.4)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(ymin, ymax)
        if label_area:
            ax.text(
                duration * 0.5,
                ymax * 0.85,
                "peak detection area (above threshold)",
                fontsize=8,
                color="#444",
                ha="center",
                va="center",
                style="italic",
            )

    setup_zscore_panel(
        ax_std,
        z_std,
        COLOR_ZSTD,
        "standard z-score (wrong): outliers contaminate the std; real events fall below the threshold",
        std_ymin,
        std_ymax,
        label_area=True,
    )
    ax_std.set_ylabel("standard z-score", fontsize=9)
    # Use the second peak in the trace, the first outlier at t = 5 s (deterministic from
    # huge_centers[0]), so the arrow tip lands exactly on its rendered peak. A tight
    # window around that t pins down the actual local-max position even with noise.
    arrow_target_t = huge_centers[0]
    arrow_target_idx = int(round(arrow_target_t * sample_rate))
    window_samples = int(0.5 * sample_rate)
    window_lo = max(0, arrow_target_idx - window_samples)
    window_hi = min(len(z_std), arrow_target_idx + window_samples + 1)
    peak_local_idx = window_lo + int(np.argmax(z_std[window_lo:window_hi]))
    peak_t = float(t[peak_local_idx])
    peak_z = float(z_std[peak_local_idx])
    ax_std.annotate(
        "missed by peak detector\n(below z = 3 threshold)",
        xy=(peak_t, peak_z),
        xytext=(peak_t - 2.5, 8.0),
        fontsize=8,
        color="#444",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
    )

    setup_zscore_panel(
        ax_mod,
        z_mod,
        COLOR_ZMOD,
        "modified z-score (right): MAD is robust to outliers; real events cross the threshold",
        mod_ymin,
        mod_ymax,
    )
    ax_mod.set_ylabel("modified z-score", fontsize=9)
    ax_mod.set_xlabel("time (s)")

    for hc in huge_centers:
        idx = int(round(hc * sample_rate))
        if z_mod[idx] > mod_ymax:
            ax_mod.annotate(
                f"→ z = {z_mod[idx]:+.0f}",
                xy=(hc, mod_ymax * 0.97),
                xytext=(hc + 0.3, mod_ymax * 0.75),
                fontsize=7,
                color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.7),
            )

    fig.tight_layout()
    fig.savefig(OUT / "fig3_modified_zscore_robustness.svg")
    plt.close(fig)


def figure_4_cross_session_limitation():
    """Two synthetic animals with the same SNR but different absolute response magnitudes.
    In dF/F units the difference is obvious; after standard z-scoring the two traces are
    visually identical, illustrating that z-score erases absolute-magnitude information
    across sessions."""
    rng = np.random.default_rng(11)
    sample_rate = 100
    duration = 30.0
    t = np.arange(0, duration, 1 / sample_rate)

    transient_centers = [5.0, 12.0, 18.0, 25.0]
    transient_amps = [3.5, 5.0, 4.0, 4.5]
    transient_width = 0.5

    drift_base = -8.0 * (t / t[-1])
    noise_base = 0.6 * rng.standard_normal(len(t))
    transient_base = sum(
        amp * np.exp(-((t - c) ** 2) / (2 * transient_width**2))
        for c, amp in zip(transient_centers, transient_amps)
    )

    def make_session(scale_factor):
        baseline = 100.0
        signal = baseline + drift_base + scale_factor * transient_base + scale_factor * noise_base
        control_fit = baseline + drift_base
        return signal, control_fit

    sig_a, ctrl_a = make_session(scale_factor=1.0)
    sig_b, ctrl_b = make_session(scale_factor=3.0)

    dff_a = deltaFF(sig_a, ctrl_a)
    dff_b = deltaFF(sig_b, ctrl_b)
    z_a = zscore_standard(dff_a)
    z_b = zscore_standard(dff_b)

    color_a = "#1f77b4"
    color_b = "#ff7f0e"

    fig = plt.figure(figsize=(8.5, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.6, wspace=0.1)
    ax_dff_a = fig.add_subplot(gs[0, 0])
    ax_dff_b = fig.add_subplot(gs[0, 1], sharey=ax_dff_a)
    ax_z = fig.add_subplot(gs[1, :])

    ax_dff_a.plot(t, dff_a, color=color_a, linewidth=1.0)
    ax_dff_a.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff_a.set_ylabel("dF/F (%)", fontsize=9)
    ax_dff_a.set_xlabel("time (s)", fontsize=9)
    ax_dff_a.set_title("Animal A (dF/F): small responses, low noise", loc="left", fontsize=10)

    ax_dff_b.plot(t, dff_b, color=color_b, linewidth=1.0)
    ax_dff_b.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_dff_b.set_xlabel("time (s)", fontsize=9)
    ax_dff_b.set_title("Animal B (dF/F): 3× larger responses, 3× higher noise (same SNR)", loc="left", fontsize=10)
    plt.setp(ax_dff_b.get_yticklabels(), visible=False)

    ax_z.plot(t, z_a, color=color_a, linewidth=1.4, label="Animal A", alpha=0.85)
    ax_z.plot(t, z_b, color=color_b, linewidth=1.4, label="Animal B", alpha=0.85, linestyle="--")
    ax_z.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_z.set_ylabel("standard z-score", fontsize=9)
    ax_z.set_xlabel("time (s)", fontsize=9)
    ax_z.legend(loc="upper right", frameon=False, fontsize=9)
    ax_z.set_title("both animals after standard z-scoring (visually indistinguishable)", loc="left", fontsize=10)

    fig.savefig(OUT / "fig4_cross_session_limitation.svg")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_zscore_cross_session()
    figure_2_baseline_zscore_contrast()
    figure_3_modified_zscore_robustness()
    figure_4_cross_session_limitation()
    print("Wrote SVGs to", OUT)
