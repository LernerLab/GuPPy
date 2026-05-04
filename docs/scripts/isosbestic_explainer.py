# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Generate SVG figures for the isosbestic-correction explanation page.

The correction math (np.polyfit degree 1, deltaFF) mirrors GuPPy's
src/guppy/analysis/z_score.py exactly, and the synthetic-fallback fit
(savgol smoothing + a + b * exp(-t/c)) mirrors helper_create_control_channel
in src/guppy/analysis/control_channel.py. Only the input traces are
synthetic, designed to make each pedagogical point clean.

Run with:

    uv run docs/scripts/isosbestic_explainer.py

Inline dependencies above let `uv run` resolve everything without needing
the surrounding project to be installed. Outputs are written directly to
docs/_static/images/isosbestic_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "isosbestic_explainer"
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

COLOR_SIGNAL = "#2ca02c"
COLOR_CONTROL = "#9467bd"
COLOR_FIT = "#888888"
COLOR_DFF = "#222222"
COLOR_SYNTHETIC = "#d62728"
COLOR_APO = "#e76f51"  # coral/terracotta: apo (calcium-free) conformation, abstract state label
COLOR_BOUND = "#2a9d8f"  # teal/seafoam: calcium-bound conformation, abstract state label
COLOR_CALCIUM = "#444444"  # neutral dark gray: calcium concentration curve


def control_fit(control, signal):
    """Mirror of GuPPy's controlFit (analysis/z_score.py)."""
    p = np.polyfit(control, signal, 1)
    return p[0] * control + p[1]


def delta_ff(signal, fit):
    """Mirror of GuPPy's deltaFF (analysis/z_score.py): percent units."""
    return (signal - fit) / fit * 100.0


def synthetic_exponential_control(signal, timestamps, window=101):
    """Mirror of helper_create_control_channel (analysis/control_channel.py)."""
    if window % 2 == 0:
        window += 1
    smoothed = savgol_filter(signal, window_length=window, polyorder=3)

    def model(x, a, b, c):
        return a + b * np.exp(-x / c)

    p0 = [smoothed[-1], smoothed[0] - smoothed[-1], 30.0]
    popt, _ = curve_fit(model, timestamps, smoothed, p0=p0, maxfev=10_000)
    return model(timestamps, *popt)


def make_bleach(t, tau=120, depth=0.3):
    return 1.0 - depth * (1.0 - np.exp(-t / tau))


def make_motion_bump(t, center, width, depth):
    return -depth * np.exp(-((t - center) ** 2) / (2 * width**2))


def make_calcium_transient(t, center, rise=0.3, decay=2.0, amplitude=0.15):
    out = np.zeros_like(t)
    mask = t >= center
    after = t[mask] - center
    out[mask] = amplitude * (1.0 - np.exp(-after / rise)) * np.exp(-after / decay)
    return out


def figure_1_response_vs_wavelength():
    """The empirical phenomenon: real-data-style fluorescence traces at three
    excitation wavelengths during a calcium event.

    Layout: a 2x1 grid. The top axis shows the underlying calcium event as a
    smooth black curve (the "input"). The bottom axis shows three fluorescence
    traces overlaid (the "outputs"), one per wavelength, each generated with
    typical GCaMP response kinetics (rise + decay) and mild Gaussian noise to
    look like a real recording. At 470 nm (green) the response is large; at
    420 nm (amber) it is smaller; at 405 nm (purple) the trace is essentially
    flat — just baseline noise.

    Pedagogical ratios for per-molecule emission (5:1 / 2:1 / 1:1) are the same
    as in fig3, so the wavelengths line up across figures. Channel colours
    (green for 470, purple for 405) match the photometry-data figures later in
    this page.
    """
    rng = np.random.default_rng(42)
    sample_rate = 100
    duration = 8.5
    t = np.arange(0, duration, 1 / sample_rate)

    def calcium_event(t, t0, amplitude, tau_rise=0.04, tau_decay=0.35):
        s = t - t0
        out = np.zeros_like(t)
        mask = s > 0
        out[mask] = amplitude * (1 - np.exp(-s[mask] / tau_rise)) * np.exp(-s[mask] / tau_decay)
        return out

    baseline = 0.10
    event_specs = [(2.0, 0.75, 0.38), (5.0, 1.0, 0.50)]
    calcium = baseline * np.ones_like(t)
    for t0, amp, td in event_specs:
        calcium = calcium + calcium_event(t, t0, amp, tau_rise=0.04, tau_decay=td)

    tau_rise_ind = 0.12
    tau_decay_ind = 0.55
    k_t = np.arange(0, 4.0, 1 / sample_rate)
    kernel = (1 - np.exp(-k_t / tau_rise_ind)) * np.exp(-k_t / tau_decay_ind)
    kernel = kernel / np.sum(kernel)
    calcium_filtered = np.convolve(calcium - baseline, kernel, mode="full")[: len(t)] + baseline

    min_bound = 0.20
    max_bound = 0.85
    ca_min = baseline
    ca_max = float(np.max(calcium))
    bound_fraction = min_bound + (max_bound - min_bound) * (calcium_filtered - ca_min) / (ca_max - ca_min)
    bound_fraction = np.clip(bound_fraction, 0, 1)
    apo_fraction = 1.0 - bound_fraction
    n_total = 24

    rows = [
        ("470 nm", "large response", 0.20, 1.00, "#2ca02c"),
        ("420 nm", "smaller response", 0.50, 1.00, "#b87333"),
        ("405 nm", "no response", 0.70, 0.70, "#9467bd"),
        ("390 nm", "inverted response", 1.00, 0.50, "#d62728"),
    ]

    noise_amp = 0.45
    traces = []
    for _, _, pa, pb, _ in rows:
        ideal = n_total * (apo_fraction * pa + bound_fraction * pb)
        noisy = ideal + noise_amp * rng.standard_normal(len(t))
        traces.append(noisy)

    peak_times = []
    for t0, _, _ in event_specs:
        local_start = int(t0 * sample_rate)
        local_end = int((t0 + 1.5) * sample_rate)
        local_segment = calcium[local_start:local_end]
        peak_idx = local_start + int(np.argmax(local_segment))
        peak_times.append(float(t[peak_idx]))

    fig = plt.figure(figsize=(9.0, 9.0))
    gs = GridSpec(
        5,
        1,
        height_ratios=[0.45, 1.0, 1.0, 1.0, 1.0],
        hspace=0.40,
        left=0.13,
        right=0.95,
        top=0.96,
        bottom=0.06,
    )

    ax_ca = fig.add_subplot(gs[0])
    for ptime in peak_times:
        ax_ca.axvline(ptime, color="#cccccc", linestyle=":", linewidth=0.8, alpha=0.85, zorder=0)
    ax_ca.plot(t, calcium, color="#222222", linewidth=1.6, zorder=2)
    ax_ca.set_xlim(t[0], t[-1])
    ax_ca.set_ylim(0, ca_max * 1.20)
    ax_ca.set_xticks([])
    ax_ca.set_yticks([])
    for spine in ax_ca.spines.values():
        spine.set_visible(False)
    ax_ca.set_ylabel(
        "[Ca²⁺]",
        rotation=0,
        ha="right",
        va="center",
        fontsize=11,
        color="#222222",
        labelpad=10,
    )

    y_max_f = max(np.max(tr) for tr in traces) * 1.05
    y_min_f = min(np.min(tr) for tr in traces) - 0.5

    panel_axes = []
    n_rows = len(rows)
    for row_idx, (trace, (wave_label, descriptor, _, _, color)) in enumerate(zip(traces, rows)):
        ax = fig.add_subplot(gs[row_idx + 1], sharex=ax_ca)
        for ptime in peak_times:
            ax.axvline(ptime, color="#cccccc", linestyle=":", linewidth=0.8, alpha=0.85, zorder=0)
        ax.plot(t, trace, color=color, linewidth=1.0, zorder=3)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_min_f, y_max_f)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine_name in ["top", "right"]:
            ax.spines[spine_name].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")

        is_last = row_idx == n_rows - 1
        if is_last:
            ax.spines["bottom"].set_color("#bbbbbb")
        else:
            ax.spines["bottom"].set_visible(False)

        ax.set_ylabel(
            f"{wave_label}\n{descriptor}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=10,
            color="#222222",
            labelpad=12,
        )

        panel_axes.append(ax)

    panel_axes[-1].set_xlabel("time", fontsize=10, color="#444444", labelpad=4)

    fig.savefig(OUT / "fig1_response_vs_wavelength.svg")
    plt.close(fig)


def figure_1b_response_columns():
    """Empirical phenomenon figure: four panels side by side as columns, one
    per wavelength. Each panel shows a single calcium event as a black trace
    overlaid on the wavelength-specific fluorescence trace. The bottom row
    contains a key/legend in the leftmost cell and a continuous
    ΔF-vs-wavelength curve spanning the remaining three cells.
    """
    _build_response_columns(spectrum_start_col=1, filename="fig1_response_columns.svg")


def figure_1_unified_decomposition():
    """Build the canonical unified figure: combined contributions row + signed
    bottom-left ΔF panel.
    """
    _build_unified_figure(decomposition="signed", combined=True, filename="fig3_decomposition.svg")


def figure_2a_calcium_chain():
    """Fig 2a: the chain calcium -> populations -> fluorescence at one
    canonical recording wavelength (470 nm), plus the equilibrium
    relationship as a side panel.

    Left column (3 rows, time-domain):
      - Top: calcium event over time.
      - Middle: apo and bound population fractions over time.
      - Bottom: total fluorescence response at 470 nm over time.
    Right column (full height): Hill saturation curve showing apo and bound
    fractions as a function of [Ca²⁺], with resting and peak markers.
    """
    sample_rate = 100
    duration = 4.5
    t = np.arange(0, duration, 1 / sample_rate)

    def calcium_event(t, t0, amplitude, tau_rise=0.04, tau_decay=0.45):
        s = t - t0
        out = np.zeros_like(t)
        mask = s > 0
        out[mask] = amplitude * (1 - np.exp(-s[mask] / tau_rise)) * np.exp(-s[mask] / tau_decay)
        return out

    baseline_ca = 0.10
    calcium = baseline_ca + calcium_event(t, t0=1.2, amplitude=1.0, tau_decay=0.45)

    tau_rise_ind = 0.12
    tau_decay_ind = 0.55
    k_t = np.arange(0, 4.0, 1 / sample_rate)
    kernel = (1 - np.exp(-k_t / tau_rise_ind)) * np.exp(-k_t / tau_decay_ind)
    kernel = kernel / np.sum(kernel)
    calcium_filtered = np.convolve(calcium - baseline_ca, kernel, mode="full")[: len(t)] + baseline_ca

    min_bound = 0.20
    max_bound = 0.85
    ca_max = float(np.max(calcium))
    bound_fraction = min_bound + (max_bound - min_bound) * (calcium_filtered - baseline_ca) / (ca_max - baseline_ca)
    bound_fraction = np.clip(bound_fraction, 0, 1)
    apo_fraction = 1.0 - bound_fraction

    n_total = 24
    # 420 nm: the per-molecule ratio is 2:1 (bound:apo). Bound still dominates,
    # but apo contributes visibly enough that the decomposition is legible.
    # At 470 nm (5:1) the bound trace would overlap the total trace and the
    # decomposition would be hard to read.
    per_apo = 0.50
    per_bound = 1.00
    sample_wavelength_label = "420 nm"
    sample_wavelength_color = "#b87333"  # amber, matches fig 2b's 420 column

    rng = np.random.default_rng(44)
    noise_amp = 0.45

    ideal_total = n_total * (apo_fraction * per_apo + bound_fraction * per_bound)
    baseline_total = float(n_total * (apo_fraction[0] * per_apo + bound_fraction[0] * per_bound))
    total_df = ideal_total + noise_amp * rng.standard_normal(len(t)) - baseline_total

    ideal_apo = n_total * apo_fraction * per_apo
    baseline_apo = float(n_total * apo_fraction[0] * per_apo)
    apo_contrib_df = ideal_apo + (noise_amp * 0.7) * rng.standard_normal(len(t)) - baseline_apo

    ideal_bound = n_total * bound_fraction * per_bound
    baseline_bound = float(n_total * bound_fraction[0] * per_bound)
    bound_contrib_df = ideal_bound + (noise_amp * 0.7) * rng.standard_normal(len(t)) - baseline_bound

    ca_grid = np.linspace(0, 2.0, 500)
    kd = 0.5
    n_hill = 2.5
    bound_curve = ca_grid**n_hill / (kd**n_hill + ca_grid**n_hill)
    apo_curve = 1.0 - bound_curve

    fig = plt.figure(figsize=(13.5, 6.6))
    gs = GridSpec(
        3,
        3,
        height_ratios=[0.6, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.30,
        hspace=0.45,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.15,
    )

    # Column header for the left chain (calcium -> populations -> fluorescence).
    # Promoted to a figure-level text so it sits above the panel and does not
    # collide with the "resting" / "peak" annotations on the calcium panel.
    fig.text(
        0.08,
        0.955,
        "calcium event drives populations, populations drive fluorescence",
        fontsize=12,
        color="#222222",
        ha="left",
        va="center",
    )

    # Identify resting and peak times for the time-domain vertical guides.
    # The "resting" vline sits before the event onset; the "peak" vline sits
    # at the peak of the (filtered) calcium signal that drives the populations.
    resting_time = 0.5
    peak_time = float(t[int(np.argmax(calcium_filtered))])

    # Calcium event (top-left)
    ax_ca = fig.add_subplot(gs[0, 0])
    for t_val in (resting_time, peak_time):
        ax_ca.axvline(t_val, color="#888888", linestyle=":", linewidth=0.8, alpha=0.65)
    ax_ca.text(resting_time, ca_max * 1.30, "resting", fontsize=8.5, color="#444444",
               ha="center", va="bottom", clip_on=False)
    ax_ca.text(peak_time, ca_max * 1.30, "peak", fontsize=8.5, color="#444444",
               ha="center", va="bottom", clip_on=False)
    ax_ca.plot(t, calcium, color="#222222", linewidth=2.0, zorder=3)
    ax_ca.set_xlim(t[0], t[-1])
    ax_ca.set_ylim(0, ca_max * 1.20)
    ax_ca.set_xticks([])
    ax_ca.set_yticks([])
    for spine in ax_ca.spines.values():
        spine.set_visible(False)
    ax_ca.set_ylabel(
        "[Ca²⁺]",
        rotation=0,
        ha="right",
        va="center",
        fontsize=11,
        color="#222222",
        labelpad=12,
    )

    # Apo and bound fractions over time (middle-left)
    ax_frac = fig.add_subplot(gs[1, 0], sharex=ax_ca)
    for t_val in (resting_time, peak_time):
        ax_frac.axvline(t_val, color="#888888", linestyle=":", linewidth=0.8, alpha=0.65, zorder=1)
    ax_frac.plot(t, apo_fraction, color=COLOR_APO, linewidth=2.0, label="apo fraction", zorder=3)
    ax_frac.plot(t, bound_fraction, color=COLOR_BOUND, linewidth=2.0, label="bound fraction", zorder=3)
    ax_frac.set_xlim(t[0], t[-1])
    ax_frac.set_ylim(-0.02, 1.05)
    ax_frac.set_xticks([])
    ax_frac.set_yticks([0, 0.5, 1.0])
    ax_frac.set_yticklabels(["0", "½", "1"], fontsize=9, color="#444444")
    for spine in ["top", "right"]:
        ax_frac.spines[spine].set_visible(False)
    ax_frac.spines["left"].set_color("#bbbbbb")
    ax_frac.spines["bottom"].set_visible(False)
    ax_frac.set_ylabel(
        "fraction",
        rotation=0,
        ha="right",
        va="center",
        fontsize=10,
        color="#222222",
        labelpad=12,
    )

    # Total fluorescence response decomposed (bottom-left). Three lines:
    # apo contribution (coral), bound contribution (teal), and their sum
    # (the total, in the wavelength's column colour). At 420 nm the per-
    # molecule ratio is 2:1 (bound:apo), so both contributions are clearly
    # visible without total overlapping bound.
    ax_f = fig.add_subplot(gs[2, 0], sharex=ax_ca)
    for t_val in (resting_time, peak_time):
        ax_f.axvline(t_val, color="#888888", linestyle=":", linewidth=0.8, alpha=0.65, zorder=1)
    ax_f.axhline(0, color="#dddddd", linewidth=0.7, alpha=0.7, zorder=2)
    ax_f.plot(t, apo_contrib_df, color=COLOR_APO, linewidth=1.3, zorder=3)
    ax_f.plot(t, bound_contrib_df, color=COLOR_BOUND, linewidth=1.3, zorder=3)
    ax_f.plot(t, total_df, color=sample_wavelength_color, linewidth=1.7, zorder=4)
    ax_f.set_xlim(t[0], t[-1])
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    for spine in ["top", "right"]:
        ax_f.spines[spine].set_visible(False)
    ax_f.spines["left"].set_color("#bbbbbb")
    ax_f.spines["bottom"].set_color("#bbbbbb")
    ax_f.set_xlabel("time  →", fontsize=10, color="#444444", labelpad=4)
    ax_f.set_ylabel(
        "ΔF",
        rotation=0,
        ha="right",
        va="center",
        fontsize=11,
        color="#222222",
        labelpad=12,
    )
    ax_f.set_title(
        f"at {sample_wavelength_label}",
        fontsize=10, color="#444444", pad=4,
    )

    # Hill curve (right two columns, spanning all rows for a wider aspect).
    # Clean lines only — no fills, no per-curve markers. The vertical
    # "resting" and "peak" guides cross-reference matching guides on the
    # time-domain panels on the left.
    ax_hill = fig.add_subplot(gs[0:2, 1:3])

    ax_hill.plot(ca_grid, apo_curve, color=COLOR_APO, linewidth=3.0, label="apo", zorder=3)
    ax_hill.plot(ca_grid, bound_curve, color=COLOR_BOUND, linewidth=3.0, label="bound", zorder=3)

    resting_ca = 0.18
    peak_ca = 1.25
    for ca_val, ca_label in [(resting_ca, "resting"), (peak_ca, "peak")]:
        ax_hill.axvline(ca_val, color="#888888", linestyle=":", linewidth=0.8, alpha=0.65)
        ax_hill.text(
            ca_val,
            1.08,
            ca_label,
            fontsize=10.5,
            color="#444444",
            ha="center",
            va="bottom",
        )

    ax_hill.set_xlim(0, 2.0)
    ax_hill.set_ylim(0, 1.18)
    ax_hill.set_xticks([])
    ax_hill.set_yticks([0, 0.5, 1.0])
    ax_hill.set_yticklabels(["0", "½", "1"], fontsize=11, color="#444444")
    for spine in ["top", "right"]:
        ax_hill.spines[spine].set_visible(False)
    ax_hill.spines["left"].set_color("#bbbbbb")
    ax_hill.spines["bottom"].set_color("#bbbbbb")
    ax_hill.set_xlabel("calcium [Ca²⁺]  →", fontsize=12, color="#444444", labelpad=6)
    ax_hill.set_ylabel(
        "fraction",
        rotation=0,
        ha="right",
        va="center",
        fontsize=12,
        color="#222222",
        labelpad=12,
    )

    # Global legend at the bottom of the figure.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#222222", linewidth=2.0, label="calcium event"),
        Line2D([0], [0], color=COLOR_APO, linewidth=2.0, label="apo"),
        Line2D([0], [0], color=COLOR_BOUND, linewidth=2.0, label="bound"),
        Line2D([0], [0], color=sample_wavelength_color, linewidth=2.0,
               label=f"total ΔF (at {sample_wavelength_label})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.savefig(OUT / "fig2_calcium_chain.svg")
    plt.close(fig)


def _build_unified_figure(decomposition, combined, filename):
    """Unified empirical-and-mechanism figure: top calcium-reference row,
    three rows of time-domain decomposition (total / apo / bound) at four
    wavelengths, and a bottom row with ΔF-vs-wavelength on the left and the
    excitation spectra on the right.

    Decomposition styles for the bottom-left ΔF panel:
      - "signed": bound contribution drawn above zero (teal area), apo
        contribution drawn below zero as a loss (coral area, mirrored). Their
        sum is the ΔF curve, drawn as a black line that crosses zero at the
        isosbestic. Mathematically pure: the two coloured areas literally add
        to give ΔF.
      - "absolute": bound contribution and apo contribution both drawn above
        zero (positive magnitudes). The ΔF curve is shown as a separate black
        line representing the difference. The crossing of the two coloured
        curves marks the isosbestic.
    """
    rng = np.random.default_rng(43)
    sample_rate = 100
    duration = 4.5
    t = np.arange(0, duration, 1 / sample_rate)

    def calcium_event(t, t0, amplitude, tau_rise=0.04, tau_decay=0.45):
        s = t - t0
        out = np.zeros_like(t)
        mask = s > 0
        out[mask] = amplitude * (1 - np.exp(-s[mask] / tau_rise)) * np.exp(-s[mask] / tau_decay)
        return out

    baseline_ca = 0.10
    calcium = baseline_ca + calcium_event(t, t0=1.2, amplitude=1.0, tau_decay=0.45)

    tau_rise_ind = 0.12
    tau_decay_ind = 0.55
    k_t = np.arange(0, 4.0, 1 / sample_rate)
    kernel = (1 - np.exp(-k_t / tau_rise_ind)) * np.exp(-k_t / tau_decay_ind)
    kernel = kernel / np.sum(kernel)
    calcium_filtered = np.convolve(calcium - baseline_ca, kernel, mode="full")[: len(t)] + baseline_ca

    min_bound = 0.20
    max_bound = 0.85
    ca_max = float(np.max(calcium))
    bound_fraction = min_bound + (max_bound - min_bound) * (calcium_filtered - baseline_ca) / (ca_max - baseline_ca)
    bound_fraction = np.clip(bound_fraction, 0, 1)
    apo_fraction = 1.0 - bound_fraction
    n_total = 24

    rows_wavelengths = [
        ("470 nm", "large response", 0.20, 1.00, "#2ca02c"),
        ("420 nm", "smaller response", 0.50, 1.00, "#b87333"),
        ("405 nm", "no response", 0.70, 0.70, "#9467bd"),
        ("390 nm", "inverted response", 1.00, 0.50, "#d62728"),
    ]

    noise_amp = 0.45
    total_traces = []
    apo_traces = []
    bound_traces = []

    for _, _, pa, pb, _ in rows_wavelengths:
        ideal_total = n_total * (apo_fraction * pa + bound_fraction * pb)
        baseline_total = float(n_total * (apo_fraction[0] * pa + bound_fraction[0] * pb))
        noisy_total = ideal_total + noise_amp * rng.standard_normal(len(t))
        total_traces.append(noisy_total - baseline_total)

        ideal_apo = n_total * apo_fraction * pa
        baseline_apo = float(n_total * apo_fraction[0] * pa)
        noisy_apo = ideal_apo + (noise_amp * 0.7) * rng.standard_normal(len(t))
        apo_traces.append(noisy_apo - baseline_apo)

        ideal_bound = n_total * bound_fraction * pb
        baseline_bound = float(n_total * bound_fraction[0] * pb)
        noisy_bound = ideal_bound + (noise_amp * 0.7) * rng.standard_normal(len(t))
        bound_traces.append(noisy_bound - baseline_bound)

    all_traces = total_traces + apo_traces + bound_traces
    y_max_f = max(np.max(tr) for tr in all_traces) * 1.10
    y_min_f = min(np.min(tr) for tr in all_traces) * 1.10

    if combined:
        n_time_rows = 2
        fig_height = 9.5
        height_ratios = [1.0, 1.0, 1.3]
    else:
        n_time_rows = 3
        fig_height = 11.0
        height_ratios = [1.0, 1.0, 1.0, 1.3]

    fig = plt.figure(figsize=(13.0, fig_height))
    gs = GridSpec(
        n_time_rows + 1,
        4,
        height_ratios=height_ratios,
        wspace=0.10,
        hspace=0.30,
        left=0.10,
        right=0.97,
        top=0.95,
        bottom=0.05,
    )

    if combined:
        row_configs = [
            ("total ΔF", [(total_traces[i], None, [c for _, _, _, _, c in rows_wavelengths][i])
                          for i in range(4)]),
            ("contributions\n(apo + bound)",
             [(apo_traces[i], bound_traces[i], None) for i in range(4)]),
        ]
    else:
        row_configs = [
            ("total ΔF", [(total_traces[i], None, [c for _, _, _, _, c in rows_wavelengths][i])
                          for i in range(4)]),
            ("apo\ncontribution",
             [(apo_traces[i], None, COLOR_APO) for i in range(4)]),
            ("bound\ncontribution",
             [(bound_traces[i], None, COLOR_BOUND) for i in range(4)]),
        ]

    first_ax = None
    for row_idx, (row_label, cell_specs) in enumerate(row_configs):
        for col_idx, (trace_a, trace_b, color) in enumerate(cell_specs):
            ax = fig.add_subplot(gs[row_idx, col_idx], sharex=first_ax)
            if first_ax is None:
                first_ax = ax
            ax.axhline(0, color="#dddddd", linewidth=0.7, alpha=0.7, zorder=1)

            if trace_b is not None:
                # Combined contributions cell: plot apo (coral) and bound (teal)
                # both centred at zero. apo goes negative (loss), bound goes
                # positive (gain). The total is shown separately in row 1; we
                # do not redraw it here. At the isosbestic column the apo and
                # bound traces are mirror images, making the cancellation a
                # single-cell visual fact.
                ax.plot(t, trace_a, color=COLOR_APO, linewidth=1.4, zorder=3,
                        label="apo (loss)" if (row_idx == 1 and col_idx == 0) else None)
                ax.plot(t, trace_b, color=COLOR_BOUND, linewidth=1.4, zorder=3,
                        label="bound (gain)" if (row_idx == 1 and col_idx == 0) else None)
            else:
                ax.plot(t, trace_a, color=color, linewidth=1.2, zorder=3)

            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(y_min_f, y_max_f)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ["top", "right", "bottom"]:
                ax.spines[spine].set_visible(False)

            if col_idx == 0:
                ax.spines["left"].set_color("#bbbbbb")
                ax.set_ylabel(
                    row_label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=10.5,
                    color="#222222",
                    labelpad=14,
                )
            else:
                ax.spines["left"].set_visible(False)

            if row_idx == 0:
                wl_label, wl_descriptor = rows_wavelengths[col_idx][0], rows_wavelengths[col_idx][1]
                ax.set_title(
                    f"{wl_label}\n{wl_descriptor}",
                    fontsize=10,
                    color="#222222",
                    pad=8,
                )

    def split_normal(x, mu, sl, sr):
        sigma = np.where(x < mu, sl, sr)
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Parameters chosen so the isosbestic crossing lands near 405 nm
    # (canonical GCaMP value) while keeping bound substantially brighter
    # than apo at 470 nm. The trade-off: apo is small at 470 nm because the
    # apo split-normal has a sharp right-side falloff. Adjusting the apo to
    # be much broader on the right would push the crossing to higher
    # wavelengths and break the 405 nm convention.
    wl_grid = np.linspace(350, 590, 1000)
    apo_spec_curve = split_normal(wl_grid, mu=395, sl=14, sr=30)
    bound_spec_curve = 3.0 * split_normal(wl_grid, mu=488, sl=55, sr=36)

    # Find the meaningful isosbestic crossing (where apo and bound spectra
    # actually cross at non-trivial values). argmin(abs(diff)) is wrong here
    # because both curves are near zero in the violet tail, producing a
    # spurious "crossing" at ~375 nm that is not the real isosbestic.
    diff = apo_spec_curve - bound_spec_curve
    search_mask = (wl_grid >= 390) & (wl_grid <= 480)
    sign_changes = np.where(np.diff(np.sign(diff[search_mask])))[0]
    iso_idx = int(np.where(search_mask)[0][sign_changes[0]])
    iso_wl = float(wl_grid[iso_idx])
    iso_val = float(apo_spec_curve[iso_idx])

    sample_specs = [(470, "#2ca02c"), (420, "#b87333"), (405, "#9467bd"), (390, "#d62728")]

    # ΔF panel with decomposition. Spans the rightmost two cells of the bottom
    # row; legend on the left (cell 0), empty middle cell (cell 1) for
    # breathing room.
    bottom_row_idx = n_time_rows
    ax_curve = fig.add_subplot(gs[bottom_row_idx, 2:4])
    delta_f_curve = bound_spec_curve - apo_spec_curve

    ax_curve.axhline(0, color="#cccccc", linewidth=0.8, alpha=0.7, zorder=1)

    if decomposition == "signed":
        ax_curve.fill_between(
            wl_grid, 0, bound_spec_curve, color=COLOR_BOUND, alpha=0.25, zorder=2,
            label="bound contribution",
        )
        ax_curve.fill_between(
            wl_grid, 0, -apo_spec_curve, color=COLOR_APO, alpha=0.25, zorder=2,
            label="apo contribution (loss)",
        )
        ax_curve.plot(wl_grid, bound_spec_curve, color=COLOR_BOUND, linewidth=1.4, zorder=3)
        ax_curve.plot(wl_grid, -apo_spec_curve, color=COLOR_APO, linewidth=1.4, zorder=3)
        ax_curve.plot(wl_grid, delta_f_curve, color="#222222", linewidth=2.2, zorder=4, label="ΔF (sum)")
    else:
        ax_curve.fill_between(
            wl_grid, 0, bound_spec_curve, color=COLOR_BOUND, alpha=0.25, zorder=2,
            label="bound (gain magnitude)",
        )
        ax_curve.fill_between(
            wl_grid, 0, apo_spec_curve, color=COLOR_APO, alpha=0.25, zorder=2,
            label="apo (loss magnitude)",
        )
        ax_curve.plot(wl_grid, bound_spec_curve, color=COLOR_BOUND, linewidth=1.4, zorder=3)
        ax_curve.plot(wl_grid, apo_spec_curve, color=COLOR_APO, linewidth=1.4, zorder=3)
        ax_curve.plot(wl_grid, delta_f_curve, color="#222222", linewidth=2.2, zorder=4, label="ΔF (bound − apo)")

    # Mark the isosbestic crossing where the ΔF curve crosses zero.
    ax_curve.axvline(iso_wl, color="#444444", linestyle="--", linewidth=1.2, alpha=0.85, zorder=3)
    y_top = float(np.max(delta_f_curve)) * 1.05
    ax_curve.text(
        iso_wl, y_top, "isosbestic", fontsize=9, color="#222222",
        ha="center", va="bottom", clip_on=False,
    )

    for wl, color in sample_specs:
        delta_at_wl = float(np.interp(wl, wl_grid, delta_f_curve))
        ax_curve.axvline(wl, color="#cccccc", linestyle=":", linewidth=0.7, alpha=0.6, zorder=2)
        ax_curve.scatter(
            [wl], [delta_at_wl], color=color, s=140, zorder=5,
            edgecolors="white", linewidths=2.0,
        )

    ax_curve.set_xlim(350, 590)
    ax_curve.set_xticks([360, 390, 405, 420, 470])
    ax_curve.set_xticklabels(["360", "390", "405", "420", "470"], fontsize=9, color="#444444")
    ax_curve.set_yticks([])
    for spine in ["top", "right"]:
        ax_curve.spines[spine].set_visible(False)
    ax_curve.spines["left"].set_color("#bbbbbb")
    ax_curve.spines["bottom"].set_color("#bbbbbb")
    ax_curve.set_xlabel("excitation wavelength (nm)", fontsize=10, color="#444444", labelpad=4)
    ax_curve.set_ylabel(
        "ΔF",
        rotation=0,
        ha="right",
        va="center",
        fontsize=11,
        color="#222222",
        labelpad=10,
    )
    ax_curve.set_title(
        "response vs wavelength",
        fontsize=10, color="#222222", pad=4,
    )

    # The bottom-left cell (gs[bottom_row_idx, 0]) holds the legend; the
    # middle cell (gs[bottom_row_idx, 1]) is intentionally empty so the
    # ΔF panel does not look stretched and the layout has breathing room.
    if combined:
        ax_legend = fig.add_subplot(gs[bottom_row_idx, 0])
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        for spine in ax_legend.spines.values():
            spine.set_visible(False)
        ax_legend.text(0.0, 0.96, "key", fontsize=11, weight="bold", color="#222222",
                       ha="left", va="top")
        ax_legend.plot([0.04, 0.20], [0.78, 0.78], color=COLOR_APO, linewidth=2.4)
        ax_legend.text(0.24, 0.78, "apo contribution (loss)", fontsize=10, color="#222222",
                       ha="left", va="center")
        ax_legend.plot([0.04, 0.20], [0.62, 0.62], color=COLOR_BOUND, linewidth=2.4)
        ax_legend.text(0.24, 0.62, "bound contribution (gain)", fontsize=10, color="#222222",
                       ha="left", va="center")
        ax_legend.plot([0.04, 0.20], [0.46, 0.46], color="#222222", linewidth=2.4)
        ax_legend.text(0.24, 0.46, "total ΔF (sum of contributions)", fontsize=10, color="#222222",
                       ha="left", va="center")
        ax_legend.text(
            0.0, 0.22,
            "ΔF: change in fluorescence\nrelative to resting baseline",
            fontsize=9, color="#444444", ha="left", va="top",
        )

    # "time →" label between the time-domain rows and the wavelength-domain row
    time_label_y = 0.40 if combined else 0.355
    fig.text(
        0.5,
        time_label_y,
        "time  →",
        fontsize=11,
        color="#444444",
        ha="center",
        va="center",
    )

    fig.savefig(OUT / filename)
    plt.close(fig)


def _build_response_columns(spectrum_start_col, filename):
    rng = np.random.default_rng(43)
    sample_rate = 100
    duration = 4.5
    t = np.arange(0, duration, 1 / sample_rate)

    def calcium_event(t, t0, amplitude, tau_rise=0.04, tau_decay=0.45):
        s = t - t0
        out = np.zeros_like(t)
        mask = s > 0
        out[mask] = amplitude * (1 - np.exp(-s[mask] / tau_rise)) * np.exp(-s[mask] / tau_decay)
        return out

    baseline_ca = 0.10
    calcium = baseline_ca + calcium_event(t, t0=1.2, amplitude=1.0, tau_decay=0.45)

    tau_rise_ind = 0.12
    tau_decay_ind = 0.55
    k_t = np.arange(0, 4.0, 1 / sample_rate)
    kernel = (1 - np.exp(-k_t / tau_rise_ind)) * np.exp(-k_t / tau_decay_ind)
    kernel = kernel / np.sum(kernel)
    calcium_filtered = np.convolve(calcium - baseline_ca, kernel, mode="full")[: len(t)] + baseline_ca

    min_bound = 0.20
    max_bound = 0.85
    ca_max = float(np.max(calcium))
    bound_fraction = min_bound + (max_bound - min_bound) * (calcium_filtered - baseline_ca) / (ca_max - baseline_ca)
    bound_fraction = np.clip(bound_fraction, 0, 1)
    apo_fraction = 1.0 - bound_fraction
    n_total = 24

    rows = [
        ("470 nm", "large response", 0.20, 1.00, "#2ca02c"),
        ("420 nm", "smaller response", 0.50, 1.00, "#b87333"),
        ("405 nm", "no response", 0.70, 0.70, "#9467bd"),
        ("390 nm", "inverted response", 1.00, 0.50, "#d62728"),
    ]

    noise_amp = 0.45
    traces_df = []
    for _, _, pa, pb, _ in rows:
        ideal = n_total * (apo_fraction * pa + bound_fraction * pb)
        baseline_value = float(n_total * (apo_fraction[0] * pa + bound_fraction[0] * pb))
        noisy = ideal + noise_amp * rng.standard_normal(len(t))
        traces_df.append(noisy - baseline_value)

    y_max_f = max(np.max(tr) for tr in traces_df) * 1.10
    y_min_f = min(np.min(tr) for tr in traces_df) * 1.10

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = GridSpec(
        2,
        4,
        height_ratios=[1.0, 1.1],
        wspace=0.10,
        hspace=0.55,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.09,
    )

    ca_scale = 4.0
    calcium_centered_scaled = (calcium - baseline_ca) * ca_scale

    sample_specs = [
        (470, "#2ca02c"),
        (420, "#b87333"),
        (405, "#9467bd"),
        (390, "#d62728"),
    ]

    for col_idx, (trace, (wave_label, descriptor, _, _, color)) in enumerate(zip(traces_df, rows)):
        ax_f = fig.add_subplot(gs[0, col_idx])

        ax_f.axhline(0, color="#dddddd", linewidth=0.7, alpha=0.7, zorder=1)
        ax_f.plot(t, calcium_centered_scaled, color="#222222", linewidth=1.7, alpha=0.85, zorder=2)
        ax_f.plot(t, trace, color=color, linewidth=1.1, zorder=3)
        ax_f.set_xlim(t[0], t[-1])
        ax_f.set_ylim(y_min_f, y_max_f)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        for spine in ["top", "right", "bottom"]:
            ax_f.spines[spine].set_visible(False)

        ax_f.set_title(
            f"{wave_label}\n{descriptor}",
            fontsize=10,
            color="#222222",
            pad=8,
        )

        if col_idx == 0:
            ax_f.spines["left"].set_color("#bbbbbb")
            ax_f.set_ylabel(
                "ΔF",
                rotation=0,
                ha="right",
                va="center",
                fontsize=11,
                color="#222222",
                labelpad=10,
            )
        else:
            ax_f.spines["left"].set_visible(False)

    # Bottom-left cell(s): legend / key for the figure.
    ax_legend = fig.add_subplot(gs[1, 0:spectrum_start_col])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for spine in ax_legend.spines.values():
        spine.set_visible(False)

    line_x = (0.04, 0.16)
    text_x = 0.20

    ax_legend.text(
        0.0,
        0.96,
        "key",
        fontsize=11,
        color="#222222",
        weight="bold",
        ha="left",
        va="top",
    )

    ax_legend.plot(line_x, [0.82, 0.82], color="#222222", linewidth=2.2, alpha=0.9)
    ax_legend.text(
        text_x,
        0.82,
        "calcium event",
        fontsize=10,
        color="#222222",
        ha="left",
        va="center",
    )

    legend_y_positions = [0.66, 0.54, 0.42, 0.30]
    for (wave_label, descriptor, _, _, color), y_pos in zip(rows, legend_y_positions):
        ax_legend.plot(line_x, [y_pos, y_pos], color=color, linewidth=2.2)
        ax_legend.text(
            text_x,
            y_pos,
            f"{wave_label}: {descriptor}",
            fontsize=9.5,
            color="#222222",
            ha="left",
            va="center",
        )

    ax_legend.text(
        0.0,
        0.16,
        "ΔF: fluorescence change\nrelative to resting baseline",
        fontsize=9,
        color="#444444",
        ha="left",
        va="top",
    )

    # Bottom row, remaining cells: ΔF as a function of excitation wavelength.
    ax_curve = fig.add_subplot(gs[1, spectrum_start_col:4])

    def _split_normal(x, mu, sl, sr):
        sigma = np.where(x < mu, sl, sr)
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    wl_grid = np.linspace(380, 590, 900)
    apo_spec = _split_normal(wl_grid, mu=395, sl=14, sr=30)
    bound_spec = 3.5 * _split_normal(wl_grid, mu=488, sl=55, sr=36)
    delta_f_curve = bound_spec - apo_spec

    ax_curve.axhline(0, color="#cccccc", linewidth=0.8, alpha=0.7, zorder=1)
    ax_curve.fill_between(
        wl_grid,
        0,
        delta_f_curve,
        where=(delta_f_curve >= 0),
        color="#bbbbbb",
        alpha=0.18,
        zorder=2,
    )
    ax_curve.fill_between(
        wl_grid,
        0,
        delta_f_curve,
        where=(delta_f_curve < 0),
        color="#bbbbbb",
        alpha=0.18,
        zorder=2,
    )
    ax_curve.plot(wl_grid, delta_f_curve, color="#222222", linewidth=2.0, zorder=3)

    for wl, color in sample_specs:
        delta_at_wl = float(np.interp(wl, wl_grid, delta_f_curve))
        ax_curve.axvline(wl, color="#cccccc", linestyle=":", linewidth=0.7, alpha=0.6, zorder=2)
        ax_curve.scatter(
            [wl],
            [delta_at_wl],
            color=color,
            s=140,
            zorder=5,
            edgecolors="white",
            linewidths=2.0,
        )

    ax_curve.set_xlim(380, 590)
    ax_curve.set_xticks([390, 405, 420, 470])
    ax_curve.set_xticklabels(["390", "405", "420", "470"], fontsize=9.5, color="#444444")
    ax_curve.set_yticks([])
    for spine in ["top", "right"]:
        ax_curve.spines[spine].set_visible(False)
    ax_curve.spines["left"].set_color("#bbbbbb")
    ax_curve.spines["bottom"].set_color("#bbbbbb")
    ax_curve.set_xlabel("excitation wavelength (nm)", fontsize=10, color="#444444", labelpad=4)
    ax_curve.set_ylabel(
        "ΔF",
        rotation=0,
        ha="right",
        va="center",
        fontsize=11,
        color="#222222",
        labelpad=10,
    )

    # "time →" label, positioned in the upper-centre region of the figure
    # close to the bottom edge of the top row of wavelength panels.
    fig.text(
        0.55,
        0.66,
        "time  ⟶",
        fontsize=14,
        color="#222222",
        weight="bold",
        ha="center",
        va="center",
    )

    fig.savefig(OUT / filename)
    plt.close(fig)


def figure_2_excitation_spectra():
    """Schematic excitation spectra of the two GCaMP conformational states.

    Lives at the end of the Background section as a summary that ties fig2's
    three wavelengths back to the underlying spectral structure. Visually
    designed as a schematic, not a measured plot:
      - asymmetric "split-normal" peaks (different sigma on each side) instead
        of pure Gaussians, to break the over-symmetric "computed schematic"
        feel and to match the qualitative asymmetry of real GCaMP spectra
      - semi-transparent fills under each curve give the curves visual weight
        and make their overlap region readable
      - axis ticks and numbers are dropped; axis labels are kept
      - wavelength markers are positioned in the top margin with thin leader
        lines down to the plot, to avoid clutter through the curves
    """
    wavelengths = np.linspace(370, 560, 800)

    def split_normal(x, mu, sigma_left, sigma_right):
        sigma = np.where(x < mu, sigma_left, sigma_right)
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    apo = split_normal(wavelengths, mu=395, sigma_left=14, sigma_right=30)
    bound = 3.5 * split_normal(wavelengths, mu=488, sigma_left=55, sigma_right=36)

    valid = wavelengths > 380
    diffs = np.abs(apo - bound)
    diffs_valid = np.where(valid, diffs, np.inf)
    iso_idx = int(np.argmin(diffs_valid))
    iso_wavelength = float(wavelengths[iso_idx])
    iso_value = float(apo[iso_idx])

    y_top = max(apo.max(), bound.max()) * 1.40

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    ax.fill_between(wavelengths, 0, apo, color=COLOR_APO, alpha=0.30, zorder=1)
    ax.fill_between(wavelengths, 0, bound, color=COLOR_BOUND, alpha=0.30, zorder=1)
    ax.plot(wavelengths, apo, color=COLOR_APO, linewidth=2.4, label="apo (calcium-free)", zorder=3)
    ax.plot(wavelengths, bound, color=COLOR_BOUND, linewidth=2.4, label="Ca²⁺-bound", zorder=3)

    ax.scatter([iso_wavelength], [iso_value], color="#222222", zorder=5, s=26)

    guides = [
        (470.0, "470 nm\nfar from isosbestic"),
        (420.0, "420 nm\ncloser to isosbestic"),
        (iso_wavelength, f"{iso_wavelength:.0f} nm\nat isosbestic"),
    ]

    label_y = y_top * 0.88
    for lam, label in guides:
        ax.plot(
            [lam, lam],
            [0, label_y - y_top * 0.05],
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )
        ax.text(
            lam,
            label_y,
            label,
            fontsize=8.8,
            color="#222222",
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "boxstyle": "round,pad=0.35",
                "alpha": 0.95,
            },
        )

    ax.set_xlabel("excitation wavelength")
    ax.set_ylabel("fluorescence per protein")
    ax.set_xlim(380, 560)
    ax.set_ylim(0, y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center right", frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_excitation_spectra.svg")
    plt.close(fig)


def figure_3_contribution_decomposition():
    """Decomposition of total emission into apo and bound contributions, as a 3x2
    grid of stacked bars (rows = wavelength, columns = before / at peak calcium).

    Each cell is a single stacked bar showing the total emission at that
    wavelength and timepoint, decomposed into a coral (apo) segment at the
    bottom and a teal (bound) segment stacked on top. The bar's full height is
    the total emission; its internal split shows the contribution decomposition.

    The conceptual punchline lives in the bottom row (at the isosbestic
    wavelength): the two bars (before / at peak) reach exactly the same total
    height, but their internal composition is reversed (coral dominates before,
    teal dominates at peak). Same total, different decomposition: the
    cancellation is the equality of bar heights despite the population shift.
    The top row (470 nm, far from isosbestic) shows the contrast: the at-peak
    bar is much taller than the before bar. The middle row (420 nm) is in
    between.

    Per-molecule emission ratios are pedagogical (5:1, 2:1, 1:1) rather than
    drawn from the spectral model in fig2, so that both segments stay visible.
    """
    apo_fraction_before = 0.80
    apo_fraction_during = 0.15
    bound_fraction_before = 1.0 - apo_fraction_before
    bound_fraction_during = 1.0 - apo_fraction_during
    n_total = 24

    # Pedagogical per-molecule emissions
    rows = [
        ("470 nm", "far from isosbestic", 0.20, 1.00, "ΔF large"),
        ("420 nm", "closer to isosbestic", 0.50, 1.00, "ΔF small"),
        ("405 nm", "at isosbestic", 0.70, 0.70, "ΔF = 0"),
    ]

    cells = []
    for _, _, per_apo, per_bound, _ in rows:
        before = (
            n_total * apo_fraction_before * per_apo,
            n_total * bound_fraction_before * per_bound,
        )
        during = (
            n_total * apo_fraction_during * per_apo,
            n_total * bound_fraction_during * per_bound,
        )
        cells.append((before, during))

    y_max = max(a + b for col in cells for a, b in col) * 1.20

    fig = plt.figure(figsize=(7.4, 7.6))
    gs = GridSpec(
        3,
        2,
        hspace=0.32,
        wspace=0.18,
        left=0.22,
        right=0.94,
        top=0.92,
        bottom=0.05,
    )

    bar_x = 0.5
    bar_width = 0.55

    panel_axes = []
    for row_idx, ((before, during), (wave_label, descriptor, _, _, deltaf_label)) in enumerate(
        zip(cells, rows)
    ):
        for col_idx, (apo_val, bound_val) in enumerate([before, during]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            total = apo_val + bound_val

            ax.bar(
                bar_x,
                apo_val,
                width=bar_width,
                color=COLOR_APO,
                edgecolor="#222222",
                linewidth=0.6,
                label="apo contribution" if (row_idx == 0 and col_idx == 0) else None,
                zorder=3,
            )
            ax.bar(
                bar_x,
                bound_val,
                width=bar_width,
                bottom=apo_val,
                color=COLOR_BOUND,
                edgecolor="#222222",
                linewidth=0.6,
                label="bound contribution" if (row_idx == 0 and col_idx == 0) else None,
                zorder=3,
            )

            # Reference line for the resting (before) total in this row
            resting_total = before[0] + before[1]
            ax.axhline(
                resting_total,
                color="#888888",
                linestyle="--",
                linewidth=0.7,
                alpha=0.65,
                zorder=1,
            )

            # Total label above the bar
            ax.text(
                bar_x,
                total + y_max * 0.03,
                f"{total:.1f}",
                fontsize=8.5,
                color="#444444",
                ha="center",
                va="bottom",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_max)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine_name in ["top", "right", "bottom"]:
                ax.spines[spine_name].set_visible(False)
            ax.spines["left"].set_color("#bbbbbb")

            if col_idx == 0:
                ax.set_ylabel(
                    f"{wave_label}\n{descriptor}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="#222222",
                    labelpad=14,
                )

            if col_idx == 1:
                ax.text(
                    1.05,
                    y_max * 0.5,
                    deltaf_label,
                    fontsize=10,
                    color="#222222",
                    weight="bold",
                    ha="left",
                    va="center",
                    transform=ax.transData,
                )

            if row_idx == 0:
                ax.set_title(
                    "before peak" if col_idx == 0 else "at peak",
                    fontsize=10.5,
                    color="#222222",
                )

            panel_axes.append(ax)

    panel_axes[0].legend(
        loc="upper left",
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.0, 1.30),
    )

    fig.savefig(OUT / "fig3_contribution_decomposition.svg")
    plt.close(fig)


def figure_4_what_405_captures():
    """Three scenarios: motion shows in both, bleach shows in both, calcium only in 470."""
    rng = np.random.default_rng(0)
    sample_rate = 100
    duration = 30
    t = np.arange(0, duration, 1 / sample_rate)

    base_405 = 0.60
    base_470 = 1.00
    noise_amp = 0.004

    motion = make_motion_bump(t, center=15, width=0.4, depth=0.10)
    s_a_405 = base_405 * (1 + motion) + noise_amp * rng.standard_normal(len(t))
    s_a_470 = base_470 * (1 + motion) + noise_amp * rng.standard_normal(len(t))

    bleach = make_bleach(t, tau=15, depth=0.25)
    s_b_405 = base_405 * bleach + noise_amp * rng.standard_normal(len(t))
    s_b_470 = base_470 * bleach + noise_amp * rng.standard_normal(len(t))

    calcium = make_calcium_transient(t, center=15, amplitude=0.18, rise=0.25, decay=1.5)
    s_c_405 = base_405 * np.ones_like(t) + noise_amp * rng.standard_normal(len(t))
    s_c_470 = base_470 * (1 + calcium) + noise_amp * rng.standard_normal(len(t))

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.2), sharex=True)
    titles = [
        "motion artifact\n(visible in both)",
        "photobleaching\n(visible in both)",
        "calcium event\n(visible only in 470)",
    ]
    pairs = [(s_a_405, s_a_470), (s_b_405, s_b_470), (s_c_405, s_c_470)]

    for col, ((s_405, s_470), title) in enumerate(zip(pairs, titles)):
        ax_top = axes[0, col]
        ax_top.plot(t, s_405, color=COLOR_CONTROL, linewidth=1.2)
        ax_top.set_ylim(0.43, 0.66)
        ax_top.set_title(title, fontsize=10)
        if col == 0:
            ax_top.set_ylabel("405 nm\n(isosbestic)")

        ax_bot = axes[1, col]
        ax_bot.plot(t, s_470, color=COLOR_SIGNAL, linewidth=1.2)
        ax_bot.set_ylim(0.70, 1.25)
        ax_bot.set_xlabel("time (s)")
        if col == 0:
            ax_bot.set_ylabel("470 nm\n(signal)")

    fig.tight_layout()
    fig.savefig(OUT / "fig4_what_405_captures.svg")
    plt.close(fig)


def figure_5_linear_fit_and_correction():
    """The GuPPy correction in action: signal, control, linear fit, residual dF/F."""
    rng = np.random.default_rng(1)
    sample_rate = 100
    duration = 60
    t = np.arange(0, duration, 1 / sample_rate)

    base_405 = 0.60
    base_470 = 1.00

    bleach = make_bleach(t, tau=25, depth=0.20)
    motion_a = make_motion_bump(t, center=20, width=0.6, depth=0.06)
    motion_b = make_motion_bump(t, center=42, width=0.6, depth=0.05)
    shared = bleach * (1 + motion_a + motion_b)

    calcium_events = (
        make_calcium_transient(t, center=10, amplitude=0.18, rise=0.3, decay=2.0)
        + make_calcium_transient(t, center=30, amplitude=0.14, rise=0.3, decay=2.0)
        + make_calcium_transient(t, center=50, amplitude=0.20, rise=0.3, decay=2.0)
    )

    control = base_405 * shared + 0.004 * rng.standard_normal(len(t))
    signal_trace = base_470 * shared * (1 + calcium_events) + 0.004 * rng.standard_normal(len(t))

    fit = control_fit(control, signal_trace)
    dff = delta_ff(signal_trace, fit)

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 5.4), sharex=True)

    ax = axes[0]
    ax.plot(t, signal_trace, color=COLOR_SIGNAL, linewidth=1.0, label="signal")
    ax.plot(t, control, color=COLOR_CONTROL, linewidth=1.0, label="control")
    ax.set_ylabel("raw fluorescence")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("raw signal and control")

    ax = axes[1]
    ax.plot(t, signal_trace, color=COLOR_SIGNAL, linewidth=1.0, label="signal")
    ax.plot(t, fit, color=COLOR_FIT, linewidth=1.4, linestyle="--", label="control_fit = m * control + b")
    ax.set_ylabel("470 with fit")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("control rescaled to signal")

    ax = axes[2]
    ax.plot(t, dff, color=COLOR_DFF, linewidth=1.0)
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F (%)")
    ax.set_title("corrected dF/F")

    fig.tight_layout()
    fig.savefig(OUT / "fig5_linear_fit_and_correction.svg")
    plt.close(fig)


def figure_6_synthetic_fallback():
    """Compare isosbestic_control=True vs =False on the same recording."""
    rng = np.random.default_rng(2)
    sample_rate = 100
    duration = 60
    t = np.arange(0, duration, 1 / sample_rate)

    base_405 = 0.60
    base_470 = 1.00

    bleach = make_bleach(t, tau=25, depth=0.22)
    # Flip the sign so the motion artifact reads as an upward deflection in
    # dF/F. With the synthetic-exponential fallback, the residual then looks
    # like a calcium event, which is the pedagogical point: the fallback
    # cannot distinguish artifacts from real events when they share polarity.
    motion = -make_motion_bump(t, center=30, width=0.7, depth=0.07)
    shared = bleach * (1 + motion)

    calcium_events = (
        make_calcium_transient(t, center=12, amplitude=0.16, rise=0.3, decay=2.0)
        + make_calcium_transient(t, center=45, amplitude=0.18, rise=0.3, decay=2.0)
    )

    control = base_405 * shared + 0.004 * rng.standard_normal(len(t))
    signal_trace = base_470 * shared * (1 + calcium_events) + 0.004 * rng.standard_normal(len(t))

    fit_real = control_fit(control, signal_trace)
    dff_real = delta_ff(signal_trace, fit_real)

    synthetic = synthetic_exponential_control(signal_trace, t, window=101)
    fit_synthetic = control_fit(synthetic, signal_trace)
    dff_synthetic = delta_ff(signal_trace, fit_synthetic)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.6))

    # Shade the motion-artifact window on the right column so the reader sees
    # which feature was preserved (i.e., not removed by the synthetic fallback).
    motion_t0 = 28.5
    motion_t1 = 31.5
    motion_color = "#f0c060"

    ax = axes[0, 0]
    ax.plot(t, signal_trace, color=COLOR_SIGNAL, linewidth=1.0, label="signal")
    ax.plot(t, control, color=COLOR_CONTROL, linewidth=1.0, label="control")
    ax.set_ylabel("raw fluorescence")
    ax.set_title("real isosbestic control")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.axvspan(motion_t0, motion_t1, color=motion_color, alpha=0.25, zorder=0)
    ax.plot(t, signal_trace, color=COLOR_SIGNAL, linewidth=1.0, label="signal")
    ax.plot(t, synthetic, color=COLOR_SYNTHETIC, linewidth=1.2, linestyle="--", label="synthetic exponential")
    ax.set_title("synthetic exponential control")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, dff_real, color=COLOR_DFF, linewidth=1.0)
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F (%)")
    ax.set_title("artifact removed, calcium events preserved", fontsize=10)

    ax = axes[1, 1]
    ax.axvspan(motion_t0, motion_t1, color=motion_color, alpha=0.25, zorder=0,
               label="motion artifact")
    ax.plot(t, dff_synthetic, color=COLOR_DFF, linewidth=1.0)
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_title("bleaching removed, motion artifact passes through", fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    ymin = min(dff_real.min(), dff_synthetic.min()) - 1
    ymax = max(dff_real.max(), dff_synthetic.max()) + 1
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(OUT / "fig8_synthetic_fallback.svg")
    plt.close(fig)


def figure_7_what_survives():
    """Single asymmetric-artifact case: a wavelength-dependent artifact (large
    at 470, small at 405) cannot be matched by any rescaling of the 405 trace,
    so a reduced version of the artifact remains in dF/F."""
    rng = np.random.default_rng(3)
    sample_rate = 100
    duration = 20
    t = np.arange(0, duration, 1 / sample_rate)

    base_405 = 0.60
    base_470 = 1.00
    bleach = make_bleach(t, tau=40, depth=0.05)
    noise = 0.003

    asym_artifact_470 = make_motion_bump(t, center=10, width=0.4, depth=0.10)
    asym_artifact_405 = make_motion_bump(t, center=10, width=0.4, depth=0.03)
    ctrl = base_405 * bleach * (1 + asym_artifact_405) + noise * rng.standard_normal(len(t))
    sig = base_470 * bleach * (1 + asym_artifact_470) + noise * rng.standard_normal(len(t))

    fit = control_fit(ctrl, sig)
    dff = delta_ff(sig, fit)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.4, 2.4))

    ax_l.plot(t, sig, color=COLOR_SIGNAL, linewidth=1.0, label="signal")
    ax_l.plot(t, ctrl, color=COLOR_CONTROL, linewidth=1.0, label="control")
    ax_l.set_title("raw traces")
    ax_l.legend(loc="center right", frameon=False, fontsize=8)
    ax_l.set_xlabel("time (s)")

    ax_r.plot(t, dff, color=COLOR_DFF, linewidth=1.0)
    ax_r.axhline(0, color="#888888", linewidth=0.5, alpha=0.7)
    ax_r.set_title("corrected dF/F (%)")
    ax_r.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT / "fig6_what_survives.svg")
    plt.close(fig)


def figure_8_control_diagnostic():
    """Two example 405 traces: a clean recording vs a noisy one."""
    rng = np.random.default_rng(4)
    sample_rate = 100
    duration = 60
    t = np.arange(0, duration, 1 / sample_rate)
    base = 0.60

    clean = base * make_bleach(t, tau=30, depth=0.18) + 0.004 * rng.standard_normal(len(t))

    bleach = make_bleach(t, tau=30, depth=0.18)
    motion = (
        make_motion_bump(t, center=12, width=0.5, depth=0.08)
        + make_motion_bump(t, center=27, width=0.4, depth=0.10)
        + make_motion_bump(t, center=45, width=0.7, depth=0.06)
    )
    drift = 0.03 * np.sin(2 * np.pi * t / 35.0)
    noisy = base * bleach * (1 + motion) + base * drift + 0.004 * rng.standard_normal(len(t))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)

    axes[0].plot(t, clean, color=COLOR_CONTROL, linewidth=1.1)
    axes[0].set_title("only photobleaching")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("isosbestic fluorescence")

    axes[1].plot(t, noisy, color=COLOR_CONTROL, linewidth=1.1)
    axes[1].set_title("motion and drift")
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT / "fig7_control_diagnostic.svg")
    plt.close(fig)


if __name__ == "__main__":
    figure_1b_response_columns()
    figure_2a_calcium_chain()
    figure_1_unified_decomposition()
    figure_4_what_405_captures()
    figure_5_linear_fit_and_correction()
    figure_7_what_survives()
    figure_8_control_diagnostic()
    figure_6_synthetic_fallback()
    print("Wrote SVGs to", OUT)
