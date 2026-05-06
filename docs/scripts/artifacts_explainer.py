# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Generate SVG figures for the artifacts explanation page.

The synthetic traces here are designed to make each pedagogical point
clean. They do not mirror specific GuPPy code paths because the artifacts
explainer is about the conceptual structure of contamination categories
and routing logic, not the math of any one correction.

Run with:

    uv run docs/scripts/artifacts_explainer.py

Inline dependencies above let `uv run` resolve everything without needing
the surrounding project to be installed. Outputs are written directly to
docs/_static/images/artifacts_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "artifacts_explainer"
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

COLOR_TRACE = "#222222"
COLOR_REAL = "#2ca02c"
COLOR_ARTIFACT = "#d62728"


def deflection(t, center, rise, decay, peak_amplitude):
    """Alpha-shaped deflection normalized to peak_amplitude."""
    out = np.zeros_like(t)
    mask = t >= center
    dt = t[mask] - center
    raw = (1 - np.exp(-dt / rise)) * np.exp(-dt / decay)
    t_peak_local = rise * decay / (decay - rise) * np.log(decay / rise)
    raw_peak = (1 - np.exp(-t_peak_local / rise)) * np.exp(-t_peak_local / decay)
    out[mask] = peak_amplitude * raw / raw_peak
    return out


def smooth_gaussian(x, sigma_samples):
    """Convolve x with a normalized Gaussian kernel; mirrors what upstream
    smoothing leaves behind in a real photometry trace. Pads edges with the
    boundary value to avoid edge artifacts from zero-padding."""
    kernel_size = int(6 * sigma_samples) | 1
    k = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(k**2) / (2 * sigma_samples**2))
    kernel /= kernel.sum()
    pad_width = kernel_size // 2
    padded = np.pad(x, pad_width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def figure_decomposition():
    """Structural decomposition of an artifact, laid out as an equation:

        recorded   =   artifact 1 + artifact 2 + artifact 3   +   biological signal

    Left big panel: recorded trace. Middle stacked panels: three artifact
    components. Right big panel: clean biological signal. The figure makes
    the additive definition of an artifact visible.
    """
    rng = np.random.default_rng(13)
    sample_rate = 100
    duration = 30.0
    t = np.arange(0, duration, 1 / sample_rate)

    # Biological signal: three calcium-like transients spread across the trace.
    event_times = [5.0, 15.0, 25.0]
    event_amps = [1.5, 1.4, 1.5]
    signal = np.zeros_like(t)
    for c, a in zip(event_times, event_amps):
        signal += deflection(t, center=c, rise=0.20, decay=1.4, peak_amplitude=a)

    # Three artifact components.
    drift = 1.6 * np.exp(-t / 25) - 1.6

    slow_wander = 0.30 * smooth_gaussian(
        rng.standard_normal(len(t)), sigma_samples=int(0.8 * sample_rate)
    )
    bout_envelope = np.zeros_like(t)
    for s, e in [(3.0, 6.5), (13.0, 17.0), (22.0, 25.5)]:
        bout_envelope[(t >= s) & (t <= e)] = 1.0
    bout_envelope = smooth_gaussian(bout_envelope, sigma_samples=int(0.4 * sample_rate))
    fast_wander = 0.9 * smooth_gaussian(
        rng.standard_normal(len(t)), sigma_samples=int(0.12 * sample_rate)
    )
    motion = slow_wander + fast_wander * bout_envelope

    electronic_noise = 0.22 * rng.standard_normal(len(t))

    recorded = signal + drift + motion + electronic_noise

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(3, 13, wspace=0.4, hspace=0.55)

    # Left: recorded trace (big panel spanning all 3 rows).
    ax_rec = fig.add_subplot(gs[:, 0:4])
    ax_rec.plot(t, recorded, color=COLOR_TRACE, linewidth=1.0)
    ax_rec.axhline(0, color="#bbbbbb", linewidth=0.5, zorder=0)
    ax_rec.set_xlabel("time (s)")
    ax_rec.set_ylabel("fluorescence (a.u.)")
    ax_rec.set_yticks([])
    ax_rec.set_xlim(0, duration)

    # Middle: three stacked artifact panels.
    artifacts = [
        (drift, "Photobleaching / LED drift"),
        (motion, "Motion"),
        (electronic_noise, "Electronic noise"),
    ]
    for i, (data, label) in enumerate(artifacts):
        ax = fig.add_subplot(gs[i, 5:9])
        ax.plot(t, data, color=COLOR_ARTIFACT, linewidth=0.8)
        ax.axhline(0, color="#bbbbbb", linewidth=0.4, zorder=0)
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0, duration)
        if i < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel("time (s)")

    # Right: biological signal (big panel spanning all 3 rows).
    ax_sig = fig.add_subplot(gs[:, 10:13])
    ax_sig.plot(t, signal, color=COLOR_REAL, linewidth=1.0)
    ax_sig.axhline(0, color="#bbbbbb", linewidth=0.5, zorder=0)
    ax_sig.set_xlabel("time (s)")
    ax_sig.set_yticks([])
    ax_sig.set_xlim(0, duration)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.13)

    # Bold column headers, each centered over its column.
    def _column_center(ax_or_axes):
        if isinstance(ax_or_axes, list):
            x0 = min(a.get_position().x0 for a in ax_or_axes)
            x1 = max(a.get_position().x1 for a in ax_or_axes)
        else:
            box = ax_or_axes.get_position()
            x0, x1 = box.x0, box.x1
        return (x0 + x1) / 2

    header_y = 0.93
    middle_axes = [fig.axes[i] for i in range(1, 4)]
    fig.text(
        _column_center(ax_rec), header_y,
        "Recorded signal", fontsize=12, fontweight="bold",
        ha="center", va="bottom", color="#222222",
    )
    fig.text(
        _column_center(middle_axes), header_y,
        "Artifacts", fontsize=12, fontweight="bold",
        ha="center", va="bottom", color="#222222",
    )
    fig.text(
        _column_center(ax_sig), header_y,
        "True signal", fontsize=12, fontweight="bold",
        ha="center", va="bottom", color="#222222",
    )

    # Operators centered mathematically in the gap between adjacent columns.
    rec_right = ax_rec.get_position().x1
    middle_left = min(a.get_position().x0 for a in middle_axes)
    middle_right = max(a.get_position().x1 for a in middle_axes)
    sig_left = ax_sig.get_position().x0
    operator_y = (ax_rec.get_position().y0 + ax_rec.get_position().y1) / 2
    fig.text(
        (rec_right + middle_left) / 2, operator_y,
        "=", fontsize=22, ha="center", va="center", color="#444444",
    )
    fig.text(
        (middle_right + sig_left) / 2, operator_y,
        "+", fontsize=22, ha="center", va="center", color="#444444",
    )

    fig.savefig(OUT / "fig1_decomposition.svg")
    plt.close(fig)


def figure_decision_tree():
    """Vertical decision tree routing an artifact to the correction method
    that catches it. Branches on intrinsic properties: off-band frequency
    content -> low-pass filter; wavelength-independent -> isosbestic
    correction; episodic time structure -> manual removal; default ->
    active-research territory (spectral methods or tolerate). Method leaves
    use the pipeline-style "X caught: ..." framing to make the catching
    semantics explicit.
    """
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    decision_x = 28
    method_x = 76
    decision_w, decision_h = 30, 10
    method_w, method_h = 36, 13

    decisions = [
        ("Off-band\nfrequency content?", 82),
        ("Wavelength-\nindependent?", 62),
        ("Episodic?", 42),
    ]
    method_yes = [
        ("Off-band caught:\nlow-pass filter", 82),
        ("Independent caught:\nisosbestic correction", 62),
        ("Episodic caught:\nmanual removal", 42),
    ]
    final_method_label = "Active research:\nspectral methods or tolerate"
    final_method_y = 18

    # Decision boxes (left column).
    for label, y in decisions:
        box = FancyBboxPatch(
            (decision_x - decision_w / 2, y - decision_h / 2),
            decision_w, decision_h,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            facecolor="#f4f4f4", edgecolor="#666666", linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(
            decision_x, y, label, ha="center", va="center",
            fontsize=10, color="#222222",
        )

    # YES method leaves (right column, with "X caught: ..." framing).
    for label, y in method_yes:
        box = FancyBboxPatch(
            (method_x - method_w / 2, y - method_h / 2),
            method_w, method_h,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            facecolor="#e8f4e8", edgecolor="#1d6f1d", linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(
            method_x, y, label, ha="center", va="center",
            fontsize=9, color="#222222",
        )

    # Default method at the bottom of the decision column (active research).
    box = FancyBboxPatch(
        (decision_x - decision_w / 2, final_method_y - decision_h / 2),
        decision_w, decision_h,
        boxstyle="round,pad=0.4,rounding_size=1.5",
        facecolor="#e8f4e8", edgecolor="#1d6f1d", linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(
        decision_x, final_method_y, final_method_label,
        ha="center", va="center", fontsize=9, color="#222222",
    )

    # "Artifact" start label and arrow into the first decision.
    ax.text(
        decision_x, 95, "Artifact", ha="center", va="center",
        fontsize=12, fontweight="bold", color="#222222",
    )
    ax.add_patch(FancyArrowPatch(
        (decision_x, 92), (decision_x, decisions[0][1] + decision_h / 2),
        arrowstyle="-|>", mutation_scale=14,
        color="#444444", linewidth=1.2,
    ))

    # YES arrows: decision right to method (green).
    for (_, dy), (_, my) in zip(decisions, method_yes):
        ax.add_patch(FancyArrowPatch(
            (decision_x + decision_w / 2, dy),
            (method_x - method_w / 2, my),
            arrowstyle="-|>", mutation_scale=14,
            color="#1d6f1d", linewidth=1.3,
        ))
        mid_x = (decision_x + decision_w / 2 + method_x - method_w / 2) / 2
        ax.text(
            mid_x, dy + 1.2, "yes", ha="center", va="bottom",
            fontsize=9, color="#1d6f1d", fontweight="bold", style="italic",
        )

    # NO arrows: decision down to next decision (or to the default method).
    no_targets = [
        (decisions[0][1], decisions[1][1]),
        (decisions[1][1], decisions[2][1]),
        (decisions[2][1], final_method_y),
    ]
    for y_top, y_bot in no_targets:
        ax.add_patch(FancyArrowPatch(
            (decision_x, y_top - decision_h / 2),
            (decision_x, y_bot + decision_h / 2),
            arrowstyle="-|>", mutation_scale=14,
            color="#888888", linewidth=1.3,
        ))
        mid_y = ((y_top - decision_h / 2) + (y_bot + decision_h / 2)) / 2
        ax.text(
            decision_x + 1.6, mid_y, "no", ha="left", va="center",
            fontsize=9, color="#888888", fontweight="bold", style="italic",
        )

    fig.savefig(OUT / "fig2_decision_tree.svg")
    plt.close(fig)


if __name__ == "__main__":
    figure_decomposition()
    figure_decision_tree()
