# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Generate SVG figures for the PSTH explanation page.

The PSTH operation (window extraction, across-event averaging, SEM band)
mirrors src/guppy/analysis/compute_psth.py. Only the input traces are
synthetic and designed to make each pedagogical point clean.

Run with:

    uv run docs/scripts/psth_explainer.py

Inline dependencies above let `uv run` resolve everything without needing
the surrounding project to be installed. Outputs are written directly to
docs/_static/images/psth_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "psth_explainer"
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

COLOR_TRACE = "#1f4e79"
COLOR_TRIAL = "#a8a8a8"
COLOR_MEAN = "#c0392b"
COLOR_EVENT = "#2c7a3a"
COLOR_WINDOW = "#cfe2f3"


def figure_psth_walkthrough():
    """Four-row walk-through of the PSTH operation:

    1. Full z-score trace with event markers.
    2. Same trace with per-event extraction windows highlighted.
    3. Extracted windows overlaid on a shared event-relative time axis.
    4. Across-event average with SEM band.
    """
    rng = np.random.default_rng(11)
    sample_rate = 30
    duration = 220.0
    t = np.arange(0, duration, 1 / sample_rate)

    pre_seconds = 5.0
    post_seconds = 10.0

    base_event_times = np.arange(15.0, duration - 20.0, 20.0)
    jitter = rng.uniform(-2.0, 2.0, len(base_event_times))
    event_times = base_event_times + jitter

    n_events = len(event_times)
    event_amps = rng.uniform(2.5, 6.0, n_events)
    event_widths = rng.uniform(0.45, 1.25, n_events)
    event_latencies = rng.uniform(-0.4, 0.6, n_events)
    event_undershoot_factors = rng.uniform(-0.9, -0.15, n_events)
    event_undershoot_delays = rng.uniform(1.4, 2.6, n_events)
    event_undershoot_widths = rng.uniform(1.1, 1.9, n_events)

    pure_noise_event = 4
    artifact_event = 7
    event_amps[pure_noise_event] = 0.0
    event_undershoot_factors[pure_noise_event] = 0.0

    def event_response_at(et, amp, width, latency, under_factor, under_delay, under_width):
        center = et + latency
        peak = amp * np.exp(-((t - center) ** 2) / (2 * width**2))
        undershoot = under_factor * amp * np.exp(
            -((t - center - under_delay) ** 2) / (2 * under_width**2)
        )
        return peak + undershoot

    z = sum(
        event_response_at(et, amp, w, lat, uf, ud, uw)
        for et, amp, w, lat, uf, ud, uw in zip(
            event_times,
            event_amps,
            event_widths,
            event_latencies,
            event_undershoot_factors,
            event_undershoot_delays,
            event_undershoot_widths,
        )
    )

    artifact_center = event_times[artifact_event] + 4.5
    artifact_amp = 7.0
    artifact_width = 0.35
    z = z + artifact_amp * np.exp(-((t - artifact_center) ** 2) / (2 * artifact_width**2))

    n_extra = 20
    rng_extra = np.random.default_rng(13)
    extra_times = []
    while len(extra_times) < n_extra:
        candidate = rng_extra.uniform(0.0, duration)
        if all(abs(candidate - et) > 4.0 for et in event_times):
            extra_times.append(candidate)
    extra_amps = rng_extra.uniform(-0.6, 1.2, n_extra)
    extra_widths = rng_extra.uniform(0.4, 1.2, n_extra)
    for et, amp, width in zip(extra_times, extra_amps, extra_widths):
        z = z + amp * np.exp(-((t - et) ** 2) / (2 * width**2))

    z = z + 0.3 * rng.standard_normal(len(t))

    n_pre = int(round(pre_seconds * sample_rate))
    n_post = int(round(post_seconds * sample_rate))
    t_psth = (np.arange(-n_pre, n_post)) / sample_rate

    traces = []
    for et in event_times:
        event_idx = int(round(et * sample_rate))
        window = z[event_idx - n_pre : event_idx + n_post]
        if len(window) == n_pre + n_post:
            traces.append(window)
    traces = np.asarray(traces)

    mean_psth = traces.mean(axis=0)
    sem_psth = traces.std(axis=0) / np.sqrt(traces.shape[0])

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.5))
    ax_full, ax_windows, ax_overlay, ax_avg = axes

    z_min, z_max = z.min() - 0.5, z.max() + 0.5

    color_pure_noise = "#e08e3c"
    color_artifact = "#8e44ad"

    ax_full.plot(t, z, color=COLOR_TRACE, linewidth=0.8)
    for et in event_times:
        ax_full.axvline(et, color=COLOR_EVENT, linewidth=1.0, alpha=0.8)
    ax_full.set_xlim(t[0], t[-1])
    ax_full.set_ylim(z_min, z_max)
    ax_full.set_ylabel("z-score", fontsize=9)
    ax_full.set_title(
        "1. full session z-score trace, with event timestamps as vertical lines",
        loc="left",
        fontsize=10,
    )

    ax_windows.plot(t, z, color=COLOR_TRACE, linewidth=0.8)
    for event_index, et in enumerate(event_times):
        if event_index == pure_noise_event:
            window_color = color_pure_noise
            window_alpha = 0.45
        elif event_index == artifact_event:
            window_color = color_artifact
            window_alpha = 0.35
        else:
            window_color = COLOR_WINDOW
            window_alpha = 0.55
        ax_windows.axvspan(
            et - pre_seconds,
            et + post_seconds,
            color=window_color,
            alpha=window_alpha,
            linewidth=0,
        )
        ax_windows.axvline(et, color=COLOR_EVENT, linewidth=0.9, alpha=0.7)
    ax_windows.set_xlim(t[0], t[-1])
    ax_windows.set_ylim(z_min, z_max)
    ax_windows.set_ylabel("z-score", fontsize=9)
    ax_windows.set_xlabel("time (s)", fontsize=9)
    ax_windows.set_title(
        f"2. extract a window per event ({pre_seconds:.0f} s before to {post_seconds:.0f} s after); "
        "the orange and purple windows mark a pure-noise event and an artifact event",
        loc="left",
        fontsize=10,
    )

    for event_index, trace in enumerate(traces):
        if event_index == pure_noise_event:
            ax_overlay.plot(
                t_psth,
                trace,
                color=color_pure_noise,
                linewidth=1.4,
                alpha=0.95,
                label="pure-noise event",
                zorder=4,
            )
        elif event_index == artifact_event:
            ax_overlay.plot(
                t_psth,
                trace,
                color=color_artifact,
                linewidth=1.4,
                alpha=0.95,
                label="artifact event",
                zorder=4,
            )
        else:
            ax_overlay.plot(
                t_psth, trace, color=COLOR_TRIAL, linewidth=0.7, alpha=0.6, zorder=2
            )
    ax_overlay.axvline(0, color=COLOR_EVENT, linewidth=1.0, alpha=0.8)
    ax_overlay.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_overlay.set_xlim(t_psth[0], t_psth[-1])
    ax_overlay.set_ylabel("z-score", fontsize=9)
    ax_overlay.set_xlabel("time from event (s)", fontsize=9)
    ax_overlay.legend(loc="upper right", frameon=False, fontsize=8)
    ax_overlay.set_title(
        f"3. stack the windows on a shared event-relative time axis (n = {traces.shape[0]} events)",
        loc="left",
        fontsize=10,
    )

    ax_avg.fill_between(
        t_psth,
        mean_psth - sem_psth,
        mean_psth + sem_psth,
        color=COLOR_MEAN,
        alpha=0.25,
        linewidth=0,
    )
    ax_avg.plot(t_psth, mean_psth, color=COLOR_MEAN, linewidth=1.6)
    ax_avg.axvline(0, color=COLOR_EVENT, linewidth=1.0, alpha=0.8)
    ax_avg.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_avg.set_xlim(t_psth[0], t_psth[-1])
    ax_avg.set_ylabel("z-score", fontsize=9)
    ax_avg.set_xlabel("time from event (s)", fontsize=9)
    ax_avg.set_title(
        "4. average across events (mean +/- SEM); the event-locked response is what survives",
        loc="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig1_psth_walkthrough.svg")
    plt.close(fig)


def figure_session_drift():
    """Three-panel figure showing how slow session drift creates per-event
    baseline differences that the session-wide z-score does not fix, and how
    per-event baseline correction resolves it.

    Top: a session with ten events spread across its duration; three of them
    (at start, middle, end) are highlighted in distinct colors, the other
    seven appear with pale gray windows so the session looks naturally busy
    rather than artificially sparse.
    Middle: the three highlighted event-aligned traces extracted on a shared
    event-relative axis. Their pre-event baselines sit at the local drift level.
    Bottom: the same three traces after per-event pre-event mean subtraction;
    every trace now starts at zero pre-event with the event response itself
    unchanged.
    """
    rng = np.random.default_rng(21)
    sample_rate = 30
    duration = 600.0
    t = np.arange(0, duration, 1 / sample_rate)

    pre_seconds = 5.0
    post_seconds = 10.0
    n_pre = int(round(pre_seconds * sample_rate))
    n_post = int(round(post_seconds * sample_rate))
    t_psth = (np.arange(-n_pre, n_post)) / sample_rate

    event_times = np.linspace(60.0, 540.0, 10) + rng.uniform(-12, 12, 10)
    highlighted_indices = [0, 4, 9]

    drift = 1.5 * (1 - 2 * t / duration)

    event_amp = 4.5
    event_width = 0.7

    def event_response_at(et):
        peak = event_amp * np.exp(-((t - et) ** 2) / (2 * event_width**2))
        undershoot = -0.5 * event_amp * np.exp(
            -((t - et - 2.0) ** 2) / (2 * 1.5**2)
        )
        return peak + undershoot

    z = drift + sum(event_response_at(et) for et in event_times)
    z = z + 0.25 * rng.standard_normal(len(t))

    traces = []
    for et in event_times:
        event_idx = int(round(et * sample_rate))
        window = z[event_idx - n_pre : event_idx + n_post]
        traces.append(window)

    highlight_colors = {0: "#1f77b4", 4: "#d62728", 9: "#2ca02c"}
    highlight_labels = {
        0: "start of session",
        4: "middle of session",
        9: "end of session",
    }
    pale_color = "#bbbbbb"

    fig, (ax_full, ax_traces, ax_corrected) = plt.subplots(3, 1, figsize=(9.0, 8.5))

    ax_full.plot(t, z, color="#3a3a3a", linewidth=0.7, alpha=0.9)
    for index, et in enumerate(event_times):
        if index in highlight_colors:
            ax_full.axvspan(
                et - pre_seconds,
                et + post_seconds,
                color=highlight_colors[index],
                alpha=0.3,
                linewidth=0,
            )
            ax_full.axvline(et, color=highlight_colors[index], linewidth=1.0, alpha=0.85)
        else:
            ax_full.axvspan(
                et - pre_seconds,
                et + post_seconds,
                color=pale_color,
                alpha=0.3,
                linewidth=0,
            )
    ax_full.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_full.set_xlim(t[0], t[-1])
    ax_full.set_xlabel("time (s)", fontsize=9)
    ax_full.set_ylabel("z-score", fontsize=9)
    ax_full.set_title(
        "full session z-score trace: ten events, three highlighted (start, middle, end); slow drift carries the baseline across the session",
        loc="left",
        fontsize=10,
    )

    baseline_color = "#ffe699"
    ax_traces.axvspan(
        -pre_seconds, 0.0, color=baseline_color, alpha=0.55, linewidth=0
    )
    for index in highlighted_indices:
        ax_traces.plot(
            t_psth,
            traces[index],
            color=highlight_colors[index],
            linewidth=1.5,
            alpha=0.95,
            label=highlight_labels[index],
        )
    ax_traces.axvline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_traces.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_traces.set_xlim(t_psth[0], t_psth[-1])
    ax_traces.set_xlabel("time from event (s)", fontsize=9)
    ax_traces.set_ylabel("z-score", fontsize=9)
    ax_traces.legend(loc="upper right", frameon=False, fontsize=9)
    ax_traces.set_title(
        "extracted event-aligned traces: pre-event baselines (shaded) sit at whatever the local drift level happened to be",
        loc="left",
        fontsize=10,
    )

    ax_traces.text(
        0.02,
        0.04,
        "Each event-aligned trace's mean over this window\nis subtracted to anchor the trace at zero pre-event",
        transform=ax_traces.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    for index in highlighted_indices:
        trace = traces[index]
        pre_event_mean = trace[:n_pre].mean()
        corrected = trace - pre_event_mean
        ax_corrected.plot(
            t_psth,
            corrected,
            color=highlight_colors[index],
            linewidth=1.5,
            alpha=0.95,
            label=highlight_labels[index],
        )
    ax_corrected.axvline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_corrected.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax_corrected.set_xlim(t_psth[0], t_psth[-1])
    ax_corrected.set_xlabel("time from event (s)", fontsize=9)
    ax_corrected.set_ylabel("z-score (baseline-corrected)", fontsize=9)
    ax_corrected.set_title(
        "after per-event baseline correction: every trace sits at zero pre-event, the response itself is unchanged",
        loc="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig2_session_drift.svg")
    plt.close(fig)


def figure_peak_vs_auc():
    """3x2 grid: same three response shapes shown twice, with peak summary
    on the left column and AUC summary on the right column.

    Row 1 (canonical clean peak): peak and AUC agree.
    Row 2 (broad sustained response): peak is small but AUC captures the
        sustained magnitude that peak misses.
    Row 3 (real response plus brief artifact spike): peak latches onto the
        artifact while AUC stays close to the real response.
    """
    sample_rate = 30
    pre_seconds = 2.0
    post_seconds = 7.0
    n_pre = int(round(pre_seconds * sample_rate))
    n_post = int(round(post_seconds * sample_rate))
    t = (np.arange(-n_pre, n_post)) / sample_rate

    psth_a = 3.0 * np.exp(-((t - 1.0) ** 2) / (2 * 0.6**2))

    psth_b = 1.2 * np.exp(-((t - 2.8) ** 2) / (2 * 1.8**2))

    real_response = 1.0 * np.exp(-((t - 1.4) ** 2) / (2 * 0.5**2))
    artifact_spike = 4.5 * np.exp(-((t - 3.8) ** 2) / (2 * 0.08**2))
    psth_c = real_response + artifact_spike

    psths = [psth_a, psth_b, psth_c]

    color_trace = "#1f77b4"
    color_peak = "#c0392b"
    color_auc = "#2ca02c"

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 8.5))

    window_mask = t >= 0
    t_window = t[window_mask]

    for row_idx, psth in enumerate(psths):
        ax_peak = axes[row_idx, 0]
        ax_auc = axes[row_idx, 1]

        psth_window = psth[window_mask]
        peak_idx = np.argmax(psth_window)
        peak_t = t_window[peak_idx]
        peak_value = psth_window[peak_idx]
        auc_value = np.trapezoid(psth_window, t_window)

        ax_peak.plot(t, psth, color=color_trace, linewidth=1.3, alpha=0.95)
        ax_peak.plot(
            [peak_t, peak_t],
            [0, peak_value],
            color=color_peak,
            linewidth=1.0,
            linestyle=":",
            alpha=0.75,
        )
        ax_peak.plot(peak_t, peak_value, "o", color=color_peak, markersize=10, zorder=5)
        ax_peak.axvline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_peak.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
        ax_peak.text(
            0.97,
            0.93,
            f"peak = {peak_value:.1f}",
            transform=ax_peak.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=color_peak,
        )

        ax_auc.fill_between(
            t_window, 0, psth_window, color=color_auc, alpha=0.32, linewidth=0
        )
        ax_auc.plot(t, psth, color=color_trace, linewidth=1.3, alpha=0.95)
        ax_auc.axvline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_auc.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
        ax_auc.text(
            0.97,
            0.93,
            f"AUC = {auc_value:.1f}",
            transform=ax_auc.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=color_auc,
        )

        psth_max = max(psth.max(), 0.5)
        for ax in (ax_peak, ax_auc):
            ax.set_ylim(-0.4, psth_max + 0.6)
            ax.set_xlim(t[0], t[-1])

        ax_peak.set_ylabel("z-score", fontsize=9)
        plt.setp(ax_auc.get_yticklabels(), visible=False)

    axes[0, 0].set_title("peak amplitude", loc="left", fontsize=11)
    axes[0, 1].set_title("AUC (area under the curve)", loc="left", fontsize=11)

    for ax in axes[-1, :]:
        ax.set_xlabel("time from event (s)", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_peak_vs_auc.svg")
    plt.close(fig)


def figure_event_rejection():
    """Two-panel figure illustrating the two event-rejection filters with
    minimal visual elements: red marks rejected events, green marks accepted
    events, in both panels.

    Left: edge rejection - one event near the recording start whose pre-event
    extraction window would clip past the recording start, and one accepted
    event later in the recording.
    Right: burst rejection - three events too close together; the first is
    kept (green) and the next two (red) fall within the inter-event threshold
    and are rejected.
    """
    rng = np.random.default_rng(31)
    sample_rate = 30
    duration = 30.0
    t = np.arange(0, duration, 1 / sample_rate)

    pre_seconds = 5.0
    post_seconds = 10.0

    color_kept = "#2ca02c"
    color_rejected = "#c0392b"

    fig, (ax_edge, ax_burst) = plt.subplots(1, 2, figsize=(12.5, 4.0))

    edge_event_t = 2.0
    accepted_event_t = 17.0
    z_edge = 0.35 * rng.standard_normal(len(t))
    z_edge = z_edge + 3.5 * np.exp(-((t - edge_event_t) ** 2) / (2 * 0.7**2))
    z_edge = z_edge + 3.5 * np.exp(-((t - accepted_event_t) ** 2) / (2 * 0.7**2))

    ax_edge.axvspan(-10, 0, facecolor="#dcdcdc", alpha=0.7, linewidth=0, zorder=0)
    ax_edge.text(
        -3.5,
        z_edge.max() * 0.55,
        "window out\nof recording",
        ha="center",
        va="center",
        fontsize=8,
        color="#777777",
        fontstyle="italic",
        zorder=1,
    )
    ax_edge.plot(t, z_edge, color="#3a3a3a", linewidth=0.7, alpha=0.85, zorder=2)
    ax_edge.axvline(edge_event_t, color=color_rejected, linewidth=1.3, alpha=0.9, zorder=3)
    ax_edge.axvline(accepted_event_t, color=color_kept, linewidth=1.3, alpha=0.9, zorder=3)
    ax_edge.axvline(0, color="#000000", linewidth=1.4, alpha=0.75, zorder=3)
    ax_edge.axvline(duration, color="#000000", linewidth=1.4, alpha=0.75, zorder=3)
    ax_edge.set_xlim(-7, duration + 7)
    ax_edge.set_ylim(z_edge.min() - 0.5, z_edge.max() + 1.0)
    ax_edge.set_xlabel("time (s)", fontsize=9)
    ax_edge.set_ylabel("z-score", fontsize=9)
    ax_edge.set_title(
        "edge rejection: extraction window clips past recording bounds",
        loc="left",
        fontsize=10,
    )
    ax_edge.legend(
        handles=[
            plt.Line2D([0], [0], color=color_rejected, linewidth=2, label="rejected event"),
            plt.Line2D([0], [0], color=color_kept, linewidth=2, label="accepted event"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )

    burst_events = [(10.0, "kept"), (12.5, "burst"), (15.0, "burst")]
    z_burst = 0.35 * rng.standard_normal(len(t))
    for et, _ in burst_events:
        z_burst = z_burst + 3.5 * np.exp(-((t - et) ** 2) / (2 * 0.7**2))

    ax_burst.plot(t, z_burst, color="#3a3a3a", linewidth=0.7, alpha=0.85, zorder=2)
    for et, category in burst_events:
        if category == "kept":
            ax_burst.axvline(et, color=color_kept, linewidth=1.3, alpha=0.9, zorder=3)
        else:
            ax_burst.axvline(et, color=color_rejected, linewidth=1.3, alpha=0.9, zorder=3)
    ax_burst.set_xlim(-7, duration + 7)
    ax_burst.set_ylim(z_burst.min() - 0.5, z_burst.max() + 1.0)
    ax_burst.set_xlabel("time (s)", fontsize=9)
    ax_burst.set_ylabel("z-score", fontsize=9)
    ax_burst.set_title(
        "burst rejection: events too close to a previous kept event",
        loc="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig4_event_rejection.svg")
    plt.close(fig)


if __name__ == "__main__":
    figure_psth_walkthrough()
    figure_session_drift()
    figure_peak_vs_auc()
    figure_event_rejection()
    print("Wrote SVGs to", OUT)
