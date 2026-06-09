# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Generate SVG figures for the transient detection explanation page.

Figures are pedagogical synthetic traces, not real GuPPy outputs. The detector
behaviour they illustrate mirrors the algorithm in src/guppy/analysis/transients.py:
per-chunk two-stage MAD with a one-sided outlier trim, then local-maxima above
the second-stage threshold.

Run with:

    uv run docs/scripts/transient_detection_explainer.py

Inline dependencies above let `uv run` resolve everything without needing the
surrounding project to be installed. Outputs are written directly to
docs/_static/images/transient_detection_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "transient_detection_explainer"
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

COLOR_TRACE = "#2ca02c"
COLOR_PEAK = "#d62728"
COLOR_THRESHOLD = "#888888"
COLOR_FILTER = "#cccccc"
COLOR_ACCEPTED = "#d62728"  # red: the algorithm flagged this (detected/accepted)
COLOR_REJECTED = "#1f4e79"  # dark navy: real event the algorithm missed or rejected
COLOR_CHUNK = "#bbbbbb"
COLOR_NOISE_SHADE = "#888888"
MARKER_SIZE_ACCEPTED = 38
MARKER_SIZE_REJECTED = 60  # slightly larger so the missed events stay visible
MARKER_LINEWIDTH_FP = 1.3  # outline width for hollow false-positive markers


def _draw_chunk_bracket(ax, x_start, x_end, label, y_frac=0.05, panel_width=None):
    """Draw a labelled horizontal bracket inside the axes marking one chunk's span.

    Default position is at the bottom of the panel (y_frac=0.05), with end ticks
    pointing up. x is in data coordinates, y in axes-fraction so the bracket sits
    at a fixed height regardless of the trace's y-range. The caller must extend
    the panel's y_min to leave clearance below the trace, otherwise the bracket
    overlaps with low-baseline stretches.

    Label placement adapts to bracket width: a wide bracket (>= 70% of the panel)
    anchors the label at the right end inside the bracket; a short bracket places
    the label to the right of the bracket so it does not collide with the y-axis.
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    color = "#555555"
    lw = 0.9
    tick_h = 0.025
    ax.plot([x_start, x_end], [y_frac, y_frac], color=color, linewidth=lw,
            transform=trans, clip_on=False, zorder=5)
    ax.plot([x_start, x_start], [y_frac, y_frac + tick_h], color=color, linewidth=lw,
            transform=trans, clip_on=False, zorder=5)
    ax.plot([x_end, x_end], [y_frac, y_frac + tick_h], color=color, linewidth=lw,
            transform=trans, clip_on=False, zorder=5)
    if panel_width is None:
        panel_xmin, panel_xmax = ax.get_xlim()
        panel_width = panel_xmax - panel_xmin
    if (x_end - x_start) > 0.7 * panel_width:
        label_x = x_end
        label_ha = "right"
    else:
        label_x = x_end + 0.01 * panel_width
        label_ha = "left"
    ax.text(label_x, y_frac + 0.04, label, transform=trans,
            ha=label_ha, va="bottom", fontsize=8, color=color, zorder=5)


def _draw_panel_label(ax, label, outside=False):
    """Place a bold panel label (A, B, C, ...) at the top-left of the axes.

    By default the label sits just inside the axes (y=1.0, va="top") so it never
    collides with the panel title. With outside=True it sits just above the axes
    (y=1.02, va="bottom"), useful when content near the axes top edge (e.g. a high
    detection threshold) would otherwise sit very close to the inside-label.
    """
    if outside:
        ax.text(0.02, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left", color="#222", zorder=5)
    else:
        ax.text(0.02, 1.0, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left", color="#222", zorder=5)


def make_calcium_transient(t, center, amplitude, rise=0.25, decay=1.5):
    """A simple alpha-like calcium transient: fast rise, slow exponential decay."""
    out = np.zeros_like(t)
    after = t >= center
    dt = t[after] - center
    out[after] = amplitude * (1 - np.exp(-dt / rise)) * np.exp(-dt / decay)
    return out


def _make_drifting_trace_with_events(seed=11, duration=60.0, sample_rate=100):
    """Synthesise a drifting trace with eight identical-amplitude calcium events.
    Returns (t, trace, event_times, event_amps, sample_rate).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / sample_rate)
    drift = 1.6 * np.sin(2 * np.pi * t / 50) + 0.9 * np.sin(2 * np.pi * t / 18)
    event_times = np.array([4.0, 11.0, 19.0, 27.0, 35.0, 43.0, 51.0, 57.0])
    event_amps = np.full(len(event_times), 2.5)
    events = np.zeros_like(t)
    for c, a in zip(event_times, event_amps):
        events += make_calcium_transient(t, center=c, amplitude=a, rise=0.25, decay=1.5)
    noise = 0.18 * rng.standard_normal(len(t))
    trace = drift + events + noise
    return t, trace, event_times, event_amps, sample_rate


def _classify_detections(detected_idx, event_times, t, tolerance_s=0.6):
    """Classify each detection as TP or FP and each ground-truth event as caught or missed.

    Returns (tp_idx, fp_idx, fn_event_indices) — three sets.
    """
    detected_times = t[detected_idx]
    matched_event = {}
    matched_detection = set()
    for di, dt in zip(detected_idx, detected_times):
        for ei, et in enumerate(event_times):
            if ei in matched_event:
                continue
            if abs(dt - et) <= tolerance_s:
                matched_event[ei] = di
                matched_detection.add(int(di))
                break
    tp_idx = matched_detection
    fp_idx = {int(i) for i in detected_idx} - matched_detection
    fn_event_indices = set(range(len(event_times))) - set(matched_event.keys())
    return tp_idx, fp_idx, fn_event_indices


def figure_1_transient_extraction():
    """Continuous trace becomes a discrete event list."""
    rng = np.random.default_rng(7)
    sample_rate = 100
    duration = 60
    t = np.arange(0, duration, 1 / sample_rate)

    event_times = np.array([4.5, 6.2, 7.8, 13.0, 19.4, 25.1, 28.3, 33.6, 39.0, 44.5, 50.2, 55.8])
    event_amps = np.array([2.3, 3.1, 1.8, 2.6, 4.0, 1.5, 2.2, 3.4, 2.0, 2.8, 3.6, 1.7])

    trace = np.zeros_like(t)
    for c, a in zip(event_times, event_amps):
        trace += make_calcium_transient(t, center=c, amplitude=a, rise=0.25, decay=1.6)

    baseline_drift = 0.4 * np.sin(2 * np.pi * t / 35) + 0.2 * np.sin(2 * np.pi * t / 11)
    noise = 0.09 * rng.standard_normal(len(t))
    trace = trace + baseline_drift + noise

    # Find the actual local maximum of the trace inside a small window after each event
    # onset, rather than a fixed-offset approximation. Overlapping events and drift can
    # shift the real maximum away from any single event's intrinsic kinetic peak.
    search_window = 1.0  # seconds after the event onset to look for the trace maximum
    window_samples = int(search_window * sample_rate)
    peak_indices = []
    for c in event_times:
        start = int(np.argmin(np.abs(t - c)))
        end = min(start + window_samples, len(t))
        peak_indices.append(start + int(np.argmax(trace[start:end])))
    peak_indices = np.array(peak_indices)
    peak_t = t[peak_indices]
    peak_y = trace[peak_indices]

    fig, (ax_trace, ax_raster) = plt.subplots(
        2, 1,
        figsize=(8.6, 3.4),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.15},
    )

    y_min = float(max(np.min(trace) - 0.3, -0.5))
    y_max = float(np.max(trace) + 0.4)

    ax_trace.plot(t, trace, color=COLOR_TRACE, linewidth=1.0)
    for pt, py in zip(peak_t, peak_y):
        ax_trace.plot([pt, pt], [y_min, py], color=COLOR_PEAK, linewidth=1.0,
                       linestyle="--", alpha=0.35, zorder=2)
    ax_trace.plot(peak_t, peak_y, "o", color=COLOR_PEAK, markersize=5, markeredgewidth=0, zorder=3)
    ax_trace.set_ylabel("z-score")
    ax_trace.set_title("continuous trace, with detected events", loc="center")
    ax_trace.set_xlim(0, duration)
    ax_trace.set_ylim(y_min, y_max)

    ax_raster.eventplot(
        peak_t,
        orientation="horizontal",
        colors=COLOR_PEAK,
        lineoffsets=0.5,
        linelengths=0.7,
        linewidths=1.4,
    )
    ax_raster.set_yticks([])
    ax_raster.set_xlabel("time (s)")
    ax_raster.set_xlim(0, duration)
    ax_raster.set_ylim(0, 1)
    ax_raster.set_ylabel("event list", rotation=0, ha="right", va="center", labelpad=8)
    ax_raster.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_transient_extraction.svg", bbox_inches="tight")
    plt.close(fig)


def figure_2_drift_failure_and_fix():
    """Top: trace with drift, fixed session-wide threshold. Bottom: same trace with chunk-local
    threshold drawn as a step function. The bottom panel shows the chunked fix in the
    per-chunk view; the top panel is the failure mode of a single session-wide statistic.
    """
    rng = np.random.default_rng(11)
    sample_rate = 100
    duration = 60
    t = np.arange(0, duration, 1 / sample_rate)

    drift = 1.6 * np.sin(2 * np.pi * t / 50) + 0.9 * np.sin(2 * np.pi * t / 18)

    event_times = np.array([4.0, 11.0, 19.0, 27.0, 35.0, 43.0, 51.0, 57.0])
    event_amps = np.full(len(event_times), 2.5)

    events = np.zeros_like(t)
    for c, a in zip(event_times, event_amps):
        events += make_calcium_transient(t, center=c, amplitude=a, rise=0.25, decay=1.5)

    noise = 0.18 * rng.standard_normal(len(t))
    trace = drift + events + noise

    K = 3.0

    window_seconds = 7.5
    window_pts = int(window_seconds * sample_rate)
    chunk_starts = list(range(0, len(t), window_pts))
    chunk_boundary_times = [t[idx] for idx in chunk_starts[1:]]

    chunk_threshold = np.zeros_like(trace)
    chunk_zone_lo = np.zeros_like(trace)
    chunk_zone_hi = np.zeros_like(trace)
    for start in range(0, len(t), window_pts):
        end = min(start + window_pts, len(t))
        chunk = trace[start:end]
        median = np.median(chunk)
        mad = np.median(np.abs(chunk - median))
        chunk_threshold[start:end] = median + K * mad
        chunk_zone_lo[start:end] = median - K * mad
        chunk_zone_hi[start:end] = median + K * mad

    peak_indices = np.array([np.argmin(np.abs(t - (c + 0.4))) for c in event_times])
    peak_t = t[peak_indices]
    peak_y = trace[peak_indices]

    session_median = np.median(trace)
    session_mad = np.median(np.abs(trace - session_median))
    session_threshold_value = session_median + K * session_mad
    session_zone_lo = session_median - K * session_mad
    session_zone_hi = session_median + K * session_mad

    def _local_max_above(values, threshold_per_sample):
        det = []
        for i in range(1, len(values) - 1):
            if values[i] > threshold_per_sample[i] and values[i] >= values[i - 1] and values[i] > values[i + 1]:
                det.append(i)
        return np.array(det, dtype=int)

    threshold_top = np.full_like(trace, session_threshold_value)
    detected_top = _local_max_above(trace, threshold_top)
    detected_bot = _local_max_above(trace, chunk_threshold)
    _, _, fn_top = _classify_detections(detected_top, event_times, t)
    _, _, fn_bot = _classify_detections(detected_bot, event_times, t)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.8, 4.8), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.4},
    )

    y_min = float(np.min(trace) - 1.6)  # extra padding below the trace for the chunk bracket
    y_max = float(np.max(trace) + 0.6)
    ax_top.set_ylim(y_min, y_max)

    # Top panel: session-wide threshold. No chunk grid: the threshold is session-wide, so the
    # chunk concept does not apply here yet. Threshold styling unified with fig3
    # (#555 medium-dark gray at ~1.2 pt) so the reference-line hierarchy is consistent
    # across the two chunking figures.
    ax_top.axhline(session_threshold_value, color="#555555", linewidth=1.2, linestyle="--", zorder=2.5)
    ax_top.plot(t, trace, color=COLOR_TRACE, linewidth=1.0, zorder=3)
    for ev_i, (pt, py) in enumerate(zip(peak_t, peak_y)):
        if ev_i in fn_top:
            ax_top.scatter(pt, py, s=MARKER_SIZE_REJECTED, c=COLOR_REJECTED, edgecolors="none", zorder=4)
        else:
            ax_top.scatter(pt, py, s=MARKER_SIZE_ACCEPTED, c=COLOR_ACCEPTED, edgecolors="none", zorder=4)
    ax_top.set_ylabel("z-score")
    ax_top.set_title("session-wide threshold: every event missed", loc="center", fontsize=11)

    # Bottom panel: chunk-local threshold. Chunk grid + bracket marker on the first chunk.
    # Threshold and boundary styling unified with fig3: threshold #555 at ~1.2 pt (forward
    # in the reference layer), chunk boundaries #aaa at ~0.6 pt (recede behind the threshold).
    ax_bot.plot(t, chunk_threshold, color="#555555", linewidth=1.2, linestyle="--", drawstyle="steps-post", zorder=2.5)
    for cb in chunk_boundary_times:
        ax_bot.axvline(cb, color="#aaaaaa", linewidth=0.6, linestyle=":", zorder=1.5, alpha=0.7)
    ax_bot.plot(t, trace, color=COLOR_TRACE, linewidth=1.0, zorder=3)
    for ev_i, (pt, py) in enumerate(zip(peak_t, peak_y)):
        if ev_i in fn_bot:
            ax_bot.scatter(pt, py, s=MARKER_SIZE_REJECTED, c=COLOR_REJECTED, edgecolors="none", zorder=4)
        else:
            ax_bot.scatter(pt, py, s=MARKER_SIZE_ACCEPTED, c=COLOR_ACCEPTED, edgecolors="none", zorder=4)
    _draw_chunk_bracket(ax_bot, 0.0, window_seconds, f"1 chunk = {window_seconds:g} s")
    ax_bot.set_ylabel("z-score")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_title("chunk-local threshold: most events recovered", loc="center", fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_ACCEPTED, markeredgecolor="none",
               markersize=8, label="detected event (TP)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_REJECTED, markeredgecolor="none",
               markersize=8, label="missed event (FN)"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2,
               label="detection threshold"),
        Line2D([0], [0], color="#aaaaaa", linestyle=":", linewidth=0.6,
               label="chunk boundary"),
    ]
    fig.tight_layout()
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=9,
               handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.16)

    fig.savefig(OUT / "fig2_drift_failure_and_fix.svg", bbox_inches="tight")
    plt.close(fig)


def figure_3_threshold_three_regimes():
    """3-row × 2-column figure showing detection at three window sizes (2 s, 15 s, 60 s).

    Wide left column shows the full trace with TP/FN markers. Narrow right column zooms
    into a 5 s slice (t = 8-13 s) and adds FP markers (hollow red circles). The zoom
    surfaces the variance-failure consequence (many false positives at short windows)
    without overloading the wide overview panels.
    """
    t, trace, event_times, _, sr = _make_drifting_trace_with_events()
    duration = t[-1] + 1 / sr
    K1, K2 = 2.0, 3.0
    windows = [1.0, 15.0, 60.0]
    titles = [
        "window = 1 s (jagged threshold; many false positives, see zoom)",
        "window = 15 s (threshold adapts; detection more reliable)",
        "window = 60 s (single flat threshold; events missed in low-baseline stretches)",
    ]
    peak_indices = np.array([np.argmin(np.abs(t - (c + 0.4))) for c in event_times])
    peak_t = t[peak_indices]
    peak_y = trace[peak_indices]

    zoom_lo, zoom_hi = 9.0, 14.0
    y_min = float(np.min(trace) - 1.6)  # extra padding below the trace for the chunk bracket
    y_max = float(np.max(trace) + 0.7)

    panel_labels = [("A", "B"), ("C", "D"), ("E", "F")]

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.5, wspace=0.08)

    for i, (w, title) in enumerate(zip(windows, titles)):
        ax_w = fig.add_subplot(gs[i, 0])
        ax_z = fig.add_subplot(gs[i, 1], sharey=ax_w)
        wide_label, zoom_label = panel_labels[i]

        chunk_size = max(2, int(round(w * sr)))
        thresholds = np.zeros_like(trace)
        zone_lo = np.zeros_like(trace)
        detected_idx_list = []
        for s in range(0, len(trace), chunk_size):
            e = min(s + chunk_size, len(trace))
            chunk = trace[s:e]
            if len(chunk) < 3:
                continue
            m = np.median(chunk)
            mad = np.median(np.abs(chunk - m))
            kept = chunk[chunk <= m + K1 * mad]
            if len(kept) < 3 or np.median(np.abs(kept - np.median(kept))) <= 0:
                thr = m + K2 * mad
                lo = m - K2 * mad
            else:
                m_c = np.median(kept)
                mad_c = np.median(np.abs(kept - m_c))
                thr = m_c + K2 * mad_c
                lo = m_c - K2 * mad_c
            thresholds[s:e] = thr
            zone_lo[s:e] = lo
            for ii in range(1, len(chunk) - 1):
                if chunk[ii] > thr and chunk[ii] >= chunk[ii - 1] and chunk[ii] > chunk[ii + 1]:
                    detected_idx_list.append(s + ii)
        detected_idx = np.array(detected_idx_list, dtype=int)
        _, fp_idx, fn_event_indices = _classify_detections(detected_idx, event_times, t)

        # Wide panel (TP + FN only). Threshold uses medium-dark gray (#555) at moderate
        # weight; chunk boundaries use lighter gray (#aaa) at thinner weight, so the
        # within-reference hierarchy gives the threshold visual priority over the boundaries.
        ax_w.plot(t, thresholds, color="#555555", linewidth=1.2, linestyle="--",
                  drawstyle="steps-post", zorder=2)
        for s in range(chunk_size, len(trace), chunk_size):
            ax_w.axvline(s / sr, color="#aaaaaa", linewidth=0.6, linestyle=":", alpha=0.7, zorder=1.5)
        ax_w.plot(t, trace, color=COLOR_TRACE, linewidth=0.9, zorder=3)
        for ev_i, (pt, py) in enumerate(zip(peak_t, peak_y)):
            if ev_i in fn_event_indices:
                ax_w.scatter(pt, py, s=MARKER_SIZE_REJECTED, c=COLOR_REJECTED, edgecolors="none", zorder=4)
            else:
                ax_w.scatter(pt, py, s=MARKER_SIZE_ACCEPTED, c=COLOR_ACCEPTED, edgecolors="none", zorder=4)
        ax_w.axvspan(zoom_lo, zoom_hi, color="#f0eef8", alpha=0.25, zorder=0.5)
        # For the very-short-window panel (A), annotate the second chunk so the bracket
        # sits clear of the y-axis; for the wider windows the first chunk is fine.
        bracket_start = w if i == 0 else 0.0
        _draw_chunk_bracket(ax_w, bracket_start, bracket_start + w, f"1 chunk = {w:g} s", panel_width=duration)
        _draw_panel_label(ax_w, wide_label, outside=(i==2))
        ax_w.set_title(title, loc="center", fontsize=10)
        ax_w.set_xlim(0, duration)
        ax_w.set_ylim(y_min, y_max)
        ax_w.set_ylabel("z-score")

        # Zoom panel (TP + FN + FP). Same threshold/chunk styling as the wide panel.
        ax_z.plot(t, thresholds, color="#555555", linewidth=1.2, linestyle="--",
                  drawstyle="steps-post", zorder=2)
        for s in range(chunk_size, len(trace), chunk_size):
            sb = s / sr
            if zoom_lo <= sb <= zoom_hi:
                ax_z.axvline(sb, color="#aaaaaa", linewidth=0.6, linestyle=":", alpha=0.7, zorder=1.5)
        ax_z.plot(t, trace, color=COLOR_TRACE, linewidth=0.9, zorder=3)
        for ev_i, (pt, py) in enumerate(zip(peak_t, peak_y)):
            if not (zoom_lo <= pt <= zoom_hi):
                continue
            if ev_i in fn_event_indices:
                ax_z.scatter(pt, py, s=MARKER_SIZE_REJECTED, c=COLOR_REJECTED, edgecolors="none", zorder=4)
            else:
                ax_z.scatter(pt, py, s=MARKER_SIZE_ACCEPTED, c=COLOR_ACCEPTED, edgecolors="none", zorder=4)
        for di in fp_idx:
            if not (zoom_lo <= t[di] <= zoom_hi):
                continue
            ax_z.scatter(t[di], trace[di], s=MARKER_SIZE_ACCEPTED, c="#ff7f0e",
                         edgecolors="none", zorder=4)
        ax_z.set_xlim(zoom_lo, zoom_hi)
        ax_z.set_ylim(y_min, y_max)
        ax_z.tick_params(labelleft=False)
        _draw_panel_label(ax_z, zoom_label, outside=(i==2))
        if i == 0:
            ax_z.set_title(f"zoom {int(zoom_lo)}-{int(zoom_hi)} s",
                           loc="center", fontsize=10, color="#444")
        if i == len(windows) - 1:
            ax_w.set_xlabel("time (s)")
            ax_z.set_xlabel("time (s)")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_ACCEPTED, markeredgecolor="none",
               markersize=8, label="detected event (TP)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_REJECTED, markeredgecolor="none",
               markersize=8, label="missed event (FN)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="#ff7f0e", markeredgecolor="none",
               markersize=8, label="false positive (FP)"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2,
               label="detection threshold"),
        Line2D([0], [0], color="#aaaaaa", linestyle=":", linewidth=0.6,
               label="chunk boundary"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=9,
               handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.12)

    fig.savefig(OUT / "fig3_threshold_three_regimes.svg", bbox_inches="tight")
    plt.close(fig)


def figure_4_two_stage_walkthrough():
    """Three-panel walkthrough showing the two-stage scheme rescuing a smaller event that
    the naive single-stage threshold would have missed. Top: naive threshold drawn on raw
    chunk. Middle: blanked samples + first-stage cutoff. Bottom: original chunk restored
    with two-stage threshold drawn dark and naive faintly for comparison.
    """
    rng = np.random.default_rng(31)
    sample_rate = 100
    duration = 15.0
    t = np.arange(0, duration, 1 / sample_rate)

    event_times = np.array([3.0, 12.0])

    events = np.zeros_like(t)
    events += make_calcium_transient(t, center=event_times[0], amplitude=18.0, rise=0.30, decay=2.0)
    events += make_calcium_transient(t, center=event_times[1], amplitude=2.4, rise=0.10, decay=0.5)

    noise = 0.50 * rng.standard_normal(len(t))
    trace = events + noise

    K1 = 2.0
    K2 = 3.0

    m_raw = np.median(trace)
    mad_raw = np.median(np.abs(trace - m_raw))
    first_cutoff = m_raw + K1 * mad_raw
    naive_threshold = m_raw + K2 * mad_raw

    above_first = trace > first_cutoff
    blanked = np.where(above_first, np.nan, trace)

    kept = trace[~above_first]
    m_clean = np.median(kept)
    mad_clean = np.median(np.abs(kept - m_clean))
    second_threshold = m_clean + K2 * mad_clean

    peak_indices = np.array([np.argmin(np.abs(t - (c + 0.3))) for c in event_times])
    peak_t = t[peak_indices]
    peak_y = trace[peak_indices]

    caught_naive = peak_y > naive_threshold
    caught_two_stage = peak_y > second_threshold

    fig, (ax_left, ax_mid, ax_right) = plt.subplots(
        3, 1, figsize=(9.5, 6.6), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.32},
    )

    for ax in (ax_left, ax_mid, ax_right):
        ax.set_xlim(0, duration)
    ax_right.set_xlabel("time (s)")

    ax_left.plot(t, trace, color=COLOR_TRACE, linewidth=1.0, zorder=3)
    ax_left.axhline(
        naive_threshold, color="#555555", linewidth=1.2, linestyle="--", zorder=2,
    )
    # Use a larger marker size in fig4 so both TP (red) and FN (navy) circles stay
    # visible at thumbnail render sizes; both rows use the same size for consistency.
    for pt, py, c in zip(peak_t, peak_y, caught_naive):
        color = COLOR_ACCEPTED if c else COLOR_REJECTED
        ax_left.scatter(pt, py, s=110, c=color, zorder=4, edgecolors="none")
    ax_left.set_ylabel("z-score")
    ax_left.set_title("(1) naive threshold misses the smaller event", loc="center", fontsize=11)
    ax_left.text(0.012, 0.97, "A", transform=ax_left.transAxes, fontsize=15, fontweight="bold", va="top", color="#222")

    # Ghost-trace encoding: the trace stays continuous, with samples above the K₁
    # cutoff drawn in faded green (the "ghost" - same data as the rest of the trace,
    # just deemphasised because the algorithm excluded them from the noise reference)
    # and samples below the cutoff drawn in saturated green (kept). The trimmed and
    # kept samples share a hue because they ARE the same data series; only their role
    # in the noise computation differs.
    trace_below = np.where(above_first, np.nan, trace)
    trace_above = np.where(~above_first, np.nan, trace)
    ax_mid.plot(t, trace_above, color=COLOR_TRACE, alpha=0.30, linewidth=1.0, zorder=2.5)
    ax_mid.plot(t, trace_below, color=COLOR_TRACE, linewidth=1.0, zorder=3)
    ax_mid.axhline(
        first_cutoff, color="#7b4fa3", linewidth=1.4, linestyle=":", zorder=2,
    )
    ax_mid.set_ylabel("z-score")
    ax_mid.set_title("(2) remove samples above K₁ × MAD: noise-only subset", loc="center", fontsize=11)
    ax_mid.text(0.012, 0.97, "B", transform=ax_mid.transAxes, fontsize=15, fontweight="bold", va="top", color="#222")

    noise_low = m_clean - K2 * mad_clean
    noise_high = m_clean + K2 * mad_clean
    ax_right.axhspan(noise_low, noise_high, color="#fff5cc", alpha=0.9, zorder=1)
    ax_right.plot(t, trace, color=COLOR_TRACE, linewidth=1.0, zorder=3)
    ax_right.axhline(
        naive_threshold, color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=1.8,
    )
    ax_right.axhline(
        second_threshold, color="#555555", linewidth=1.6, linestyle="--", zorder=2,
    )
    for pt, py, c in zip(peak_t, peak_y, caught_two_stage):
        color = COLOR_ACCEPTED if c else COLOR_REJECTED
        ax_right.scatter(pt, py, s=110, c=color, zorder=4, edgecolors="none")
    ax_right.set_ylabel("z-score")
    ax_right.set_title("(3) two-stage threshold catches both events", loc="center", fontsize=11)
    ax_right.text(0.012, 0.97, "C", transform=ax_right.transAxes, fontsize=15, fontweight="bold", va="top", color="#222")

    # Three semantic groups in three stacked legends: markers, threshold lines,
    # annotations. matplotlib's single fig.legend() can't produce a 2-3-2 layout
    # natively (it fills uniformly by ncol), so we stack three legends vertically.
    markers_legend = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_ACCEPTED, markeredgecolor="none",
               markersize=8, label="detected event (TP)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_REJECTED, markeredgecolor="none",
               markersize=8, label="missed event (FN)"),
    ]
    thresholds_legend = [
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.6,
               label="two-stage threshold"),
        Line2D([0], [0], color="#7b4fa3", linestyle=":", linewidth=1.4,
               label="K₁ cutoff: samples above are trimmed"),
        Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.0,
               label="naive threshold (for comparison)"),
    ]
    annotations_legend = [
        Line2D([0], [0], color=COLOR_TRACE, alpha=0.30, linewidth=1.0,
               label="trimmed samples (excluded from noise reference)"),
        Patch(facecolor="#fff5cc", alpha=0.9, label="cleaned noise band"),
    ]
    fig.tight_layout()
    leg_kwargs = dict(loc="lower center", frameon=False, fontsize=9,
                      handletextpad=0.5, columnspacing=1.6)
    leg1 = fig.legend(handles=markers_legend, ncol=2,
                      bbox_to_anchor=(0.5, 0.13), **leg_kwargs)
    leg2 = fig.legend(handles=thresholds_legend, ncol=3,
                      bbox_to_anchor=(0.5, 0.07), **leg_kwargs)
    leg3 = fig.legend(handles=annotations_legend, ncol=2,
                      bbox_to_anchor=(0.5, 0.01), **leg_kwargs)
    fig.subplots_adjust(bottom=0.26)

    fig.savefig(OUT / "fig4_two_stage_walkthrough.svg", bbox_inches="tight")
    plt.close(fig)


def figure_5_summary_statistics():
    """Two summary scalars (event rate, mean amplitude) on a single detected event list.
    Left (two stacked panels showing the same trace as a continuation): trace with detected
    events marked. Right: histogram of per-event amplitudes (mean amplitude as centroid).
    """
    rng = np.random.default_rng(42)
    sample_rate = 100
    duration = 360.0  # 6-minute recording, shown as two stacked 3-min halves
    t = np.arange(0, duration, 1 / sample_rate)

    n_events = 40  # ~7/min, in the realistic range for spontaneous photometry
    event_times = np.sort(rng.uniform(3, duration - 3, n_events))
    event_amps = rng.uniform(1.5, 4.0, n_events)

    events = np.zeros_like(t)
    for c, a in zip(event_times, event_amps):
        events += make_calcium_transient(t, center=c, amplitude=a, rise=0.10, decay=1.0)
    # Smooth white noise into a low-pass-filtered process so it has the temporal
    # correlation length of a real preprocessed photometry trace (~150 ms here);
    # without this, white noise at 100 Hz produces too many K = 3 crossings and the
    # rate is dominated by noise rather than by the synthesised events.
    raw_noise = rng.standard_normal(len(t))
    window = 25
    kernel = np.ones(window) / window
    smoothed = np.convolve(raw_noise, kernel, mode="same")
    noise = 0.30 * smoothed / np.std(smoothed)
    trace = events + noise

    median = np.median(trace)
    mad = np.median(np.abs(trace - median))

    is_local_max = np.zeros(len(trace), dtype=bool)
    is_local_max[1:-1] = (trace[1:-1] > trace[:-2]) & (trace[1:-1] > trace[2:])

    # Plot the trace and report amplitudes in MAD-relative units so the trace y-axis
    # and the histogram x-axis share the same units; the threshold sits at K = 3 in
    # those units, matching the histogram's first bin starting at 3.
    trace_norm = (trace - median) / mad
    threshold_norm = 3.0

    # Greedy peak-finding with a minimum-distance constraint, so each biological event
    # contributes one detection at its peak rather than several from sub-peak noise on
    # the flanks. Without this step the rate is dominated by within-event multi-firing
    # rather than the underlying biological event rate.
    candidate_idx = np.where(is_local_max & (trace_norm > threshold_norm))[0]
    min_distance_samples = int(1.5 * sample_rate)  # ~1.5 s refractory window
    candidate_sorted = candidate_idx[np.argsort(-trace_norm[candidate_idx])]
    selected_set = []
    for idx in candidate_sorted:
        if all(abs(idx - s) >= min_distance_samples for s in selected_set):
            selected_set.append(idx)
    detected = np.array(sorted(selected_set), dtype=int)

    rate = len(detected) / (duration / 60)
    amps_mad = trace_norm[detected]
    mean_amp = float(np.mean(amps_mad))

    # Stacked-continuation layout: the 6-minute trace is split into two 3-minute halves
    # so individual events stay visible in the trace panels while the histogram on the
    # right accumulates ~40 events for a more stable amplitude distribution.
    fig = plt.figure(figsize=(12.5, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.2],
                          hspace=0.35, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharey=ax_top)
    ax_right = fig.add_subplot(gs[:, 1])

    half = duration / 2
    detected_t = t[detected]
    detected_y = trace_norm[detected]

    for ax, x_lo, x_hi in ((ax_top, 0.0, half), (ax_bot, half, duration)):
        in_range = (t >= x_lo) & (t <= x_hi)
        ax.plot(t[in_range], trace_norm[in_range], color=COLOR_TRACE, linewidth=0.9, zorder=2)
        ax.axhline(threshold_norm, color="#555555", linewidth=1.2, linestyle="--", zorder=2)
        det_mask = (detected_t >= x_lo) & (detected_t <= x_hi)
        ax.plot(detected_t[det_mask], detected_y[det_mask], "o", color=COLOR_PEAK,
                markersize=5, markeredgewidth=0, zorder=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("z-score")

    ax_top.set_title(f"event rate = {rate:.1f} / min", loc="center", fontsize=11)
    ax_top.tick_params(labelbottom=False)
    ax_top.text(0.99, 0.04, "↓ continued below", transform=ax_top.transAxes,
                ha="right", va="bottom", fontsize=8, color="#666", style="italic")
    ax_bot.set_xlabel("time (s)")

    bins = np.linspace(0, float(np.max(amps_mad)) * 1.10, 15)
    ax_right.hist(amps_mad, bins=bins, color="#85929e", edgecolor="white",
                  linewidth=0.5, zorder=2)
    mean_color = "#b8860b"  # dark gold: distinct from red bars and red peak markers
    ax_right.axvline(mean_amp, color=mean_color, linewidth=1.6, linestyle="--", zorder=3)
    ax_right.set_xlabel("amplitude (× MAD)")
    ax_right.set_ylabel("event count")
    ax_right.set_title(f"mean amplitude = {mean_amp:.1f} × MAD",
                       loc="center", fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOR_PEAK, markeredgecolor="none",
               markersize=8, label="detected event (TP)"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2,
               label="detection threshold"),
        Line2D([0], [0], color=mean_color, linestyle="--", linewidth=1.6,
               label="mean amplitude"),
    ]
    fig.tight_layout()
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9,
               handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.10)

    fig.savefig(OUT / "fig5_summary_statistics.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_transient_extraction()
    figure_2_drift_failure_and_fix()
    figure_3_threshold_three_regimes()
    figure_4_two_stage_walkthrough()
    figure_5_summary_statistics()
    print("Wrote SVGs to", OUT)
