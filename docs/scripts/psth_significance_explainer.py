# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Generate SVG figures for the PSTH significance explanation page.

The bootstrap and the consecutive-run filter mirror
src/guppy/analysis/psth_significance.py, and the significance bar is drawn inside the
plot at the same height fraction the Step 5 tab uses, so the figures match what the
visualization shows. Only the input traces are synthetic and designed to make each
point clean.

Run with:

    uv run docs/scripts/psth_significance_explainer.py

Inline dependencies above let `uv run` resolve everything without needing the
surrounding project to be installed. Outputs are written directly to
docs/_static/images/psth_significance_explainer/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats

OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "psth_significance_explainer"
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

COLOR_ESTIMATE = "#1f77b4"
COLOR_SIGNIFICANT = "#d62728"
COLOR_MUTED = "#999999"
SESSION_COLORS = ("#ff7f0e", "#2ca02c", "#9467bd")

# Matches _BAR_HEIGHT_FRACTION in src/guppy/visualization/psth_significance.py.
BAR_HEIGHT_FRACTION = 0.04

# Every figure on the page draws the same example session, so a reader following the
# explainer sees one worked example rather than a new dataset per figure.
FILTER_WINDOW = 15
MINIMUM_RUN = 2 * FILTER_WINDOW
BASELINE_TIME = -1.0
PEAK_TIME = 1.0


def example_session() -> tuple[np.ndarray, np.ndarray]:
    """The one example session every figure on the page is drawn from.

    Most trials respond modestly and a few respond strongly, which is both realistic and
    what makes the resampling distribution at the peak lopsided enough to show what the
    BCa correction does.
    """
    rng = np.random.default_rng(4)
    time = np.linspace(-2, 4, 400)
    shape = np.exp(-((time - PEAK_TIME) ** 2) / (2 * 0.5**2))
    amplitudes = np.concatenate([rng.normal(1.2, 0.35, 15), rng.normal(4.2, 0.5, 3)])
    trials = moving_average(amplitudes[:, None] * shape + rng.normal(0, 1.5, size=(18, time.size)), FILTER_WINDOW)
    return time, trials


def timepoint_index(time: np.ndarray, moment: float) -> int:
    """Index of the sample nearest ``moment``."""
    return int(np.argmin(np.abs(time - moment)))


def moving_average(trials: np.ndarray, window: int) -> np.ndarray:
    """Mirror of GuPPy's moving-average filter, applied per trial."""
    kernel = np.ones(window) / window
    return np.array([np.convolve(trial, kernel, mode="same") for trial in trials])


def bootstrap_result(
    samples: np.ndarray, num_resamples: int = 1000, rng: np.random.Generator | None = None
) -> stats._resampling.BootstrapResult:
    """Mirror of GuPPy's BCa bootstrap, returning scipy's full result."""
    return stats.bootstrap(
        (samples,),
        np.nanmean,
        axis=0,
        n_resamples=num_resamples,
        confidence_level=0.95,
        method="BCa",
        batch=100,
        rng=rng,
    )


def bootstrap_interval(
    samples: np.ndarray, num_resamples: int = 1000, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of GuPPy's BCa interval on the mean."""
    result = bootstrap_result(samples, num_resamples=num_resamples, rng=rng)
    return result.confidence_interval.low, result.confidence_interval.high


def surviving_runs(excludes_zero: np.ndarray, minimum_consecutive_samples: int) -> np.ndarray:
    """Mirror of GuPPy's consecutive-run filter."""
    mask = np.zeros_like(excludes_zero, dtype=bool)
    labeled, _ = ndimage.label(excludes_zero)
    for run_slice in ndimage.find_objects(labeled):
        if run_slice[0].stop - run_slice[0].start > minimum_consecutive_samples:
            mask[run_slice] = True
    return mask


def draw_significance_bar(
    axes: plt.Axes,
    time: np.ndarray,
    mask: np.ndarray,
    *,
    bottom: float,
    top: float,
    offset: float = 0.0,
    color: str = COLOR_SIGNIFICANT,
    label: str | None = None,
) -> None:
    """Draw significant stretches as a bar inside the plot, as the Step 5 tab does."""
    height = (top - bottom) * BAR_HEIGHT_FRACTION
    bar_top = top - offset * height * 1.4
    boundaries = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    drawn = False
    for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        axes.fill_between(
            [time[start], time[end - 1]],
            bar_top - height,
            bar_top,
            color=color,
            lw=0,
            label=None if drawn else label,
        )
        drawn = True


def figure_1_bootstrap_mechanism() -> None:
    """Trials, the resampled means built from them, and the distribution at two timepoints."""
    time, trials = example_session()

    result = bootstrap_result(trials, rng=np.random.default_rng(5))
    lower, upper = result.confidence_interval.low, result.confidence_interval.high
    # scipy returns the resampled statistic as (timepoint, resample).
    distribution = result.bootstrap_distribution
    estimate = np.nanmean(trials, axis=0)

    baseline_index = timepoint_index(time, BASELINE_TIME)
    peak_index = timepoint_index(time, PEAK_TIME)

    figure, axes = plt.subplot_mosaic(
        [["trials", "resamples"], ["baseline", "peak"]], figsize=(8.0, 5.6), constrained_layout=True
    )

    def mark_timepoints(panel: plt.Axes) -> None:
        for index, color in ((baseline_index, "#7f7f7f"), (peak_index, COLOR_SIGNIFICANT)):
            panel.axvline(time[index], color=color, lw=1.2, ls=":")

    # 1. the individual trials the resampling draws from
    for trial in trials:
        axes["trials"].plot(time, trial, color=COLOR_MUTED, lw=0.7, alpha=0.6)
    axes["trials"].plot(time, estimate, color=COLOR_ESTIMATE, lw=2, label="mean of all trials")
    axes["trials"].set_title("The trials, and their mean", fontsize=10)
    axes["trials"].legend(loc="upper left", frameon=False, fontsize=8)

    # 2. the resampled means, each one an average of a different draw of those trials
    for row in distribution.T[:150]:
        axes["resamples"].plot(time, row, color=COLOR_ESTIMATE, lw=0.5, alpha=0.12)
    axes["resamples"].plot(time, estimate, color=COLOR_ESTIMATE, lw=2)
    axes["resamples"].set_title("Steps 1-3: each faint line is one resampled mean", fontsize=10)

    for name in ("trials", "resamples"):
        mark_timepoints(axes[name])
        axes[name].axhline(0, color="black", ls="--", lw=0.8)
        axes[name].set_xlabel("Time from event (s)")
        axes[name].set_ylabel("z-score")
    axes["resamples"].sharey(axes["trials"])

    # 3. the distribution of those means at one timepoint, which the interval is taken from
    for name, index, color, caption in (
        ("baseline", baseline_index, "#7f7f7f", f"at t = {BASELINE_TIME} s"),
        ("peak", peak_index, COLOR_SIGNIFICANT, f"at t = {PEAK_TIME} s"),
    ):
        panel = axes[name]
        panel.hist(distribution[index], bins=40, color=COLOR_ESTIMATE, alpha=0.45, lw=0)
        panel.axvline(0, color="black", ls="--", lw=1.2)
        for bound in (lower[index], upper[index]):
            panel.axvline(bound, color=color, lw=1.6)
        excludes = (lower[index] > 0) or (upper[index] < 0)
        panel.set_title(
            f"Step 4: resampled means {caption}\n"
            f"{'interval excludes zero' if excludes else 'interval contains zero'}",
            fontsize=10,
            color=color,
        )
        panel.set_xlabel("mean z-score")
        panel.set_ylabel("resamples")

    figure.savefig(OUT / "fig1_bootstrap_mechanism.svg", bbox_inches="tight")
    plt.close(figure)


def figure_2_bca_percentiles() -> None:
    """Where BCa puts the two percentiles, against the naive 2.5th and 97.5th."""
    time, trials = example_session()
    cases = (
        (f"At t = {BASELINE_TIME} s, evenly spread", trials[:, [timepoint_index(time, BASELINE_TIME)]]),
        (f"At t = {PEAK_TIME} s, lopsided", trials[:, [timepoint_index(time, PEAK_TIME)]]),
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    for panel, (title, samples) in zip(axes, cases):
        intervals = {}
        for method in ("percentile", "BCa"):
            result = stats.bootstrap(
                (samples,),
                np.nanmean,
                axis=0,
                n_resamples=2000,
                confidence_level=0.95,
                method=method,
                batch=200,
                rng=np.random.default_rng(3),
            )
            intervals[method] = (result.confidence_interval.low[0], result.confidence_interval.high[0])
            distribution = result.bootstrap_distribution[0]

        panel.hist(distribution, bins=45, color=COLOR_ESTIMATE, alpha=0.4, lw=0)
        for index, bound in enumerate(intervals["percentile"]):
            panel.axvline(bound, color="#555555", lw=1.4, ls="--", label="2.5th / 97.5th percentile" if not index else None)
        for index, bound in enumerate(intervals["BCa"]):
            panel.axvline(bound, color=COLOR_SIGNIFICANT, lw=1.6, label="BCa" if not index else None)
        panel.set_title(title, fontsize=10)
        panel.set_xlabel("mean z-score")
        panel.set_ylabel("resamples")

    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    figure.savefig(OUT / "fig2_bca_percentiles.svg", bbox_inches="tight")
    plt.close(figure)


def figure_4_significance_result() -> None:
    """The mean PSTH, its bootstrap interval, and the significant stretches it implies."""
    time, trials = example_session()

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(5))
    estimate = np.nanmean(trials, axis=0)
    mask = surviving_runs((lower > 0) | (upper < 0), MINIMUM_RUN)

    figure, axes = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    bottom = float(np.nanmin(lower))
    top = float(np.nanmax(upper))
    top = top + (top - bottom) * (BAR_HEIGHT_FRACTION * 3)

    axes.fill_between(time, lower, upper, color=COLOR_ESTIMATE, alpha=0.25, lw=0, label="95% confidence interval")
    axes.plot(time, estimate, color=COLOR_ESTIMATE, lw=1.8, label="mean PSTH")
    axes.axhline(0, color="black", ls="--", lw=1)
    draw_significance_bar(axes, time, mask, bottom=bottom, top=top, label="significant")

    axes.set_ylim(bottom, top)
    axes.set_xlabel("Time from event (s)")
    axes.set_ylabel("z-score")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9, ncols=3)

    figure.savefig(OUT / "fig4_significance_result.svg", bbox_inches="tight")
    plt.close(figure)


def figure_3_minimum_duration() -> None:
    """Isolated timepoints clear zero by chance; only long enough stretches are kept."""
    time, trials = example_session()

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(5))
    estimate = np.nanmean(trials, axis=0)
    raw = (lower > 0) | (upper < 0)
    kept = surviving_runs(raw, MINIMUM_RUN)

    figure, axes = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    bottom = float(np.nanmin(lower))
    top = float(np.nanmax(upper))
    top = top + (top - bottom) * (BAR_HEIGHT_FRACTION * 8)

    axes.fill_between(time, lower, upper, color=COLOR_ESTIMATE, alpha=0.25, lw=0)
    axes.plot(time, estimate, color=COLOR_ESTIMATE, lw=1.8, label="mean PSTH")
    axes.axhline(0, color="black", ls="--", lw=1)
    draw_significance_bar(
        axes, time, raw, bottom=bottom, top=top, offset=0, color=COLOR_MUTED, label="interval excludes zero"
    )
    draw_significance_bar(
        axes, time, kept, bottom=bottom, top=top, offset=1.6, label="kept: longer than the filter window"
    )

    axes.set_ylim(bottom, top)
    axes.set_xlabel("Time from event (s)")
    axes.set_ylabel("z-score")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9, ncols=3)

    figure.savefig(OUT / "fig3_minimum_duration.svg", bbox_inches="tight")
    plt.close(figure)


def example_group(num_sessions: int = 6) -> tuple[np.ndarray, list[np.ndarray]]:
    """A group of sessions, each with its own trials.

    Sessions differ in how strongly they respond, which is the variability a group-level
    interval is built from.
    """
    rng = np.random.default_rng(9)
    time = np.linspace(-2, 4, 400)
    shape = np.exp(-((time - PEAK_TIME) ** 2) / (2 * 0.5**2))

    sessions = []
    for _ in range(num_sessions):
        amplitude = rng.normal(1.4, 0.5)
        sessions.append(moving_average(amplitude * shape + rng.normal(0, 1.5, size=(20, time.size)), FILTER_WINDOW))
    return time, sessions


def figure_5_group_significance() -> None:
    """The path from individual sessions' trials up to the group's interval."""
    time, sessions = example_group()
    session_averages = np.array([session.mean(axis=0) for session in sessions])
    # Highlight a weak, a middling and a strong session rather than whichever three come
    # first, so the coloured ones span the between-session spread instead of clustering.
    by_peak = np.argsort(session_averages.max(axis=1))
    shown = [by_peak[0], by_peak[len(by_peak) // 2], by_peak[-1]]

    figure, axes = plt.subplot_mosaic(
        [["session0", "session0", "session1", "session1", "session2", "session2"],
         ["averages", "averages", "averages", "result", "result", "result"]],
        figsize=(10.5, 6.0),
        constrained_layout=True,
    )

    # Top row: each highlighted session, reduced to its own average.
    for position, (session_index, color) in enumerate(zip(shown, SESSION_COLORS)):
        panel = axes[f"session{position}"]
        for trial in sessions[session_index]:
            panel.plot(time, trial, color=color, lw=0.5, alpha=0.3)
        panel.plot(time, session_averages[session_index], color=color, lw=2)
        panel.set_title(f"Session {position + 1}", fontsize=10, color=color)
        panel.set_xlabel("Time from event (s)")
    axes["session0"].set_ylabel("z-score")
    axes["session1"].sharey(axes["session0"])
    axes["session2"].sharey(axes["session0"])

    # Bottom left: every session's average, the highlighted ones keeping their color.
    averages = axes["averages"]
    for index, average in enumerate(session_averages):
        if index not in shown:
            averages.plot(time, average, color=COLOR_MUTED, lw=1.1, alpha=0.8)
    for session_index, color in zip(shown, SESSION_COLORS):
        averages.plot(time, session_averages[session_index], color=color, lw=1.6)
    group_mean = np.nanmean(session_averages, axis=0)
    averages.plot(time, group_mean, color="black", lw=2.4, label="group mean")
    averages.plot([], [], color=COLOR_MUTED, lw=1.1, label="another session's average")
    averages.set_title("Each session contributes one average", fontsize=10)
    averages.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=8.5, ncols=2)
    averages.set_ylabel("z-score")

    # Bottom right: resampling those averages gives the group's interval.
    lower, upper = bootstrap_interval(session_averages, rng=np.random.default_rng(2))
    mask = surviving_runs((lower > 0) | (upper < 0), MINIMUM_RUN)
    result = axes["result"]
    bottom = float(min(np.nanmin(lower), session_averages.min()))
    top = float(max(np.nanmax(upper), session_averages.max()))
    top = top + (top - bottom) * (BAR_HEIGHT_FRACTION * 4)
    result.fill_between(time, lower, upper, color=COLOR_ESTIMATE, alpha=0.3, lw=0, label="95% confidence interval")
    result.plot(time, group_mean, color="black", lw=2.4)
    draw_significance_bar(result, time, mask, bottom=bottom, top=top, label="significant")
    result.set_title("Resampling the averages gives the group's interval", fontsize=10)
    # Below the axes: the significance bar occupies the top and the trace the middle.
    result.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=8.5, ncols=2)
    result.set_ylim(bottom, top)
    result.sharey(axes["averages"])

    for name in ("averages", "result"):
        axes[name].set_xlabel("Time from event (s)")
    for panel in axes.values():
        panel.axhline(0, color="black", ls="--", lw=0.8)

    figure.savefig(OUT / "fig5_group_significance.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    figure_1_bootstrap_mechanism()
    figure_2_bca_percentiles()
    figure_3_minimum_duration()
    figure_4_significance_result()
    figure_5_group_significance()
    print(f"Wrote figures to {OUT}")
