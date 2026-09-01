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

# Matches _BAR_HEIGHT_FRACTION in src/guppy/visualization/psth_significance.py.
BAR_HEIGHT_FRACTION = 0.04


def moving_average(trials: np.ndarray, window: int) -> np.ndarray:
    """Mirror of GuPPy's moving-average filter, applied per trial."""
    kernel = np.ones(window) / window
    return np.array([np.convolve(trial, kernel, mode="same") for trial in trials])


def bootstrap_interval(
    samples: np.ndarray, num_resamples: int = 1000, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of GuPPy's BCa interval on the mean."""
    result = stats.bootstrap(
        (samples,),
        np.nanmean,
        axis=0,
        n_resamples=num_resamples,
        confidence_level=0.95,
        method="BCa",
        batch=100,
        rng=rng,
    )
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


def figure_1_bootstrap_interval() -> None:
    """The mean PSTH, its bootstrap interval, and the significant stretches it implies."""
    rng = np.random.default_rng(1)
    time = np.linspace(-2, 4, 400)
    response = 2.2 * np.exp(-((time - 1.0) ** 2) / (2 * 0.6**2))
    trials = moving_average(response + rng.normal(0, 1.4, size=(24, time.size)), window=15)

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(2))
    estimate = np.nanmean(trials, axis=0)
    mask = surviving_runs((lower > 0) | (upper < 0), 12)

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

    figure.savefig(OUT / "fig1_bootstrap_interval.svg", bbox_inches="tight")
    plt.close(figure)


def figure_2_minimum_duration() -> None:
    """Isolated timepoints clear zero by chance; only long enough stretches are kept."""
    rng = np.random.default_rng(7)
    time = np.linspace(-2, 4, 600)
    response = 1.6 * np.exp(-((time - 1.2) ** 2) / (2 * 0.5**2))
    trials = response + rng.normal(0, 1.5, size=(16, time.size))

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(8))
    estimate = np.nanmean(trials, axis=0)
    raw = (lower > 0) | (upper < 0)
    kept = surviving_runs(raw, 25)

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

    figure.savefig(OUT / "fig2_minimum_duration.svg", bbox_inches="tight")
    plt.close(figure)


def figure_3_resampling_unit() -> None:
    """Which level of the hierarchy is resampled in a run folder and in a group folder."""
    rng = np.random.default_rng(11)
    num_sessions = 5

    figure, axes = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)

    session_means = []
    for index in range(num_sessions):
        row = num_sessions - index
        session_mean = rng.normal(0, 0.6)
        session_means.append((session_mean, row))
        trials = session_mean + rng.normal(0, 0.3, 14)
        axes.scatter(trials, np.full_like(trials, row), s=16, color=COLOR_MUTED, zorder=2)
        axes.scatter([session_mean], [row], s=80, color=COLOR_ESTIMATE, marker="D", zorder=3)

    axes.set_yticks(range(1, num_sessions + 1))
    axes.set_yticklabels([f"session {index}" for index in range(num_sessions, 0, -1)], fontsize=9)
    axes.set_xlabel("response")
    axes.set_xlim(-1.9, 3.1)
    axes.set_ylim(0.4, num_sessions + 0.8)

    # Point at one session's trials: the unit inside a run folder.
    top_mean = session_means[0][0]
    axes.annotate(
        "a run folder resamples\nthe trials of one session",
        xy=(top_mean + 0.55, num_sessions),
        xytext=(top_mean + 1.15, num_sessions + 0.55),
        fontsize=8.5,
        color="#444444",
        va="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_MUTED, lw=1.3),
    )

    # Bracket the column of session means: the unit inside a group folder.
    bracket_x = max(mean for mean, _ in session_means) + 1.0
    axes.annotate(
        "", xy=(bracket_x, 1), xytext=(bracket_x, num_sessions),
        arrowprops=dict(arrowstyle="-", color=COLOR_ESTIMATE, lw=1.5),
    )
    axes.text(
        bracket_x + 0.12, (num_sessions + 1) / 2,
        "a group folder\nresamples the\nsession means",
        fontsize=8.5, color=COLOR_ESTIMATE, va="center",
    )

    axes.scatter([], [], s=16, color=COLOR_MUTED, label="trial")
    axes.scatter([], [], s=80, color=COLOR_ESTIMATE, marker="D", label="session mean")
    axes.legend(loc="lower left", frameon=False, fontsize=8.5, ncols=2)

    figure.savefig(OUT / "fig3_resampling_unit.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    figure_1_bootstrap_interval()
    figure_2_minimum_duration()
    figure_3_resampling_unit()
    print(f"Wrote figures to {OUT}")
