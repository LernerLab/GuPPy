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
src/guppy/analysis/psth_significance.py. Only the input traces are synthetic and
designed to make each point clean.

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


def figure_1_confidence_interval_is_the_test() -> None:
    """A CI that clears zero is the test; the shaded band is what does the work."""
    rng = np.random.default_rng(1)
    time = np.linspace(-2, 4, 400)
    response = 2.2 * np.exp(-((time - 1.0) ** 2) / (2 * 0.6**2))
    trials = moving_average(response + rng.normal(0, 1.4, size=(24, time.size)), window=15)

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(2))
    estimate = np.nanmean(trials, axis=0)
    excludes_zero = (lower > 0) | (upper < 0)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 4.6), sharex=True, height_ratios=[3, 0.5], constrained_layout=True
    )

    top.fill_between(time, lower, upper, color=COLOR_ESTIMATE, alpha=0.25, lw=0, label="95% bootstrap CI")
    top.plot(time, estimate, color=COLOR_ESTIMATE, lw=1.8, label="mean PSTH")
    top.axhline(0, color="black", ls="--", lw=1, label="no change from baseline")
    top.set_ylabel("z-score")
    top.legend(loc="upper left", frameon=False, fontsize=9)
    top.set_title("The interval is the test: significant where the band clears zero")

    bottom.imshow(
        excludes_zero.reshape(1, -1),
        aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["#00000000", COLOR_SIGNIFICANT]),
        extent=[time[0], time[-1], 0, 1],
        interpolation="nearest",
    )
    bottom.set_yticks([])
    bottom.set_xlabel("Time from event (s)")
    bottom.set_ylabel("CI\nexcludes 0", rotation=0, ha="right", va="center", fontsize=9)

    figure.savefig(OUT / "fig1_interval_is_the_test.svg", bbox_inches="tight")
    plt.close(figure)


def figure_2_consecutive_run_filter() -> None:
    """Pointwise intervals scatter false positives; the run-length filter removes them."""
    rng = np.random.default_rng(7)
    time = np.linspace(-2, 4, 600)
    response = 1.6 * np.exp(-((time - 1.2) ** 2) / (2 * 0.5**2))
    trials = response + rng.normal(0, 1.5, size=(16, time.size))

    lower, upper = bootstrap_interval(trials, rng=np.random.default_rng(8))
    estimate = np.nanmean(trials, axis=0)
    raw = (lower > 0) | (upper < 0)
    minimum_consecutive_samples = 25
    kept = surviving_runs(raw, minimum_consecutive_samples)

    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(7.4, 5.0), sharex=True, height_ratios=[3, 0.5, 0.5], constrained_layout=True
    )

    top.fill_between(time, lower, upper, color=COLOR_ESTIMATE, alpha=0.25, lw=0)
    top.plot(time, estimate, color=COLOR_ESTIMATE, lw=1.8)
    top.axhline(0, color="black", ls="--", lw=1)
    top.set_ylabel("z-score")
    top.set_title("Each timepoint is tested on its own, so noise alone clears zero sometimes")

    strip = plt.matplotlib.colors.ListedColormap(["#00000000", COLOR_MUTED])
    kept_map = plt.matplotlib.colors.ListedColormap(["#00000000", COLOR_SIGNIFICANT])
    middle.imshow(
        raw.reshape(1, -1), aspect="auto", cmap=strip, extent=[time[0], time[-1], 0, 1], interpolation="nearest"
    )
    middle.set_yticks([])
    middle.set_ylabel("every\ntimepoint", rotation=0, ha="right", va="center", fontsize=9)

    bottom.imshow(
        kept.reshape(1, -1),
        aspect="auto",
        cmap=kept_map,
        extent=[time[0], time[-1], 0, 1],
        interpolation="nearest",
    )
    bottom.set_yticks([])
    bottom.set_ylabel("runs longer\nthan the filter", rotation=0, ha="right", va="center", fontsize=9)
    bottom.set_xlabel("Time from event (s)")

    figure.savefig(OUT / "fig2_consecutive_run_filter.svg", bbox_inches="tight")
    plt.close(figure)


def figure_3_hierarchy() -> None:
    """Pooling trials across sessions inflates the false-positive rate; averaging does not."""
    rng = np.random.default_rng(11)
    num_sessions, num_trials = 5, 50
    between_session_sd, trial_sd = 1.0, 1.0
    num_simulations = 300

    def one_group() -> np.ndarray:
        session_effects = rng.normal(0, between_session_sd, num_sessions)
        return np.array([rng.normal(effect, trial_sd, num_trials) for effect in session_effects])

    def interval_excludes_zero(sample_a: np.ndarray, sample_b: np.ndarray, num_resamples: int = 600) -> bool:
        count_a, count_b = len(sample_a), len(sample_b)
        differences = np.array(
            [
                sample_a[rng.integers(0, count_a, count_a)].mean() - sample_b[rng.integers(0, count_b, count_b)].mean()
                for _ in range(num_resamples)
            ]
        )
        low, high = np.percentile(differences, [2.5, 97.5])
        return (low > 0) or (high < 0)

    pooled_hits = averaged_hits = 0
    for _ in range(num_simulations):
        group_a, group_b = one_group(), one_group()
        pooled_hits += interval_excludes_zero(group_a.ravel(), group_b.ravel())
        averaged_hits += interval_excludes_zero(group_a.mean(axis=1), group_b.mean(axis=1))

    pooled_rate = 100 * pooled_hits / num_simulations
    averaged_rate = 100 * averaged_hits / num_simulations

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.8, 3.4), width_ratios=[1.15, 1], constrained_layout=True)

    # Left: the nesting itself.
    for session_index in range(num_sessions):
        offset = num_sessions - session_index
        session_mean = rng.normal(0, 0.55)
        trial_values = session_mean + rng.normal(0, 0.28, 12)
        left.scatter(trial_values, np.full_like(trial_values, offset), s=14, color=COLOR_MUTED, zorder=2)
        left.scatter([session_mean], [offset], s=70, color=COLOR_ESTIMATE, marker="D", zorder=3)
    left.set_yticks(range(1, num_sessions + 1))
    left.set_yticklabels([f"session {index}" for index in range(num_sessions, 0, -1)], fontsize=9)
    left.set_xlabel("response")
    left.set_title("Trials (grey) nest inside sessions (blue)")
    left.scatter([], [], s=14, color=COLOR_MUTED, label="trial")
    left.scatter([], [], s=70, color=COLOR_ESTIMATE, marker="D", label="session mean")
    left.set_ylim(0.3, num_sessions + 0.9)
    left.legend(loc="upper left", frameon=False, fontsize=8, ncols=2)

    # Right: what each choice of unit does to the false-positive rate.
    bars = right.bar(
        ["pool all\ntrials", "average per\nsession"],
        [pooled_rate, averaged_rate],
        color=[COLOR_SIGNIFICANT, COLOR_ESTIMATE],
        width=0.55,
    )
    right.axhline(5, color="black", ls="--", lw=1)
    right.set_xlim(-0.55, 1.85)
    right.text(1.32, 7.0, "nominal 5%", fontsize=8, color="#444444", ha="left")
    right.set_ylabel("false positives (%)")
    right.set_ylim(0, max(pooled_rate, 100) * 1.05)
    right.set_title("With no true difference between groups")
    for bar, rate in zip(bars, [pooled_rate, averaged_rate]):
        right.text(bar.get_x() + bar.get_width() / 2, rate + 1.5, f"{rate:.0f}%", ha="center", fontsize=10)

    figure.savefig(OUT / "fig3_hierarchy.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    figure_1_confidence_interval_is_the_test()
    figure_2_consecutive_run_filter()
    figure_3_hierarchy()
    print(f"Wrote figures to {OUT}")
