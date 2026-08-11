"""Generate the figures for the "Custom plots from GuPPy outputs" tutorial.

Unlike the ``*_explainer.py`` scripts in this directory, which draw synthetic data, this one runs
against real GuPPy output: it copies the sample CSV session, runs Steps 1-4 headlessly, and then
loads the resulting run folder with the same calls the tutorial prints. If a load call in the
tutorial goes stale, this script fails.

Run with:

    uv run python docs/scripts/custom_plots_example.py

It needs ``guppy`` importable (so it runs in the project environment rather than under a PEP-723
header), and the sample CSVs are Git LFS payloads:

    git lfs pull --include="stubbed_testing_data/csv/sample_data_csv_1/*"

Outputs are written to docs/_static/images/custom_plots/.
"""

import shutil
import tempfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from guppy.testing.api import step1, step2, step3, step4

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_SESSION = REPO_ROOT / "stubbed_testing_data" / "csv" / "sample_data_csv_1"
OUT = Path(__file__).resolve().parent.parent / "_static" / "images" / "custom_plots"
OUT.mkdir(parents=True, exist_ok=True)

# The Label Stores mapping from "Your First Analysis": recording site A, event RewardPort.
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_A",
    "Sample_Signal_Channel": "signal_A",
    "Sample_TTL": "RewardPort",
}
RUN_NAME = "1"

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
        # These figures carry tens of thousands of real samples, so they are saved as PNG
        # rather than as the SVGs the explainer scripts produce.
        "savefig.dpi": 200,
    }
)

COLOR_TRACE = "#1f4e79"
COLOR_TRIAL = "#a8a8a8"
COLOR_MEAN = "#c0392b"
COLOR_EVENT = "#2c7a3a"


def run_pipeline(*, base_directory: Path) -> Path:
    """Copy the sample session into ``base_directory`` and run Steps 1-4 over it.

    Returns
    -------
    Path
        The run folder the four steps wrote into.
    """
    session = base_directory / SAMPLE_SESSION.name
    shutil.copytree(SAMPLE_SESSION, session)

    selected_folders = [str(session)]
    selected_runs = {str(session): [RUN_NAME]}

    step1(
        base_dir=str(base_directory),
        selected_folders=selected_folders,
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
        run_name=RUN_NAME,
    )
    step2(base_dir=str(base_directory), selected_folders=selected_folders, selected_runs=selected_runs)
    step3(base_dir=str(base_directory), selected_folders=selected_folders, selected_runs=selected_runs)
    step4(base_dir=str(base_directory), selected_folders=selected_folders, selected_runs=selected_runs)

    return session / f"{session.name}_output_{RUN_NAME}"


# --------------------------------------------------------------------------------------
# Everything below is the tutorial's own code. Keep the two in sync.
# --------------------------------------------------------------------------------------


def figure_session_trace(*, run_folder: Path) -> None:
    """Figure 1: the whole-session z-score trace with the five events marked."""
    with h5py.File(run_folder / "timeCorrection_A.hdf5", "r") as time_correction_file:
        timestamps = time_correction_file["timestampNew"][:]
        sampling_rate = time_correction_file["sampling_rate"][0]

    with h5py.File(run_folder / "z_score_A.hdf5", "r") as z_score_file:
        z_score = z_score_file["data"][:]

    with h5py.File(run_folder / "RewardPort_A.hdf5", "r") as event_file:
        event_times = event_file["ts"][:]

    figure, axes = plt.subplots(figsize=(7.5, 2.8), constrained_layout=True)
    axes.plot(timestamps, z_score, color=COLOR_TRACE, linewidth=0.5)
    for event_time in event_times:
        axes.axvline(event_time, color=COLOR_EVENT, linewidth=1.0, linestyle="--")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("z-score")
    axes.set_title(f"Recording site A — {sampling_rate:.1f} Hz, {len(event_times)} RewardPort events")
    figure.savefig(OUT / "session_trace.png")
    plt.close(figure)


def load_psth(*, run_folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the PSTH table and return it alongside its single-trial column labels."""
    psth = pd.read_hdf(run_folder / "RewardPort_A_z_score_A.h5", key="df")
    summary_columns = {"timestamps", "mean", "err"}
    trial_columns = [
        column for column in psth.columns if column not in summary_columns and not column.startswith("bin_")
    ]
    return psth, trial_columns


def figure_psth(*, run_folder: Path) -> None:
    """Figure 2: the trial mean with its SEM band, over the single trials."""
    psth, trial_columns = load_psth(run_folder=run_folder)

    figure, axes = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    for index, column in enumerate(trial_columns):
        axes.plot(
            psth["timestamps"],
            psth[column],
            color=COLOR_TRIAL,
            linewidth=0.7,
            label="single trials" if index == 0 else None,
        )
    axes.fill_between(
        psth["timestamps"],
        psth["mean"] - psth["err"],
        psth["mean"] + psth["err"],
        color=COLOR_MEAN,
        alpha=0.25,
        linewidth=0,
        label="SEM",
    )
    axes.plot(psth["timestamps"], psth["mean"], color=COLOR_MEAN, linewidth=1.8, label="mean")
    axes.axvline(0.0, color=COLOR_EVENT, linewidth=1.0, linestyle="--")
    axes.set_xlabel("Time from RewardPort (s)")
    axes.set_ylabel("z-score")
    axes.legend(frameon=False, fontsize=9)
    figure.savefig(OUT / "psth.png")
    plt.close(figure)


def figure_heatmap(*, run_folder: Path) -> None:
    """Figure 3: the same trials as a heatmap."""
    psth, trial_columns = load_psth(run_folder=run_folder)
    trials = psth[trial_columns].to_numpy().T

    figure, axes = plt.subplots(figsize=(6.0, 2.8), constrained_layout=True)
    image = axes.imshow(
        trials,
        aspect="auto",
        interpolation="nearest",
        cmap="plasma",
        extent=(psth["timestamps"].iloc[0], psth["timestamps"].iloc[-1], len(trial_columns) + 0.5, 0.5),
    )
    axes.axvline(0.0, color="white", linewidth=1.0, linestyle="--")
    axes.set_yticks(np.arange(len(trial_columns)) + 1)
    axes.set_xlabel("Time from RewardPort (s)")
    axes.set_ylabel("Trial")
    figure.colorbar(image, ax=axes, label="z-score")
    figure.savefig(OUT / "heatmap.png")
    plt.close(figure)


def print_peak_and_area(*, run_folder: Path) -> None:
    """Print the peak/AUC table, so the tutorial's quoted output stays honest."""
    peak_and_area = pd.read_csv(run_folder / "peak_AUC_RewardPort_A_z_score_A.csv", index_col=0)
    print(peak_and_area.to_string())


def main() -> None:
    """Run the pipeline over a throwaway copy of the sample session, then draw the figures."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_folder = run_pipeline(base_directory=Path(temporary_directory))
        print(f"Run folder: {run_folder}")

        figure_session_trace(run_folder=run_folder)
        print("Saved session_trace.png")
        figure_psth(run_folder=run_folder)
        print("Saved psth.png")
        figure_heatmap(run_folder=run_folder)
        print("Saved heatmap.png")
        print_peak_and_area(run_folder=run_folder)


if __name__ == "__main__":
    main()
