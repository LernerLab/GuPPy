"""Generate tutorial screenshots for the GuPPy documentation.

Run with:
    uv run --group test python docs/take_screenshots.py

Screenshots are saved to docs/_static/images/ and should be committed to the repository.
Re-run this script whenever the GUI changes.

Playwright browser binaries must be installed:
    uv run --group test playwright install chromium

The DANDI screenshots query the live DANDI Archive for the demo dandiset's asset
list, so this script needs network access. No API token is required: browsing a
public dandiset is unauthenticated (only streaming an asset's data authenticates).
"""

from __future__ import annotations

import math
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import panel as pn

from guppy.analysis.standard_io import write_tonic_to_hdf5
from guppy.analysis.tonic import compute_tonic_means
from guppy.frontend.artifact_windows_page import ArtifactWindowSelector
from guppy.frontend.covariate_correlation_view import build_covariate_correlation_view
from guppy.frontend.custom_events_config import CustomEventsConfig
from guppy.frontend.dandi_selector import DandiSelector
from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.frontend.group_labeling import GroupLabelingPage
from guppy.frontend.input_parameters import ParameterForm
from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.store_labeling_selector import StoreLabelingSelector
from guppy.frontend.tonic_epochs import TonicEpochConfig, TonicResultsView
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.orchestration.home import build_homepage
from guppy.orchestration.metadata import build_metadata_template
from guppy.orchestration.store_labeling import build_store_labeling_template
from guppy.testing.covariate_session import SESSION_NAME as COVARIATE_SESSION_NAME
from guppy.testing.covariate_session import run_covariate_session
from guppy.utils._hdf5_io import write_hdf5
from guppy.utils.nwb_metadata import (
    Channel,
    build_metadata_dict,
    load_yaml,
    parse_metadata_dict,
)

if TYPE_CHECKING:
    from panel.template.base import BasicTemplate
    from playwright.sync_api import Page

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "_static" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1280, "height": 900}
SAMPLE_DATA_DIR = REPO_ROOT / "stubbed_testing_data" / "csv" / "sample_data_csv_1"

# Public dandiset used for the DANDI how-to screenshots. The asset is the same one
# the live DANDI test pins, so the guide and the test describe the same recording.
DANDI_DEMO_DANDISET_ID = "000971"
DANDI_DEMO_SUBJECT = "sub-112-283"
DANDI_DEMO_ASSET = "sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"


def _wait_for_port(port: int, retries: int = 50, delay: float = 0.05) -> None:
    for _ in range(retries):
        try:
            socket.create_connection(("localhost", port), timeout=0.1).close()
            return
        except OSError:
            time.sleep(delay)
    raise RuntimeError(f"Server on port {port} did not start in time.")


def _serve(template: BasicTemplate) -> str:
    port = scanPortsAndFind()
    pn.serve(template, port=port, show=False, threaded=True)
    _wait_for_port(port)
    return f"http://localhost:{port}"


def _sidebar_clip(page: Page, from_step: str | None) -> dict[str, float]:
    """Clip rectangle over the sidebar, optionally starting at one step's label.

    Without ``from_step`` the crop runs from the top of the page. With it, the crop starts
    just above that step's label and ends below the Export to NWB progress bar, which is
    the last thing in the sidebar. Both edges are measured off the rendered page rather
    than hard-coded, since where a step lands depends on how many precede it.
    """
    viewport_height = page.viewport_size["height"]
    if from_step is None:
        return {"x": 0, "y": 0, "width": 360, "height": viewport_height}

    top = page.get_by_text(from_step).first.bounding_box()["y"] - 14
    # The export progress bar is the last thing in the sidebar; anchoring on it rather than
    # on the button above keeps the whole bar inside the crop whatever its height.
    export_progress = page.locator("progress").last
    export_progress.wait_for(timeout=15000)
    export_bar = export_progress.bounding_box()
    bottom = min(export_bar["y"] + export_bar["height"] + 16, viewport_height)
    return {"x": 0, "y": top, "width": 340, "height": bottom - top}


def screenshot_homepage(page: Page) -> None:
    """Screenshot 1: the Input Parameters GUI landing page."""
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(path=OUTPUT_DIR / "01_homepage.png", full_page=False)
    print("Saved 01_homepage.png")

    pn.state.kill_all_servers()


def screenshot_import_custom_events_button(page: Page) -> None:
    """How-to: the sidebar top showing the Import Custom Events button above Step 1."""
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "import_custom_events_button.png",
        clip={"x": 0, "y": 0, "width": 340, "height": 320},
    )
    print("Saved import_custom_events_button.png")

    pn.state.kill_all_servers()


def screenshot_select_artifact_windows_button(page: Page) -> None:
    """How-to: the sidebar showing the two optional artifact steps between Step 3 and Step 4."""
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "select_artifact_windows_button.png",
        clip={"x": 0, "y": 400, "width": 340, "height": 440},
    )
    print("Saved select_artifact_windows_button.png")

    pn.state.kill_all_servers()


def _synthetic_pair_traces(*, recording_sites: list[str]) -> dict[str, dict[str, object]]:
    """Build stand-in control/signal/fit traces containing one obvious artifact.

    Mirrors the shape ``load_pair_traces`` returns, so the marking page can be rendered
    without a preprocessed run folder on disk.
    """
    timestamps = np.arange(0.0, 300.0, 0.01)
    random_generator = np.random.default_rng(0)
    bleaching = 40.0 * np.exp(-timestamps / 260.0)
    # A knocked patchcord: both channels drop out together for a few seconds.
    dropout = -22.0 * np.exp(-(((timestamps - 134.0) / 3.0) ** 2))

    pair_traces = {}
    for site in recording_sites:
        transients = sum(3.0 * np.exp(-(((timestamps - onset) / 0.9) ** 2)) for onset in np.arange(12.0, 300.0, 17.0))
        fit = 100.0 + bleaching + dropout
        pair_traces[site] = {
            "x": timestamps,
            "control": fit + random_generator.normal(0.0, 0.6, timestamps.size),
            "signal": fit + transients + random_generator.normal(0.0, 0.6, timestamps.size),
            "fit": fit,
            "plot_name": [f"control_{site}", f"signal_{site}", f"cntrl_sig_fit_{site}"],
        }
    return pair_traces


def screenshot_select_artifact_windows(page: Page, tmp_path: Path) -> None:
    """How-to: the Select Artifact Windows page with the dropout marked as one period.

    The selector is built against synthetic traces rather than a preprocessed run folder
    so the screenshot does not depend on running Steps 1-3 first.
    """
    run_folder = tmp_path / "sample_data_csv_1_output_1"
    run_folder.mkdir(exist_ok=True)

    selector = ArtifactWindowSelector(str(run_folder), _synthetic_pair_traces(recording_sites=["DMS", "DLS"]))
    selector.set_windows("DMS", [(128.0, 140.0)])

    template = pn.template.BootstrapTemplate(title="GuPPy — Select Artifact Windows")
    template.main.append(selector.widget)
    url = _serve(template)

    # Render tall enough that every trace panel lays out (bokeh does not draw plots that
    # never enter the viewport), then clip to the content instead of the padded page.
    page.set_viewport_size({"width": 1280, "height": 1700})
    page.goto(url)
    page.get_by_text("Select Artifact Windows").first.wait_for()
    page.wait_for_timeout(3000)
    page.screenshot(
        path=OUTPUT_DIR / "select_artifact_windows.png",
        clip={"x": 0, "y": 0, "width": 1280, "height": 1390},
    )
    print("Saved select_artifact_windows.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


# Equal-length windows placed inside each phase, clear of the transitions at 60 s and 120 s —
# the placement the how-to recommends, so the shaded spans read as three separate windows.
TONIC_EPOCHS = [("baseline", 10.0, 50.0), ("drug", 70.0, 110.0), ("washout", 138.0, 178.0)]


def _synthetic_site_traces(*, recording_sites: list[str]) -> dict[str, dict[str, object]]:
    """Build stand-in z-score / dF/F traces for a bolus-injection recording.

    Mirrors the shape ``load_site_traces`` returns, and the three phases of the
    ``sample_data_csv_injection_1`` stub, so the epoch page can be rendered without a
    preprocessed run folder on disk and the picture matches the worked example in the docs.
    """
    timestamps = np.arange(0.0, 180.0, 0.01)
    random_generator = np.random.default_rng(0)
    # Held from the injection at 60 s, then cleared exponentially to a residual from 120 s.
    step = np.where(timestamps >= 60.0, 1.0, 0.0)
    decay = np.exp(-np.maximum(timestamps - 120.0, 0.0) / 6.0)
    drug_effect = step * np.where(timestamps >= 120.0, 0.25 + 0.75 * decay, 1.0)

    site_traces = {}
    for index, site in enumerate(recording_sites):
        amplitude = 3.0 - index  # sites respond at different magnitudes
        z_score = amplitude * drug_effect + random_generator.normal(0.0, 0.25, timestamps.size)
        site_traces[site] = {
            "x": timestamps,
            "y_zscore": z_score,
            "y_dff": z_score / 10.0,
        }
    return site_traces


def screenshot_tonic_analysis_button(page: Page) -> None:
    """How-to: the sidebar showing the optional tonic step between Remove Artifacts and Step 4."""
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)
    # The crop has to reach Step 4 to show where the optional step falls in the order, which
    # runs past the bottom of the default viewport — a clip beyond it is silently truncated.
    page.set_viewport_size({"width": 1280, "height": 1200})
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "tonic_analysis_button.png",
        clip={"x": 0, "y": 400, "width": 340, "height": 680},
    )
    print("Saved tonic_analysis_button.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_tonic_analysis(page: Page, tmp_path: Path) -> None:
    """How-to: the Tonic Analysis page with the three injection phases filled in."""
    run_folder = tmp_path / "sample_data_csv_injection_1_output_1"
    run_folder.mkdir(exist_ok=True)

    config = TonicEpochConfig(str(run_folder), _synthetic_site_traces(recording_sites=["DMS", "DLS"]))
    config.set_epochs("DMS", TONIC_EPOCHS)

    template = pn.template.BootstrapTemplate(title="GuPPy — Tonic Analysis")
    template.main.append(config.widget)
    url = _serve(template)

    # Render tall enough that both trace panels lay out (bokeh does not draw plots that
    # never enter the viewport), then clip to the content instead of the padded page.
    page.set_viewport_size({"width": 1280, "height": 1500})
    page.goto(url)
    page.get_by_text("Tonic Analysis").first.wait_for()
    page.wait_for_timeout(3000)
    page.screenshot(
        path=OUTPUT_DIR / "tonic_analysis.png",
        clip={"x": 0, "y": 0, "width": 1280, "height": 1060},
    )
    print("Saved tonic_analysis.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_tonic_results(page: Page, tmp_path: Path) -> None:
    """How-to: the Tonic tab of the visualization, showing each epoch's change from baseline.

    ``TonicResultsView`` reads everything from the run folder, so the preprocessed traces and
    the saved epochs are staged on disk first.
    """
    run_folder = tmp_path / "sample_data_csv_injection_1_output_2"
    run_folder.mkdir(exist_ok=True)

    site = "DMS"
    trace = _synthetic_site_traces(recording_sites=[site])[site]
    write_hdf5(trace["x"], "timeCorrection_" + site, str(run_folder), "timestampNew")
    write_hdf5(trace["y_zscore"], "z_score_" + site, str(run_folder), "data")
    write_hdf5(trace["y_dff"], "dff_" + site, str(run_folder), "data")

    epochs = pd.DataFrame(TONIC_EPOCHS, columns=["label", "start", "end"])
    epochs.to_csv(run_folder / f"tonic_epochs_{site}.csv", index=False)
    write_tonic_to_hdf5(
        str(run_folder),
        compute_tonic_means(trace["y_zscore"], trace["y_dff"], trace["x"], epochs),
        site,
    )

    template = pn.template.BootstrapTemplate(title="GuPPy — Visualization")
    template.main.append(TonicResultsView(str(run_folder)).widget)
    url = _serve(template)

    page.set_viewport_size({"width": 1280, "height": 1500})
    page.goto(url)
    page.get_by_text("Tonic / basal analysis").first.wait_for()
    page.wait_for_timeout(3000)
    page.screenshot(
        path=OUTPUT_DIR / "tonic_results.png",
        clip={"x": 0, "y": 0, "width": 1280, "height": 1220},
    )
    print("Saved tonic_results.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_covariate_label_stores(page: Page) -> None:
    """How-to: the Label Stores GUI with both behavioral covariates typed and named.

    Built directly rather than driven through the UI, for the reasons
    ``screenshot_label_stores_configured`` documents. The cached labels are the
    ``covariate_<name>`` strings Step 1 writes, which the config parses back into a
    "behavioral covariate" type and a name.
    """
    store_ids = ["Sample_Control_Channel", "Sample_Signal_Channel", "Sample_TTL", "akinesia", "grooming"]

    selector = StoreLabelingSelector(allnames=store_ids)
    selector.cross_selector.value = store_ids
    selector.set_change_widgets(store_ids)
    selector.store_ids = store_ids
    selector.configure_store_ids(
        store_id_to_store_labels={
            "Sample_Control_Channel": ["control_DMS"],
            "Sample_Signal_Channel": ["signal_DMS"],
            "Sample_TTL": ["ttl"],
            "akinesia": ["covariate_akinesia"],
            "grooming": ["covariate_grooming"],
        }
    )

    template = pn.template.BootstrapTemplate(title="Label Stores GUI - sample_data_csv_covariate_1")
    template.main.append(selector.widget)
    url = _serve(template)

    page.set_viewport_size({"width": 1280, "height": 1100})
    page.goto(url)
    page.get_by_text("Label Stores").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "covariate_label_stores.png",
        clip={"x": 0, "y": 0, "width": 660, "height": 1090},
    )
    print("Saved covariate_label_stores.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_covariate_correlations(page: Page, tmp_path: Path) -> None:
    """How-to: the Covariates tab of the visualization.

    Unlike every other screenshot here, this one runs the real pipeline. The view reads
    the binned metrics, binned covariates and correlations straight off disk, and those
    three tables are exactly what the how-to is describing, so staging fabricated ones
    would let the picture drift from what Step 4 actually writes.
    """
    base_directory = tmp_path / "covariate_run"
    base_directory.mkdir(exist_ok=True)
    run_folder = run_covariate_session(
        session_path=REPO_ROOT / "stubbed_testing_data" / "csv" / COVARIATE_SESSION_NAME,
        base_directory=base_directory,
    )

    template = pn.template.BootstrapTemplate(title="GuPPy — Visualization")
    template.main.append(build_covariate_correlation_view(run_folder))
    url = _serve(template)

    page.set_viewport_size({"width": 1280, "height": 2000})
    page.goto(url)
    page.get_by_text("Covariate").first.wait_for()
    page.wait_for_timeout(5000)
    page.screenshot(
        path=OUTPUT_DIR / "covariate_correlations.png",
        clip={"x": 0, "y": 0, "width": 1280, "height": 1560},
    )
    print("Saved covariate_correlations.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_import_custom_events(page: Page) -> None:
    """How-to: the Import Custom Events pop-out with two example events filled in.

    We build the config component directly and set the widget values before serving so the
    paste boxes render pre-filled (mirroring screenshot_label_stores_configured).
    """
    config = CustomEventsConfig()
    config.rows[0][0].value = "movement_onset"
    config.rows[0][1].value = "5.0\n9.5\n14.2\n21.0"
    config.add_event_row()
    config.rows[1][0].value = "reward_delivery"
    config.rows[1][1].value = "7.3\n12.1\n18.8"

    template = pn.template.BootstrapTemplate(title="Import Custom Events - sample_data_csv_1")
    template.main.append(config.widget)
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Import Custom Events").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(path=OUTPUT_DIR / "import_custom_events.png", full_page=False)
    print("Saved import_custom_events.png")

    pn.state.kill_all_servers()


def screenshot_label_stores(page: Page, tmp_path: Path) -> None:
    """Screenshot 2: the Label Stores GUI with CSV channel names."""
    # Pass the real sample-data directory so the page title reads
    # "Label Stores GUI - sample_data_csv_1" instead of leaking a tmp dir basename.
    template = build_store_labeling_template(
        events=["Sample_Control_Channel", "Sample_Signal_Channel", "Sample_TTL"],
        flags=[],
        folder_path=str(SAMPLE_DATA_DIR),
    )
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Select Stores").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(path=OUTPUT_DIR / "02_label_stores.png", full_page=False)
    print("Saved 02_label_stores.png")

    pn.state.kill_all_servers()


def screenshot_data_selection(page: Page) -> None:
    """Screenshot for Step 2 substep 1: the file-selector portion of the homepage.

    Captures the top of the Individual Analysis card so the reader can see the
    file browser they are about to interact with.
    """
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "02_data_selection.png",
        clip={"x": 0, "y": 0, "width": 1280, "height": 480},
    )
    print("Saved 02_data_selection.png")
    pn.state.kill_all_servers()


def screenshot_parameters(page: Page) -> None:
    """Screenshot for Step 2 substep 2: the parameter widgets in the Individual
    Analysis card.

    The homepage uses a sticky header and sticky sidebar, so window-level
    scrolling does not move the parameters into view. Instead we render with a
    tall viewport so the whole Individual Analysis card lays out without
    scrolling, then clip to the parameter region in absolute page coordinates.
    """
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    # The Individual Analysis card is collapsed by default; expand it so the
    # parameter widgets render and fall inside the clip region below.
    for card in template.main:
        if isinstance(card, pn.Card) and card.title == "Individual Analysis":
            card.collapsed = False
    url = _serve(template)
    page.set_viewport_size({"width": 1280, "height": 1800})
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "02_parameters.png",
        clip={"x": 0, "y": 600, "width": 1280, "height": 800},
    )
    print("Saved 02_parameters.png")
    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_label_groups_page(page: Page) -> None:
    """How-to: the Label Groups page, showing its member-runs and destination sections."""
    labeling_page = GroupLabelingPage(start_path=str(SAMPLE_DATA_DIR.parent), selected_group_folders=[])
    url = _serve(labeling_page.build_template())
    # The page lays out two 640px columns side by side, so it needs a wide viewport.
    page.set_viewport_size({"width": 1600, "height": 1300})
    page.goto(url)
    page.get_by_text("Group name").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "label_groups_page.png",
        clip={"x": 0, "y": 0, "width": 1600, "height": 1050},
    )
    print("Saved label_groups_page.png")
    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_group_psth_plot(page: Page, tmp_path: Path) -> None:
    """How-to: the Visualization dashboard's PSTH plot for a group.

    Mirrors screenshot_visualization, but names the data columns after session folders
    the way a group-averaged PSTH file does, and synthesizes event-evoked traces so the
    figure shows a real response shape with its across-session error band.
    """
    events = ["RewardPort"]
    sessions = ["session_1", "session_2", "session_3"]
    n_timepoints = 600
    timestamps = np.linspace(-10.0, 20.0, n_timepoints)
    # bin_1 / bin_err_1 keep the column count and trailing order the same shape the
    # dashboard's default-selection logic expects (mirrors screenshot_visualization);
    # they do not represent real group-average output, which has no bin_* columns.
    columns = [*sessions, "bin_1", "timestamps", "mean", "err", "bin_err_1"]

    def session_trace(*, amplitude: float, latency: float, seed: int) -> np.ndarray:
        """One session's averaged z-score: flat baseline, then an event-evoked transient."""
        random_generator = np.random.default_rng(seed)
        # Rise into the peak just after the event, then an exponential return to baseline.
        response = amplitude * np.exp(-(((timestamps - latency) / 1.4) ** 2))
        decay = 0.45 * amplitude * np.exp(-np.clip(timestamps - latency, 0.0, None) / 6.0)
        decay[timestamps < latency] = 0.0
        return response + decay + random_generator.normal(0.0, 0.16, n_timepoints)

    traces = {
        "session_1": session_trace(amplitude=2.6, latency=1.1, seed=0),
        "session_2": session_trace(amplitude=1.9, latency=1.5, seed=1),
        "session_3": session_trace(amplitude=3.1, latency=0.9, seed=2),
    }
    stacked = np.vstack([traces[name] for name in sessions])

    def make_df() -> pd.DataFrame:
        data = dict(traces)
        data["timestamps"] = timestamps
        # The group's mean and error bar are computed across member runs, as they are on disk.
        data["mean"] = stacked.mean(axis=0)
        data["err"] = stacked.std(axis=0) / math.sqrt(stacked.shape[0])
        data["bin_1"] = data["mean"]
        data["bin_err_1"] = data["err"]
        return pd.DataFrame({column: data[column] for column in columns})

    df_new = pd.concat([make_df() for _ in events], keys=events, axis=1)

    plotter = ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["All", *sessions, "mean"],
        heatmap_y_objects=[f"1 - {sessions[0]}", "All"],
        psth_y_objects=None,
        filepath=str(tmp_path),
        columns_dict={e: columns for e in events},
        df_new=df_new,
        x_min=-10.0,
        x_max=20.0,
    )
    dashboard = VisualizationDashboard(plotter=plotter, basename="average")
    template = dashboard.build_template()
    url = _serve(template)

    page.set_viewport_size({"width": 1280, "height": 1600})
    page.goto(url)
    page.get_by_text("PSTH").first.wait_for()
    page.wait_for_timeout(2500)
    # Clip to the rendered trace rather than the controls above it: the point of the
    # figure is the per-member-run curves, not the dashboard chrome.
    plot = page.locator("canvas").first
    plot.wait_for(timeout=15000)
    box = plot.bounding_box()
    page.screenshot(
        path=OUTPUT_DIR / "group_psth_plot.png",
        clip={"x": box["x"] - 60, "y": box["y"] - 20, "width": box["width"] + 90, "height": box["height"] + 60},
    )
    print("Saved group_psth_plot.png")
    page.set_viewport_size(VIEWPORT)

    pn.state.kill_all_servers()


def screenshot_sidebar_progress(
    page: Page,
    progress_index: int,
    output_name: str,
    *,
    viewport_height: int = 900,
    clip_from_step: str | None = None,
) -> None:
    """Screenshot the homepage sidebar with one progress bar mid-fill.

    The homepage sidebar contains five Progress indicators that fill from 0 to 100 as
    the corresponding pipeline step runs. We render the homepage with one of them
    pre-set to 60 to illustrate the in-progress state, then clip the screenshot to the
    leftmost 360 px so the result is just the sidebar.

    progress_index follows sidebar order: 0 = read raw data, 1 = preprocess,
    2 = remove artifacts, 3 = PSTH computation, 4 = export to NWB.

    The steps below the fold need a taller viewport to render at all, and a clip
    anchored on the step they belong to rather than on the top of the page — pass
    viewport_height and clip_from_step (the sidebar label to start the crop at) for those.
    """
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))

    progress_bars = [w for w in template.sidebar if isinstance(w, pn.indicators.Progress)]
    progress_bars[progress_index].value = 60

    url = _serve(template)
    page.set_viewport_size({"width": VIEWPORT["width"], "height": viewport_height})
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(path=OUTPUT_DIR / output_name, clip=_sidebar_clip(page, clip_from_step))
    print(f"Saved {output_name}")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_label_stores_configured(page: Page, tmp_path: Path) -> None:
    """Screenshot 2b: the Label Stores GUI after clicking Select Stores.

    Shows the Label Stores panel with one row per channel
    (Type dropdown + Name text field), which is the state the user
    needs to fill out.

    We build the configured state directly instead of driving the UI:
    pre-setting ``cross_selector.value`` does not propagate to served
    Panel sessions, and clicking the button via Playwright fires the
    callback while the cross-selector is empty, which leaves the
    Label Stores pane invisible. Constructing the selector
    here and calling ``configure_store_ids`` ourselves bypasses both.
    """
    events = ["Sample_Control_Channel", "Sample_Signal_Channel", "Sample_TTL"]

    selector = StoreLabelingSelector(allnames=events)
    selector.cross_selector.value = events
    selector.set_change_widgets(events)
    selector.store_ids = events
    selector.configure_store_ids(store_id_to_store_labels={})

    template = pn.template.BootstrapTemplate(title="Label Stores GUI - sample_data_csv_1")
    template.main.append(selector.widget)
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Label Stores").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(path=OUTPUT_DIR / "02b_label_stores_configured.png", full_page=False)
    print("Saved 02b_label_stores_configured.png")

    pn.state.kill_all_servers()


def screenshot_visualization(page: Page, tmp_path: Path) -> None:
    """Screenshot 3: the Visualization Dashboard with a synthetic PSTH."""
    events = ["RewardPort"]
    n_timepoints = 30
    timestamps = np.linspace(-10.0, 20.0, n_timepoints)
    columns = ["trial_1", "trial_2", "trial_3", "bin_1", "timestamps", "mean", "err", "bin_err_1"]

    def make_df() -> pd.DataFrame:
        return pd.DataFrame({col: (timestamps if col == "timestamps" else np.zeros(n_timepoints)) for col in columns})

    df_new = pd.concat([make_df() for _ in events], keys=events, axis=1)

    plotter = ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["trial_1", "mean"],
        heatmap_y_objects=["1 - trial_1", "All"],
        psth_y_objects=None,
        filepath=str(tmp_path),
        columns_dict={e: columns for e in events},
        df_new=df_new,
        x_min=-10.0,
        x_max=20.0,
    )
    dashboard = VisualizationDashboard(plotter=plotter, basename="sample_data_csv_1")
    template = dashboard.build_template()
    url = _serve(template)

    page.goto(url)
    page.get_by_text("PSTH").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(path=OUTPUT_DIR / "03_visualization.png", full_page=False)
    print("Saved 03_visualization.png")

    pn.state.kill_all_servers()


def screenshot_compare_parameters_run_name(page: Page) -> None:
    """Screenshot for the compare-parameters tutorial: the save section with a named run.

    Only the save block is in shot, with **Run name** filled in, since naming a run
    after the parameter it varies is the point of that guide.
    """
    selector = StoreLabelingSelector(allnames=["Sample_Control_Channel"])
    selector.run_name.value = "filter_250"

    template = pn.template.BootstrapTemplate(title="Label Stores GUI - sample_data_csv_1")
    template.main.append(pn.Column(selector.mark_down_for_overwrite, selector.overwrite_button, selector.run_name))
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Choose how to save this store_array").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "compare_parameters_run_name.png",
        clip={"x": 0, "y": 0, "width": 660, "height": 320},
    )
    print("Saved compare_parameters_run_name.png")


def screenshot_dandi_source_selection(page: Page) -> None:
    """How-to: Input Folder Selection with the Data Source toggle set to ``dandi``.

    Assigning ``source_mode`` and ``dandiset_input`` fires their param watchers
    synchronously, so the DANDI panel is swapped in and the dandiset's assets are
    fetched before the template is served (mirroring screenshot_label_stores_configured).
    Fetching the asset list touches one zero-byte placeholder per NWB asset under the
    system temp dir; the tree is reused on subsequent runs.
    """
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    template._widgets["source_mode"].value = "dandi"
    template._widgets["dandi_selector"].dandiset_input.value = DANDI_DEMO_DANDISET_ID
    url = _serve(template)

    page.goto(url)
    page.get_by_text("DANDI source").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "dandi_source_selection.png",
        clip={"x": 0, "y": 80, "width": 1280, "height": 355},
    )
    print("Saved dandi_source_selection.png")

    pn.state.kill_all_servers()


def screenshot_compare_parameters_existing_runs(page: Page) -> None:
    """Screenshot for the compare-parameters tutorial: the run folders to choose between.

    The three runs match what a reader following the tutorial sees: the unnamed run left
    by the first-analysis tutorial, plus the two named ones this tutorial builds. The run
    folders are created inside the real sample-data directory, and removed afterwards, so
    the Directory field shows a normal session path rather than a temp-dir basename.
    """
    run_names = ("1", "filter_100", "filter_250")
    run_folders = [SAMPLE_DATA_DIR / f"sample_data_csv_1_output_{name}" for name in run_names]
    for run_folder in run_folders:
        run_folder.mkdir(exist_ok=True)

    try:
        # Drive the card directly rather than through the homepage: selecting a run
        # in a served FileSelector takes several dependent clicks, and only this one
        # card is in shot. Assigning the inner cross-selector's value is what moves
        # an entry into the "Selected files" pane; setting FileSelector.value alone
        # updates the parameter without redrawing the panes.
        form = ParameterForm(
            template=pn.template.MaterialTemplate(title="Input Parameters GUI"),
            start_path=str(SAMPLE_DATA_DIR.parent),
        )
        form.outputs_selector._directory.value = str(SAMPLE_DATA_DIR)
        form.outputs_selector._update_files()
        form.outputs_selector._selector.value = [str(run_folders[2])]
        form.output_folder_selection.collapsed = False

        template = pn.template.MaterialTemplate(title="Input Parameters GUI")
        template.main.append(form.output_folder_selection)
        url = _serve(template)

        page.goto(url)
        page.get_by_text("Existing runs (steps 2–5)").first.wait_for()
        page.wait_for_timeout(1500)
        page.screenshot(
            path=OUTPUT_DIR / "compare_parameters_existing_runs.png",
            clip={"x": 0, "y": 0, "width": 1060, "height": 520},
        )
        print("Saved compare_parameters_existing_runs.png")

        pn.state.kill_all_servers()
    finally:
        for run_folder in run_folders:
            run_folder.rmdir()


def screenshot_dandi_asset_browser(page: Page) -> None:
    """How-to: the DANDI asset browser descended into a subject folder, one asset selected.

    Built from the component directly rather than the homepage so the browser is not
    pushed down by the sidebar and card chrome. ``FileSelector`` computes its
    selected/unselected lists at construction, so the widget is built with ``value``
    already set and swapped into the selector's slot rather than assigned afterwards.
    """
    selector = DandiSelector()
    selector.dandiset_input.value = DANDI_DEMO_DANDISET_ID
    subject_directory = os.path.join(selector._current_mirror_root, DANDI_DEMO_SUBJECT)
    file_selector = pn.widgets.FileSelector(
        subject_directory,
        root_directory=selector._current_mirror_root,
        file_pattern="*.nwb",
        name="NWB assets",
        value=[os.path.join(subject_directory, DANDI_DEMO_ASSET)],
        width=950,
    )
    file_selector._directory.visible = False
    selector._asset_file_selector_slot[:] = [file_selector]

    template = pn.template.BootstrapTemplate(title="Input Folder Selection - DANDI source")
    template.main.append(selector.panel)
    url = _serve(template)

    page.set_viewport_size({"width": 1280, "height": 1700})
    page.goto(url)
    page.get_by_text("DANDI source").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "dandi_asset_browser.png",
        clip={"x": 0, "y": 335, "width": 1130, "height": 430},
    )
    print("Saved dandi_asset_browser.png")
    page.set_viewport_size(VIEWPORT)

def screenshot_export_to_nwb_button(page: Page) -> None:
    """How-to: the sidebar bottom showing the two optional NWB steps below Step 5."""
    template = build_homepage(start_path=str(SAMPLE_DATA_DIR.parent))
    url = _serve(template)
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 1400})
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / "export_to_nwb_button.png",
        clip=_sidebar_clip(page, "Step 5 : Visualization"),
    )
    print("Saved export_to_nwb_button.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def _metadata_template() -> BasicTemplate:
    """Build the Step 6 form pre-filled from the example metadata, for the how-to figures.

    ``build_metadata_template`` takes its channels as an argument, so the form renders without
    a pipeline run behind it: the two channels below stand in for what ``derive_channels`` would
    read out of a session's ``storesList.csv``.
    """
    channels = [Channel("dms", "control", "Dv1A"), Channel("dms", "signal", "Dv2A")]
    example = load_yaml(str(REPO_ROOT / "tests" / "data" / "fiber_photometry_metadata_example.yaml"))
    devices, _rows, scalars = parse_metadata_dict(metadata=example, channels=channels)
    channel_rows = [
        {
            "excitation_wavelength_in_nm": 405.0,
            "emission_wavelength_in_nm": 525.0,
            "indicator": "dms_green_fluorophore",
            "optical_fiber": "optical_fiber",
            "excitation_source": "excitation_source_isosbestic_control",
            "photodetector": "photodetector",
        },
        {
            "excitation_wavelength_in_nm": 465.0,
            "emission_wavelength_in_nm": 525.0,
            "indicator": "dms_green_fluorophore",
            "optical_fiber": "optical_fiber",
            "excitation_source": "excitation_source_calcium_signal",
            "photodetector": "photodetector",
        },
    ]
    scalars = {
        **scalars,
        "session_description": "Fiber photometry recording during RI30 training.",
        "session_start_time": "2018-10-30T10:33:32-05:00",
        "lab": "Lerner Lab",
        "institution": "Northwestern University",
        "subject_id": "Photo_63_207",
        "sex": "M",
        "species": "Mus musculus",
        "experimenter": ["Doe, Jane"],
    }
    return build_metadata_template(
        session_label="Photo_63_207 (1)",
        channels=channels,
        metadata=build_metadata_dict(
            devices=devices, channel_rows=channel_rows, scalars=scalars, channels=channels
        ),
        metadata_yaml_path=str(SAMPLE_DATA_DIR / "nwb_metadata.yaml"),
    )


def _card_top(page: Page, title: str) -> float:
    """Y coordinate of a Panel Card's header, located by its title.

    Matched on the header element rather than the title text, because the form's intro
    paragraph names several of the cards in bold and would otherwise match first.
    """
    header = page.locator(".card-header").filter(has_text=title).first
    header.wait_for(timeout=15000)
    return header.bounding_box()["y"]


def screenshot_input_metadata(page: Page) -> None:
    """How-to: the top of the Step 6 form, down to the end of the Core NWB metadata card."""
    pn.extension(notifications=True)
    url = _serve(_metadata_template())
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 1400})
    page.goto(url)
    page.get_by_text("NWB metadata").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "input_metadata.png",
        clip={"x": 0, "y": 0, "width": VIEWPORT["width"], "height": _card_top(page, "Optical hardware") - 8},
    )
    print("Saved input_metadata.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_input_metadata_channels(page: Page) -> None:
    """How-to: the Step 6 channel section.

    The channel card sits well below the fold, so the card element is screenshotted
    directly: that scrolls it into view and sizes the image to the card, where a clipped
    page screenshot would have to address a point outside the viewport.
    """
    pn.extension(notifications=True)
    url = _serve(_metadata_template())
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 1400})
    page.goto(url)
    page.get_by_text("NWB metadata").first.wait_for()
    page.wait_for_timeout(1500)
    card = page.locator(".card").filter(has_text="Fiber-photometry channels").first
    card.wait_for(timeout=15000)
    card.screenshot(path=OUTPUT_DIR / "input_metadata_channels.png", timeout=30000)
    print("Saved input_metadata_channels.png")

    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def main() -> None:
    """Launch a headless browser and regenerate every tutorial screenshot in order."""
    import tempfile

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            screenshot_homepage(page)
            screenshot_import_custom_events_button(page)
            screenshot_import_custom_events(page)
            screenshot_data_selection(page)
            screenshot_parameters(page)
            screenshot_dandi_source_selection(page)
            screenshot_dandi_asset_browser(page)
            screenshot_label_stores(page, tmp_path)
            screenshot_label_stores_configured(page, tmp_path)
            screenshot_sidebar_progress(page, 0, "04_read_progress.png")
            screenshot_sidebar_progress(page, 1, "05_preprocess_progress.png")
            screenshot_sidebar_progress(page, 3, "06_psth_progress.png")
            screenshot_select_artifact_windows_button(page)
            screenshot_select_artifact_windows(page, tmp_path)
            screenshot_tonic_analysis_button(page)
            screenshot_tonic_analysis(page, tmp_path)
            screenshot_tonic_results(page, tmp_path)
            screenshot_covariate_label_stores(page)
            screenshot_covariate_correlations(page, tmp_path)
            screenshot_visualization(page, tmp_path)
            screenshot_compare_parameters_run_name(page)
            screenshot_compare_parameters_existing_runs(page)
            screenshot_label_groups_page(page)
            screenshot_group_psth_plot(page, tmp_path)
            screenshot_export_to_nwb_button(page)
            screenshot_sidebar_progress(
                page,
                4,
                "export_to_nwb_progress.png",
                viewport_height=1400,
                clip_from_step="Step 5 : Visualization",
            )
            screenshot_input_metadata(page)
            screenshot_input_metadata_channels(page)

        browser.close()

    print(f"\nAll screenshots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
