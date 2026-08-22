"""Generate tutorial screenshots for the GuPPy documentation.

Run with:
    uv run --group test python docs/take_screenshots.py

Screenshots are saved to docs/_static/images/ and should be committed to the repository.
Re-run this script whenever the GUI changes.

Playwright browser binaries must be installed:
    uv run --group test playwright install chromium
"""

from __future__ import annotations

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
from guppy.frontend.custom_events_config import CustomEventsConfig
from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.store_labeling_selector import StoreLabelingSelector
from guppy.frontend.tonic_epochs import TonicEpochConfig, TonicResultsView
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.orchestration.home import build_homepage
from guppy.orchestration.metadata import build_metadata_template
from guppy.orchestration.store_labeling import build_store_labeling_template
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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
    url = _serve(template)

    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(path=OUTPUT_DIR / "01_homepage.png", full_page=False)
    print("Saved 01_homepage.png")

    pn.state.kill_all_servers()


def screenshot_import_custom_events_button(page: Page) -> None:
    """How-to: the sidebar top showing the Import Custom Events button above Step 1."""
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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
        clip={"x": 0, "y": 0, "width": 1280, "height": 1310},
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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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


def screenshot_group_analysis_card(page: Page) -> None:
    """How-to: the Group Analysis card, expanded, with its folder browser and toggle.

    The card is collapsed by default, so expand it before rendering, then clip to its
    region — mirrors screenshot_parameters.
    """
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
    for card in template.main:
        if isinstance(card, pn.Card) and card.title == "Group Analysis":
            card.collapsed = False
    url = _serve(template)
    page.set_viewport_size({"width": 1280, "height": 1800})
    page.goto(url)
    page.get_by_text("Group Analysis").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "group_analysis_card.png",
        clip={"x": 0, "y": 620, "width": 1280, "height": 750},
    )
    print("Saved group_analysis_card.png")
    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_visualize_average_toggle(page: Page) -> None:
    """How-to: the Visualization Parameters card showing the Visualize Average Results? toggle."""
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
    for card in template.main:
        if isinstance(card, pn.Card) and card.title == "Visualization Parameters":
            card.collapsed = False
    url = _serve(template)
    page.set_viewport_size({"width": 1280, "height": 1800})
    page.goto(url)
    page.get_by_text("Visualization Parameters").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(
        path=OUTPUT_DIR / "visualize_average_results_toggle.png",
        clip={"x": 0, "y": 600, "width": 1280, "height": 300},
    )
    print("Saved visualize_average_results_toggle.png")
    page.set_viewport_size(VIEWPORT)
    pn.state.kill_all_servers()


def screenshot_group_psth_plot(page: Page, tmp_path: Path) -> None:
    """How-to: the Visualization dashboard's PSTH plot with one trace per session.

    Mirrors screenshot_visualization, but names the data columns like session folders
    rather than trials, matching the real shape of a group-averaged PSTH file.
    """
    events = ["RewardPort"]
    sessions = ["session_1", "session_2", "session_3"]
    n_timepoints = 30
    timestamps = np.linspace(-10.0, 20.0, n_timepoints)
    # bin_1 / bin_err_1 keep the column count and trailing order the same shape the
    # dashboard's default-selection logic expects (mirrors screenshot_visualization);
    # they do not represent real group-average output, which has no bin_* columns.
    columns = [*sessions, "bin_1", "timestamps", "mean", "err", "bin_err_1"]

    def make_df() -> pd.DataFrame:
        return pd.DataFrame({col: (timestamps if col == "timestamps" else np.zeros(n_timepoints)) for col in columns})

    df_new = pd.concat([make_df() for _ in events], keys=events, axis=1)

    plotter = ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=[*sessions, "mean"],
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

    page.goto(url)
    page.get_by_text("PSTH").first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(path=OUTPUT_DIR / "group_psth_plot.png", full_page=False)
    print("Saved group_psth_plot.png")

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
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()

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


def screenshot_export_to_nwb_button(page: Page) -> None:
    """How-to: the sidebar bottom showing the two optional NWB steps below Step 5."""
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()
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
            screenshot_visualization(page, tmp_path)
            screenshot_group_analysis_card(page)
            screenshot_visualize_average_toggle(page)
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
