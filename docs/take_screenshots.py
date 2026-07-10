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

from guppy.frontend.custom_events_config import CustomEventsConfig
from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.store_labeling_selector import StoreLabelingSelector
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.orchestration.home import build_homepage
from guppy.orchestration.store_labeling import build_store_labeling_template

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


def screenshot_sidebar_progress(page: Page, progress_index: int, output_name: str) -> None:
    """Screenshot the homepage sidebar with one progress bar mid-fill.

    The homepage sidebar contains three Progress indicators (read raw data,
    preprocess, PSTH) that fill from 0 to 100 as the corresponding pipeline
    step runs. We render the homepage with one of them pre-set to 60 to
    illustrate the in-progress state, then clip the screenshot to the
    leftmost 360 px so the result is just the sidebar.

    progress_index: 0 = read raw data, 1 = preprocess, 2 = PSTH computation.
    """
    os.environ["GUPPY_BASE_DIR"] = str(SAMPLE_DATA_DIR.parent)
    template = build_homepage()

    progress_bars = [w for w in template.sidebar if isinstance(w, pn.indicators.Progress)]
    progress_bars[progress_index].value = 60

    url = _serve(template)
    page.goto(url)
    page.get_by_text("Individual Analysis").first.wait_for()
    page.wait_for_timeout(1000)
    page.screenshot(
        path=OUTPUT_DIR / output_name,
        clip={"x": 0, "y": 0, "width": 360, "height": 900},
    )
    print(f"Saved {output_name}")

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
    selector.configure_store_ids(
        store_id_dropdowns={},
        store_id_textboxes={},
        store_ids=events,
        store_id_to_store_labels={},
    )

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
        return pd.DataFrame(
            {col: (timestamps if col == "timestamps" else np.zeros(n_timepoints)) for col in columns}
        )

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
            screenshot_sidebar_progress(page, 2, "06_psth_progress.png")
            screenshot_visualization(page, tmp_path)

        browser.close()

    print(f"\nAll screenshots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
