import html
import logging

import panel as pn

from .tab_opener import open_placeholder_tab_on_click
from ..utils.page_links import PageLink

logger = logging.getLogger(__name__)

# Buttons whose click opens a page in a new tab straight away.
PAGE_OPENING_BUTTONS = (
    "import_custom_events",
    "open_label_stores",
    "select_artifact_windows",
    "tonic_analysis",
    "label_groups",
    "open_visualization",
    "open_metadata",
)

# Buttons for long-running steps, whose results page opens from a Show results button beside them.
RESULTS_BUTTONS = ("preprocess", "remove_artifacts", "psth_computation")

_RESULTS_BUTTON_STYLE = (
    "display: inline-block; box-sizing: border-box; width: 100%; padding: 5px 0; border: 1px solid; "
    "border-radius: 4px; text-align: center; text-decoration: none; font-size: 13px; white-space: nowrap;"
)


class Sidebar:
    """Panel sidebar component containing step labels, action buttons, and progress bars.

    Parameters
    ----------
    template : panel.template.base.BasicTemplate
        The Panel template whose ``sidebar`` area will receive the widgets.
    """

    def __init__(self, template: object) -> None:
        self.template = template
        self.setup_markdown()
        self.setup_buttons()
        self.setup_progress_bars()
        self.setup_results_buttons()
        for button_name in PAGE_OPENING_BUTTONS:
            open_placeholder_tab_on_click(button=getattr(self, button_name), key=button_name)

    def setup_markdown(self) -> None:
        """Create step-label ``Markdown`` panes and store them as instance attributes."""
        self.mark_down_import_events = pn.pane.Markdown("""**Import Custom Events (optional)**""", width=300)
        self.mark_down_label_stores = pn.pane.Markdown("""**Step 1 : Label Stores**""", width=300)
        self.mark_down_read = pn.pane.Markdown("""**Step 2 : Read Raw Data**""", width=300)
        self.mark_down_preprocess = pn.pane.Markdown("""**Step 3 : Preprocess**""", width=300)
        self.mark_down_select_windows = pn.pane.Markdown("""**Select Artifact Windows (optional)**""", width=300)
        self.mark_down_remove_artifacts = pn.pane.Markdown("""**Remove Artifacts (optional)**""", width=300)
        self.mark_down_tonic_analysis = pn.pane.Markdown("""**Tonic Analysis (optional)**""", width=300)
        self.mark_down_psth = pn.pane.Markdown("""**Step 4 : PSTH Computation**""", width=300)
        self.mark_down_label_groups = pn.pane.Markdown("""**Label Groups (optional)**""", width=300)
        self.mark_down_group_analysis = pn.pane.Markdown("""**Group Analysis (optional)**""", width=300)
        self.mark_down_visualization = pn.pane.Markdown("""**Step 5 : Visualization**""", width=300)
        self.mark_down_metadata = pn.pane.Markdown("""**Step 6 : Input Metadata**""", width=300)
        self.mark_down_export_nwb = pn.pane.Markdown("""**Step 7 : Export to NWB**""", width=300)

    def setup_buttons(self) -> None:
        """Create pipeline-step action buttons and store them as instance attributes."""
        self.import_custom_events = pn.widgets.Button(
            name="Import Custom Events", button_type="default", width=300, align="end"
        )
        self.open_label_stores = pn.widgets.Button(name="Label Stores", button_type="primary", width=300, align="end")
        self.read_rawData = pn.widgets.Button(name="Read Raw Data", button_type="primary", width=300, align="end")
        self.preprocess = pn.widgets.Button(name="Preprocess", button_type="primary", width=300, align="end")
        self.select_artifact_windows = pn.widgets.Button(
            name="Select Artifact Windows", button_type="default", width=300, align="end"
        )
        self.remove_artifacts = pn.widgets.Button(
            name="Remove Artifacts", button_type="default", width=300, align="end"
        )
        self.tonic_analysis = pn.widgets.Button(name="Tonic Analysis", button_type="default", width=300, align="end")
        self.psth_computation = pn.widgets.Button(
            name="PSTH Computation", button_type="primary", width=300, align="end"
        )
        self.label_groups = pn.widgets.Button(name="Label Groups", button_type="default", width=300, align="end")
        self.group_analysis = pn.widgets.Button(name="Group Analysis", button_type="default", width=300, align="end")
        self.open_visualization = pn.widgets.Button(
            name="Open Visualization GUI", button_type="primary", width=300, align="end"
        )
        self.open_metadata = pn.widgets.Button(name="Input Metadata", button_type="primary", width=300, align="end")
        self.export_nwb = pn.widgets.Button(name="Export to NWB", button_type="primary", width=300, align="end")

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on sidebar buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (e.g. ``"read_rawData"``) to the
            callable that should be invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            button.on_click(onclick_fn)

    def setup_progress_bars(self) -> None:
        """Create ``Progress`` indicator widgets for the read, extract, and PSTH steps."""
        self.read_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.extract_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.psth_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.remove_artifacts_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.export_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.group_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)

    def setup_results_buttons(self) -> None:
        """Create a Show results button, disabled until there are results, for each long-running step."""
        self.results_buttons = {
            button_name: pn.pane.HTML(width=100, margin=(5, 10, 5, 0)) for button_name in RESULTS_BUTTONS
        }
        for button_name in RESULTS_BUTTONS:
            getattr(self, button_name).width = 190
            self.show_results(button_name=button_name, link=None)

    def show_results(self, *, button_name: str, link: PageLink | None) -> None:
        """Point a step's Show results button at its results page, or disable it.

        Parameters
        ----------
        button_name : str
            The long-running step's button, e.g. ``"preprocess"``.
        link : PageLink or None
            The results page, which the button opens in a new tab; ``None`` disables the button.
        """
        if link is None:
            self.results_buttons[button_name].object = (
                f'<span title="Available once the step has run" style="{_RESULTS_BUTTON_STYLE} '
                f'color: #adb5bd; border-color: #dee2e6;">Show results</span>'
            )
        else:
            self.results_buttons[button_name].object = (
                f'<a href="{html.escape(link.href)}" target="_blank" '
                f'title="Opens {html.escape(link.title)} in a new tab" '
                f'style="{_RESULTS_BUTTON_STYLE} color: #0072b5; border-color: #0072b5;">Show results</a>'
            )

    def add_to_template(self) -> None:
        """Append all sidebar widgets to the template's sidebar area in pipeline order."""
        self.template.sidebar.append(self.mark_down_import_events)
        self.template.sidebar.append(self.import_custom_events)
        self.template.sidebar.append(self.mark_down_label_stores)
        self.template.sidebar.append(self.open_label_stores)
        self.template.sidebar.append(self.mark_down_read)
        self.template.sidebar.append(self.read_rawData)
        self.template.sidebar.append(self.read_progress)
        self.template.sidebar.append(self.mark_down_preprocess)
        self.template.sidebar.append(pn.Row(self.preprocess, self.results_buttons["preprocess"]))
        self.template.sidebar.append(self.extract_progress)
        self.template.sidebar.append(self.mark_down_select_windows)
        self.template.sidebar.append(self.select_artifact_windows)
        self.template.sidebar.append(self.mark_down_remove_artifacts)
        self.template.sidebar.append(pn.Row(self.remove_artifacts, self.results_buttons["remove_artifacts"]))
        self.template.sidebar.append(self.remove_artifacts_progress)
        self.template.sidebar.append(self.mark_down_tonic_analysis)
        self.template.sidebar.append(self.tonic_analysis)
        self.template.sidebar.append(self.mark_down_psth)
        self.template.sidebar.append(pn.Row(self.psth_computation, self.results_buttons["psth_computation"]))
        self.template.sidebar.append(self.psth_progress)
        self.template.sidebar.append(self.mark_down_label_groups)
        self.template.sidebar.append(self.label_groups)
        self.template.sidebar.append(self.mark_down_group_analysis)
        self.template.sidebar.append(self.group_analysis)
        self.template.sidebar.append(self.group_progress)
        self.template.sidebar.append(self.mark_down_visualization)
        self.template.sidebar.append(self.open_visualization)
        self.template.sidebar.append(self.mark_down_metadata)
        self.template.sidebar.append(self.open_metadata)
        self.template.sidebar.append(self.mark_down_export_nwb)
        self.template.sidebar.append(self.export_nwb)
        self.template.sidebar.append(self.export_progress)
