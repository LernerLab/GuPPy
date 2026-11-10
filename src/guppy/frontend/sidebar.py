import logging
from collections.abc import Callable

import panel as pn

from .step_status import StepStatus

logger = logging.getLogger(__name__)

# Every step's button, in pipeline order.
STEP_NAMES = (
    "import_custom_events",
    "open_label_stores",
    "read_rawData",
    "preprocess",
    "select_artifact_windows",
    "remove_artifacts",
    "tonic_analysis",
    "psth_computation",
    "label_groups",
    "group_analysis",
    "open_visualization",
    "open_metadata",
    "export_nwb",
)

# Steps that run in the background behind a progress bar.
PROGRESS_STEPS = ("read_rawData", "preprocess", "remove_artifacts", "psth_computation", "group_analysis", "export_nwb")

# Long-running steps with a results page, opened from a View results button once they are done.
RESULTS_STEPS = ("read_rawData", "preprocess", "remove_artifacts", "psth_computation")


class Sidebar:
    """Panel sidebar component containing step labels, action buttons, and each step's status.

    Parameters
    ----------
    template : panel.template.base.BasicTemplate
        The Panel template whose ``sidebar`` area will receive the widgets.
    """

    def __init__(self, template: object) -> None:
        self.template = template
        self.setup_markdown()
        self.setup_buttons()
        self.setup_statuses()

    def setup_markdown(self) -> None:
        """Create step-label ``Markdown`` panes and store them as instance attributes."""
        self.mark_down_import_events = pn.pane.Markdown("""**Import Custom Events** (optional)""", width=300)
        self.mark_down_label_stores = pn.pane.Markdown("""**Step 1: Label Stores**""", width=300)
        self.mark_down_read = pn.pane.Markdown("""**Step 2: Read Raw Data**""", width=300)
        self.mark_down_preprocess = pn.pane.Markdown("""**Step 3: Preprocess**""", width=300)
        self.mark_down_select_windows = pn.pane.Markdown("""**Select Artifact Windows** (optional)""", width=300)
        self.mark_down_remove_artifacts = pn.pane.Markdown("""**Remove Artifacts** (optional)""", width=300)
        self.mark_down_tonic_analysis = pn.pane.Markdown("""**Tonic Analysis** (optional)""", width=300)
        self.mark_down_psth = pn.pane.Markdown("""**Step 4: PSTH Computation**""", width=300)
        self.mark_down_label_groups = pn.pane.Markdown("""**Label Groups** (optional)""", width=300)
        self.mark_down_group_analysis = pn.pane.Markdown("""**Group Analysis** (optional)""", width=300)
        self.mark_down_visualization = pn.pane.Markdown("""**Step 5: Visualization**""", width=300)
        self.mark_down_metadata = pn.pane.Markdown("""**Step 6: Input Metadata**""", width=300)
        self.mark_down_export_nwb = pn.pane.Markdown("""**Step 7: Export to NWB**""", width=300)

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

    def setup_statuses(self) -> None:
        """Create the status line shown under every step's button."""
        self.statuses = {
            step_name: StepStatus(has_progress=step_name in PROGRESS_STEPS, has_results=step_name in RESULTS_STEPS)
            for step_name in STEP_NAMES
        }
        self.results_buttons = {step_name: self.statuses[step_name].results_button for step_name in RESULTS_STEPS}
        self._open_results: dict[str, Callable[[], None] | None] = {}
        for step_name in RESULTS_STEPS:
            self.results_buttons[step_name].on_click(lambda event, step_name=step_name: self._open_results[step_name]())

    def show_results(self, *, button_name: str, open_results: Callable[[], None] | None) -> None:
        """Point a step's View results button at what opens its results page, or hide it.

        Parameters
        ----------
        button_name : str
            The long-running step's button, e.g. ``"preprocess"``.
        open_results : callable or None
            Called with no arguments when the button is clicked; ``None`` hides the button.
        """
        self._open_results[button_name] = open_results
        self.results_buttons[button_name].visible = open_results is not None

    def add_to_template(self) -> None:
        """Append every step's label, button, and status to the template's sidebar in pipeline order."""
        labels = (
            self.mark_down_import_events,
            self.mark_down_label_stores,
            self.mark_down_read,
            self.mark_down_preprocess,
            self.mark_down_select_windows,
            self.mark_down_remove_artifacts,
            self.mark_down_tonic_analysis,
            self.mark_down_psth,
            self.mark_down_label_groups,
            self.mark_down_group_analysis,
            self.mark_down_visualization,
            self.mark_down_metadata,
            self.mark_down_export_nwb,
        )
        for label, step_name in zip(labels, STEP_NAMES, strict=True):
            self.template.sidebar.extend([label, getattr(self, step_name), self.statuses[step_name].layout])
