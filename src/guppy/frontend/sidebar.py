import logging

import panel as pn

logger = logging.getLogger(__name__)


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

    def setup_markdown(self) -> None:
        """Create step-label ``Markdown`` panes and store them as instance attributes."""
        self.mark_down_storenames = pn.pane.Markdown(
            """**Step 1 : Open Storenames GUI <br> and save storenames**""", width=300
        )
        self.mark_down_read = pn.pane.Markdown("""**Step 2 : Read Raw Data**""", width=300)
        self.mark_down_preprocess = pn.pane.Markdown("""**Step 3 : Preprocess and Remove Artifacts**""", width=300)
        self.mark_down_psth = pn.pane.Markdown("""**Step 4 : PSTH Computation**""", width=300)
        self.mark_down_visualization = pn.pane.Markdown("""**Step 5 : Visualization**""", width=300)

    def setup_buttons(self) -> None:
        """Create pipeline-step action buttons and store them as instance attributes."""
        self.open_storenames = pn.widgets.Button(
            name="Open Storenames GUI", button_type="primary", width=300, align="end"
        )
        self.read_rawData = pn.widgets.Button(name="Read Raw Data", button_type="primary", width=300, align="end")
        self.preprocess = pn.widgets.Button(
            name="Preprocess and Remove Artifacts", button_type="primary", width=300, align="end"
        )
        self.psth_computation = pn.widgets.Button(
            name="PSTH Computation", button_type="primary", width=300, align="end"
        )
        self.open_visualization = pn.widgets.Button(
            name="Open Visualization GUI", button_type="primary", width=300, align="end"
        )

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

    def add_to_template(self) -> None:
        """Append all sidebar widgets to the template's sidebar area in pipeline order."""
        self.template.sidebar.append(self.mark_down_storenames)
        self.template.sidebar.append(self.open_storenames)
        self.template.sidebar.append(self.mark_down_read)
        self.template.sidebar.append(self.read_rawData)
        self.template.sidebar.append(self.read_progress)
        self.template.sidebar.append(self.mark_down_preprocess)
        self.template.sidebar.append(self.preprocess)
        self.template.sidebar.append(self.extract_progress)
        self.template.sidebar.append(self.mark_down_psth)
        self.template.sidebar.append(self.psth_computation)
        self.template.sidebar.append(self.psth_progress)
        self.template.sidebar.append(self.mark_down_visualization)
        self.template.sidebar.append(self.open_visualization)
