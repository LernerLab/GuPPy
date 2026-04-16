import logging

import panel as pn

logger = logging.getLogger(__name__)


class Sidebar:
    def __init__(self, template):
        self.template = template
        self.setup_markdown()
        self.setup_buttons()
        self.setup_progress_bars()

    def setup_markdown(self):
        self.mark_down_ip = pn.pane.Markdown("""**Step 1 : Save Input Parameters**""", width=300)
        self.mark_down_ip_note = pn.pane.Markdown(
            """***Note : ***<br>
                            - Save Input Parameters will save input parameters used for the analysis
                            in all the folders you selected for the analysis (useful for future
                            reference). All analysis steps will run without saving input parameters.
                            """,
            width=300,
        )
        self.mark_down_storenames = pn.pane.Markdown(
            """**Step 2 : Open Storenames GUI <br> and save storenames**""", width=300
        )
        self.mark_down_read = pn.pane.Markdown("""**Step 3 : Read Raw Data**""", width=300)
        self.mark_down_preprocess = pn.pane.Markdown("""**Step 4 : Preprocess and Remove Artifacts**""", width=300)
        self.mark_down_psth = pn.pane.Markdown("""**Step 5 : PSTH Computation**""", width=300)
        self.mark_down_visualization = pn.pane.Markdown("""**Step 6 : Visualization**""", width=300)

    def setup_buttons(self):
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
        self.save_button = pn.widgets.Button(name="Save to file...", button_type="primary", width=300, align="end")

    def attach_callbacks(self, button_name_to_onclick_fn: dict):
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            button.on_click(onclick_fn)

    def setup_progress_bars(self):
        self.read_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.extract_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        self.psth_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)

    def add_to_template(self):
        self.template.sidebar.append(self.mark_down_ip)
        self.template.sidebar.append(self.mark_down_ip_note)
        self.template.sidebar.append(self.save_button)
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
