import logging

# holoviews must be imported before the first pn.extension() call so Panel wires up the HoloViews
# bokeh opts namespace that the visualization step relies on; importing this module first (it sorts
# ahead of store_ids in home.py) would otherwise leave those opts unregistered. Mirrors the
# import-then-extension ordering in store_labeling_instructions.py.
import holoviews as hv  # noqa: F401
import panel as pn

pn.extension()

logger = logging.getLogger(__name__)

_INSTRUCTIONS = """
## Import Custom Events

Add behavioral events you derived outside GuPPy (e.g. movement onsets or
manually scored events). For each event, enter a **name** and paste its
**timestamps** — one per row, in seconds — copied straight from a spreadsheet
column. Click **Add event** for more, then **Save**.

Each event is written into this session as ``<name>.csv`` and appears as a
selectable store the next time you open the Label Stores GUI. This step is
optional — leave it empty and just continue if you have no custom events.
"""


class CustomEventsConfig:
    """Panel widget for pasting one or more named custom events for a session.

    Renders a header, a growable list of event rows (each with a name field and
    a paste box for timestamps), an "Add event" button, an "Overwrite existing"
    checkbox, a "Save" button, and an alert pane. The widget only collects user
    input and displays alerts; writing the CSV files is the orchestration
    layer's responsibility.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[pn.widgets.TextInput, pn.widgets.TextAreaInput]] = []

        self.instructions = pn.pane.Markdown(_INSTRUCTIONS, width=600)
        self.event_rows = pn.Column()
        self.add_event = pn.widgets.Button(name="Add event", button_type="default", width=600)
        self.overwrite = pn.widgets.Checkbox(name="Overwrite existing events with the same name", value=False)
        self.save = pn.widgets.Button(name="Save", button_type="primary", width=600)
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="primary", width=600)

        self.add_event_row()

        self.widget = pn.Column(
            self.instructions,
            self.event_rows,
            self.add_event,
            pn.Spacer(height=10),
            self.overwrite,
            self.save,
            self.alert,
        )

    def add_event_row(self, event: object = None) -> None:
        """Append a fresh, empty event row (name field + paste box).

        Parameters
        ----------
        event : object, optional
            Unused click-event argument so this can be wired directly to the
            "Add event" button.
        """
        name_input = pn.widgets.TextInput(name="Event name", placeholder="e.g. movement_onset", width=200)
        timestamps_input = pn.widgets.TextAreaInput(
            name="Timestamps (one per row, seconds)", placeholder="0.0\n1.5\n2.3", height=150, width=380
        )
        self.rows.append((name_input, timestamps_input))
        self.event_rows.append(pn.Row(name_input, timestamps_input, margin=(5, 0)))

    def get_rows(self) -> list[tuple[str, str]]:
        """Return the ``(name, pasted_text)`` pair currently entered in each row.

        Returns
        -------
        list of tuple of (str, str)
            One ``(name, pasted_text)`` pair per event row, in display order.
        """
        return [(name_input.value, timestamps_input.value) for name_input, timestamps_input in self.rows]

    def set_alert_message(self, message: str) -> None:
        """Set the text shown in the alert pane.

        Parameters
        ----------
        message : str
            Markdown-formatted alert message.
        """
        self.alert.object = message

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on the widget's buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (``"add_event"`` or ``"save"``) to
            the callable invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            button.on_click(onclick_fn)
