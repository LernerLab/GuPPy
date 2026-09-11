import json
import logging
from pathlib import Path

import panel as pn

from . import nwb_form_style as style
from .store_labeling_config import StoreLabelingConfig

pn.extension(notifications=True)

logger = logging.getLogger(__name__)


class StoreLabelingSelector:
    """Panel widget for selecting, naming, and saving store_ids for a session.

    Parameters
    ----------
    allnames : list of str
        All store_ids discovered from the data files, offered as selectable
        options in the cross-selector and multi-choice widgets.
    """

    def __init__(self, allnames: list[str]) -> None:
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="danger", width=600, visible=False)
        if len(allnames) == 0:
            self.set_alert_message(
                "####Alert !! \n No store_ids found. There are not any TDT files or csv files to look for store_ids."
            )

        # creating different buttons and selectors for the GUI
        self.cross_selector = pn.widgets.CrossSelector(name="Store Selection", value=[], options=allnames, width=600)
        self.multi_choice = pn.widgets.MultiChoice(
            name="Select Stores which you want more than once (multi-choice: multiple options selection)",
            value=[],
            options=allnames,
        )

        self.literal_input_1 = pn.widgets.LiteralInput(
            name="Number of times you want the above store (list)", value=[], type=list
        )

        self.repeat_stores = pn.widgets.Checkbox(name="Stores to repeat", value=False)
        self.repeat_store_wd = pn.WidgetBox("", width=600)

        self.repeat_stores.link(self.repeat_store_wd, callbacks={"value": self.callback})

        self.update_options = pn.widgets.Button(name="Select Stores", width=600)
        self.save = pn.widgets.Button(name="Save", width=600)

        self.text = pn.widgets.LiteralInput(value=[], name="Selected Stores", type=list, width=600)

        self._saved_notification = None

        self.mark_down_for_overwrite = pn.pane.Markdown(
            """
**Choose where to save these store labels:**

- **Create new run** — save into a new output folder named after **Run name**.
- **Overwrite existing run** — replace the output folder picked in **Run to overwrite**.
            """,
            width=600,
        )

        self.run_name = pn.widgets.TextInput(name="Run name", value="", width=600)

        self.select_location = pn.widgets.Select(name="Run to overwrite", options=[], width=600, visible=False)

        self.overwrite_mode = pn.widgets.RadioBoxGroup(
            options={"Create new run": "create_new_file", "Overwrite existing run": "over_write_file"},
            value="create_new_file",
            inline=True,
            stylesheets=[
                f":host label {{ color: {style.INK}; font-size: 13px; }} "
                f':host input[type="radio"] {{ accent-color: {style.ACCENT}; }}'
            ],
        )

        self.literal_input_2 = pn.widgets.CodeEditor(
            value="""{}""", theme="tomorrow", language="json", height=250, width=600
        )

        self.take_widgets = pn.WidgetBox(self.multi_choice, self.literal_input_1)

        self.change_widgets = pn.WidgetBox(self.text)

        self.store_id_config_widgets = pn.Column(visible=False)
        self.show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

        # Store-selection state shared across the GUI button callbacks. ``store_ids``
        # is the list of store_ids the user selected; ``store_id_dropdowns`` and
        # ``store_id_textboxes`` are populated by ``configure_store_ids`` and read
        # back when the configuration is applied.
        self.store_ids: list[str] = []
        self.store_id_dropdowns: dict[str, pn.widgets.Select] = {}
        self.store_id_textboxes: dict[str, pn.widgets.TextInput] = {}
        # Control rows reference the signal store they pair with (widget_key), so the
        # pair name is entered only once (on the signal row) and cannot be mismatched.
        self.store_id_control_refs: dict[str, pn.widgets.Select] = {}

        self.widget = pn.Column(
            self.repeat_stores,
            self.repeat_store_wd,
            pn.Spacer(height=20),
            self.cross_selector,
            self.update_options,
            self.store_id_config_widgets,
            pn.Spacer(height=10),
            self.text,
            self.literal_input_2,
            self.alert,
            self.mark_down_for_overwrite,
            self.overwrite_mode,
            self.run_name,
            self.select_location,
            self.save,
        )

    def callback(self, target: pn.WidgetBox, event: object) -> None:
        """Show or hide the store_ids-to-repeat widget box based on the checkbox state.

        Parameters
        ----------
        target : pn.WidgetBox
            The widget box to populate or clear.
        event : param.parameterized.Event
            The watch event fired by ``repeat_stores``; ``event.new`` is the
            new checkbox value.
        """
        if event.new == True:
            target.objects = [self.multi_choice, self.literal_input_1]
        elif event.new == False:
            target.clear()

    def get_select_location(self) -> str:
        """Return the currently selected overwrite-location option.

        Returns
        -------
        str
            Value of the ``select_location`` selector widget.
        """
        return self.select_location.value

    def set_select_location_options(self, options: list[str]) -> None:
        """Replace the options in the overwrite-location selector.

        Parameters
        ----------
        options : list of str
            Run folder paths to offer, each labelled by its folder name.
        """
        # Pick the value alongside the options: when Panel picks it on its own, the pick never reaches the
        # browser and the dropdown renders blank.
        value = self.select_location.value if self.select_location.value in options else next(iter(options), None)
        self.select_location.param.update(options={Path(option).name: option for option in options}, value=value)

    def set_alert_message(self, message: str) -> None:
        """Set the text shown in the alert pane, hiding the pane when there is nothing to report.

        Parameters
        ----------
        message : str
            Markdown-formatted alert message; ``"#### No alerts !!"`` hides the pane.
        """
        self.alert.object = message
        self.alert.visible = message != "#### No alerts !!"

    def get_literal_input_2(self) -> dict[str, object]:  # TODO: come up with a better name for this method.
        """Parse and return the JSON store_ids mapping from the code editor widget.

        Returns
        -------
        dict
            Parsed JSON object from the ``literal_input_2`` code editor.
        """
        store_labeling_config = json.loads(self.literal_input_2.value)
        return store_labeling_config

    def set_literal_input_2(
        self, store_labeling_config: dict[str, object]
    ) -> None:  # TODO: come up with a better name for this method.
        """Serialise ``store_labeling_config`` as pretty-printed JSON and set the code editor value.

        Parameters
        ----------
        store_labeling_config : dict
            Dictionary to serialise into the ``literal_input_2`` code editor.
        """
        self.literal_input_2.value = str(json.dumps(store_labeling_config, indent=2))

    def get_take_widgets(self) -> list[object]:
        """Return the current values of the repeat-store_ids widgets.

        Returns
        -------
        list
            One entry per widget in ``take_widgets`` containing that widget's
            current value.
        """
        return [widget.value for widget in self.take_widgets]

    def set_change_widgets(self, value: object) -> None:
        """Set all ``change_widgets`` to ``value``.

        Parameters
        ----------
        value : object
            Value to assign to every widget in the ``change_widgets`` box.
        """
        for widget in self.change_widgets:
            widget.value = value

    def set_events(self, *, events: list[str]) -> None:
        """Replace the selectable store_id options in the store-selection widgets.

        Used after deferred discovery (e.g. once the NPM configuration is
        confirmed) to populate the cross-selector and multi-choice with the
        newly discovered store_ids.

        Parameters
        ----------
        events : list of str
            store_id strings to offer as options.
        """
        self.cross_selector.options = events
        self.multi_choice.options = events

    def get_cross_selector(self) -> list[str]:
        """Return the store_ids currently selected in the cross-selector.

        Returns
        -------
        list of str
            Values selected by the user in ``cross_selector``.
        """
        return self.cross_selector.value

    def show_saved_message(self, message: str) -> None:
        """Pop up a confirmation that stays until dismissed, replacing any earlier one.

        Parameters
        ----------
        message : str
            HTML-formatted confirmation message.
        """
        self.hide_saved_message()
        if pn.state.notifications is not None:
            self._saved_notification = pn.state.notifications.success(message, duration=0)

    def hide_saved_message(self) -> None:
        """Dismiss the save confirmation, if one is showing."""
        if self._saved_notification is not None:
            self._saved_notification.destroy()
            self._saved_notification = None

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on selector buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (e.g. ``"save"``) to the callable
            that should be invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            getattr(self, button_name).on_click(onclick_fn)

    def attach_overwrite_mode_watcher(self, callback: object) -> None:
        """Attach a watcher that fires when the create-new / overwrite choice changes.

        Creating a new run shows the run-name field; overwriting shows the run picker instead.

        Parameters
        ----------
        callback : callable
            Function with signature ``callback(event)`` where ``event.new`` is
            ``"create_new_file"`` or ``"over_write_file"``.
        """

        def show_mode_widgets_then_call(event: object) -> None:
            creating = event.new == "create_new_file"
            self.run_name.visible = creating
            self.select_location.visible = not creating
            callback(event)

        self.overwrite_mode.param.watch(show_mode_widgets_then_call, "value")

    def set_run_name(self, value: str) -> None:
        """Set the run-name TextInput value.

        Parameters
        ----------
        value : str
            Run name to display.
        """
        self.run_name.value = value

    def attach_run_name_watcher(self, callback: object) -> None:
        """Attach a watcher that fires when the run-name TextInput value changes.

        Parameters
        ----------
        callback : callable
            Function with signature ``callback(event)`` where ``event.new`` is
            the new run-name string.
        """
        self.run_name.param.watch(callback, "value")

    def get_run_name(self) -> str:
        """Return the current value of the run-name TextInput.

        Returns
        -------
        str
            Run-name string entered by the user (may be empty).
        """
        return self.run_name.value

    def get_overwrite_mode(self) -> str:
        """Return the current overwrite-vs-create mode.

        Returns
        -------
        str
            ``"over_write_file"`` or ``"create_new_file"``.
        """
        return self.overwrite_mode.value

    def configure_store_ids(self, store_id_to_store_labels: dict[str, list[str]]) -> None:
        """Build the store_id-configuration panel for ``self.store_ids`` and make it visible.

        Reads the currently selected store_ids from ``self.store_ids`` and
        populates ``self.store_id_dropdowns`` / ``self.store_id_textboxes`` with
        the per-store dropdown and text-input widgets.

        Parameters
        ----------
        store_id_to_store_labels : dict
            Previously saved store_id assignments for pre-population.
        """
        # Create Panel widgets for store_id configuration
        self.store_labeling_config = StoreLabelingConfig(
            show_config_button=self.show_config_button,
            store_id_dropdowns=self.store_id_dropdowns,
            store_id_textboxes=self.store_id_textboxes,
            store_id_control_refs=self.store_id_control_refs,
            store_ids=self.store_ids,
            store_id_to_store_labels=store_id_to_store_labels,
        )

        # Update the configuration panel
        self.store_id_config_widgets.objects = self.store_labeling_config.config_widgets
        self.store_id_config_widgets.visible = len(self.store_ids) > 0
