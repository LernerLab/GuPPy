import json
import logging

import panel as pn

from .storenames_config import StorenamesConfig

pn.extension()

logger = logging.getLogger(__name__)


class StorenamesSelector:
    """Panel widget for selecting, naming, and saving storenames for a session.

    Parameters
    ----------
    allnames : list of str
        All storenames discovered from the data files, offered as selectable
        options in the cross-selector and multi-choice widgets.
    """

    def __init__(self, allnames):
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="danger", height=80, width=600)
        if len(allnames) == 0:
            self.alert.object = (
                "####Alert !! \n No storenames found. There are not any TDT files or csv files to look for storenames."
            )

        # creating different buttons and selectors for the GUI
        self.cross_selector = pn.widgets.CrossSelector(
            name="Store Names Selection", value=[], options=allnames, width=600
        )
        self.multi_choice = pn.widgets.MultiChoice(
            name="Select Storenames which you want more than once (multi-choice: multiple options selection)",
            value=[],
            options=allnames,
        )

        self.literal_input_1 = pn.widgets.LiteralInput(
            name="Number of times you want the above storename (list)", value=[], type=list
        )
        # self.literal_input_2 = pn.widgets.LiteralInput(name='Names for Storenames (list)', type=list)

        self.repeat_storenames = pn.widgets.Checkbox(name="Storenames to repeat", value=False)
        self.repeat_storename_wd = pn.WidgetBox("", width=600)

        self.repeat_storenames.link(self.repeat_storename_wd, callbacks={"value": self.callback})
        # self.repeat_storename_wd = pn.WidgetBox('Storenames to repeat (leave blank if not needed)', multi_choice, literal_input_1, background="white", width=600)

        self.update_options = pn.widgets.Button(name="Select Storenames", width=600)
        self.save = pn.widgets.Button(name="Save", width=600)

        self.text = pn.widgets.LiteralInput(value=[], name="Selected Store Names", type=list, width=600)

        self.path = pn.widgets.TextInput(name="Location to Stores List file", width=600)

        self.mark_down_for_overwrite = pn.pane.Markdown(
            """
**Choose how to save this storesList:**

- **create_new_file** — create a new output folder. Optionally set **Run name** below; leave blank to use the next available integer.
- **over_write_file** — replace an existing output folder. Pick which one in **Select location of the file to over-write**.
            """,
            width=600,
        )

        self.run_name = pn.widgets.TextInput(
            name="Run name",
            value="",
            placeholder="optional — defaults to next available integer",
            width=600,
        )

        self.select_location = pn.widgets.Select(
            name="Select location of the file to over-write", value="None", options=["None"], width=600
        )

        self.overwrite_button = pn.widgets.MenuButton(
            name="over-write storeslist file or create a new one?  ",
            items=["over_write_file", "create_new_file"],
            button_type="default",
            split=True,
            width=600,
        )
        self._current_overwrite_mode = "create_new_file"

        self.literal_input_2 = pn.widgets.CodeEditor(
            value="""{}""", theme="tomorrow", language="json", height=250, width=600
        )

        self.take_widgets = pn.WidgetBox(self.multi_choice, self.literal_input_1)

        self.change_widgets = pn.WidgetBox(self.text)

        # Panel-based storename configuration (replaces Tkinter dialog)
        self.storename_config_widgets = pn.Column(visible=False)
        self.show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

        self.widget = pn.Column(
            self.repeat_storenames,
            self.repeat_storename_wd,
            pn.Spacer(height=20),
            self.cross_selector,
            self.update_options,
            self.storename_config_widgets,
            pn.Spacer(height=10),
            self.text,
            self.literal_input_2,
            self.alert,
            self.mark_down_for_overwrite,
            self.overwrite_button,
            self.run_name,
            self.select_location,
            self.save,
            self.path,
        )

    def callback(self, target, event):
        """Show or hide the storenames-to-repeat widget box based on the checkbox state.

        Parameters
        ----------
        target : pn.WidgetBox
            The widget box to populate or clear.
        event : param.parameterized.Event
            The watch event fired by ``repeat_storenames``; ``event.new`` is the
            new checkbox value.
        """
        if event.new == True:
            target.objects = [self.multi_choice, self.literal_input_1]
        elif event.new == False:
            target.clear()

    def get_select_location(self):
        """Return the currently selected overwrite-location option.

        Returns
        -------
        str
            Value of the ``select_location`` selector widget.
        """
        return self.select_location.value

    def set_select_location_options(self, options):
        """Replace the options in the overwrite-location selector.

        Parameters
        ----------
        options : list of str
            New list of location options to display.
        """
        self.select_location.options = options

    def set_alert_message(self, message):
        """Set the text shown in the alert pane.

        Parameters
        ----------
        message : str
            Markdown-formatted alert message.
        """
        self.alert.object = message

    def get_literal_input_2(self):  # TODO: come up with a better name for this method.
        """Parse and return the JSON storenames mapping from the code editor widget.

        Returns
        -------
        dict
            Parsed JSON object from the ``literal_input_2`` code editor.
        """
        storenames_config = json.loads(self.literal_input_2.value)
        return storenames_config

    def set_literal_input_2(self, storenames_config):  # TODO: come up with a better name for this method.
        """Serialise ``storenames_config`` as pretty-printed JSON and set the code editor value.

        Parameters
        ----------
        storenames_config : dict
            Dictionary to serialise into the ``literal_input_2`` code editor.
        """
        self.literal_input_2.value = str(json.dumps(storenames_config, indent=2))

    def get_take_widgets(self):
        """Return the current values of the repeat-storenames widgets.

        Returns
        -------
        list
            One entry per widget in ``take_widgets`` containing that widget's
            current value.
        """
        return [w.value for w in self.take_widgets]

    def set_change_widgets(self, value):
        """Set all ``change_widgets`` to ``value``.

        Parameters
        ----------
        value : object
            Value to assign to every widget in the ``change_widgets`` box.
        """
        for w in self.change_widgets:
            w.value = value

    def get_cross_selector(self):
        """Return the storenames currently selected in the cross-selector.

        Returns
        -------
        list of str
            Values selected by the user in ``cross_selector``.
        """
        return self.cross_selector.value

    def set_path(self, value):
        """Set the displayed path in the location text input.

        Parameters
        ----------
        value : str
            Path string to display in the ``path`` widget.
        """
        self.path.value = value

    def attach_callbacks(self, button_name_to_onclick_fn: dict):
        """Register click-handler callbacks on selector buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (e.g. ``"save"``) to the callable
            that should be invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            if button_name == "overwrite_button":
                # Wrap the user callback so we can also remember the current
                # mode (for get_overwrite_mode) and hide the run-name field in
                # overwrite mode where it has no effect.
                def remember_then_call(event, _user_callback=onclick_fn):
                    self._current_overwrite_mode = event.new
                    self.run_name.visible = event.new == "create_new_file"
                    _user_callback(event)

                button.on_click(remember_then_call)
            else:
                button.on_click(onclick_fn)

    def attach_run_name_watcher(self, callback):
        """Attach a watcher that fires when the run-name TextInput value changes.

        Parameters
        ----------
        callback : callable
            Function with signature ``callback(event)`` where ``event.new`` is
            the new run-name string.
        """
        self.run_name.param.watch(callback, "value")

    def get_run_name(self):
        """Return the current value of the run-name TextInput.

        Returns
        -------
        str
            Run-name string entered by the user (may be empty).
        """
        return self.run_name.value

    def get_overwrite_mode(self):
        """Return the current overwrite-vs-create mode.

        Returns
        -------
        str
            ``"over_write_file"`` or ``"create_new_file"``.
        """
        return self._current_overwrite_mode

    def configure_storenames(self, storename_dropdowns, storename_textboxes, storenames, storenames_cache):
        """Build the storename-configuration panel and make it visible.

        Parameters
        ----------
        storename_dropdowns : dict
            Mutable mapping populated by ``StorenamesConfig`` with dropdown widgets.
        storename_textboxes : dict
            Mutable mapping populated by ``StorenamesConfig`` with text-input widgets.
        storenames : list of str
            Raw storenames selected by the user.
        storenames_cache : dict
            Previously saved storename assignments for pre-population.
        """
        # Create Panel widgets for storename configuration
        self.storenames_config = StorenamesConfig(
            show_config_button=self.show_config_button,
            storename_dropdowns=storename_dropdowns,
            storename_textboxes=storename_textboxes,
            storenames=storenames,
            storenames_cache=storenames_cache,
        )

        # Update the configuration panel
        self.storename_config_widgets.objects = self.storenames_config.config_widgets
        self.storename_config_widgets.visible = len(storenames) > 0
