#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import json
import logging
import os
import socket
import tkinter as tk
from pathlib import Path
from random import randint
from tkinter import StringVar, messagebox, ttk

import h5py
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from numpy import float32, float64, int32, int64, uint16

from guppy.readTevTsq import import_csv
from guppy.tdt_step2 import readtsq
from guppy.csv_step2 import import_csv_step2
from guppy.doric_step2 import import_doric
from guppy.npm_step2 import import_npm

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


def scanPortsAndFind(start_port=5000, end_port=5200, host="127.0.0.1"):
    while True:
        port = randint(start_port, end_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.001)  # Set timeout to avoid long waiting on closed ports
        result = sock.connect_ex((host, port))
        if result == 0:  # If the connection is successful, the port is open
            continue
        else:
            break

    return port


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


# function to show location for over-writing or creating a new stores list file.
def show_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename + "_output_" + str(i))
        if not os.path.exists(op):
            break
        i += 1
    return op


def make_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename + "_output_" + str(i))
        if not os.path.exists(op):
            os.mkdir(op)
            break
        i += 1

    return op


# function to show GUI and save
def saveStorenames(inputParameters, data, event_name, flag, filepath):

    logger.debug("Saving stores list file.")
    # getting input parameters
    inputParameters = inputParameters

    # Headless path: if storenames_map provided, write storesList.csv without building the Panel UI
    storenames_map = inputParameters.get("storenames_map")
    if isinstance(storenames_map, dict) and len(storenames_map) > 0:
        op = make_dir(filepath)
        arr = np.asarray([list(storenames_map.keys()), list(storenames_map.values())], dtype=str)
        np.savetxt(os.path.join(op, "storesList.csv"), arr, delimiter=",", fmt="%s")
        logger.info(f"Storeslist file saved at {op}")
        logger.info("Storeslist : \n" + str(arr))
        return

    # reading storenames from the data fetched using 'readtsq' function
    if isinstance(data, pd.DataFrame):
        data["name"] = np.asarray(data["name"], dtype=str)
        allnames = np.unique(data["name"])
        index = []
        for i in range(len(allnames)):
            length = len(str(allnames[i]))
            if length < 4:
                index.append(i)
        allnames = np.delete(allnames, index, 0)
        allnames = list(allnames)

    else:
        allnames = []

    if "data_np_v2" in flag or "data_np" in flag or "event_np" in flag:
        path_chev = glob.glob(os.path.join(filepath, "*chev*"))
        path_chod = glob.glob(os.path.join(filepath, "*chod*"))
        path_chpr = glob.glob(os.path.join(filepath, "*chpr*"))
        combine_paths = path_chev + path_chod + path_chpr
        d = dict()
        for i in range(len(combine_paths)):
            basename = (os.path.basename(combine_paths[i])).split(".")[0]
            df = pd.read_csv(combine_paths[i])
            d[basename] = {"x": np.array(df["timestamps"]), "y": np.array(df["data"])}
        keys = list(d.keys())
        mark_down_np = pn.pane.Markdown(
            """
                                        ### Extra Instructions to follow when using Neurophotometrics data :
                                        - Guppy will take the NPM data, which has interleaved frames
                                        from the signal and control channels, and divide it out into
                                        separate channels for each site you recordded.
                                        However, since NPM does not automatically annotate which
                                        frames belong to the signal channel and which belong to the
                                        control channel, the user must specify this for GuPPy.
                                        - Each of your recording sites will have a channel
                                        named “chod” and a channel named “chev”
                                        - View the plots below and, for each site,
                                        determine whether the “chev” or “chod” channel is signal or control
                                        - When you give your storenames, name the channels appropriately.
                                        For example, “chev1” might be “signal_A” and
                                        “chod1” might be “control_A” (or vice versa).

                                            """
        )
        plot_select = pn.widgets.Select(
            name="Select channel to see correspondings channels", options=keys, value=keys[0]
        )

        @pn.depends(plot_select=plot_select)
        def plot(plot_select):
            return hv.Curve((d[plot_select]["x"], d[plot_select]["y"])).opts(width=550)

    else:
        pass

    # finalizing all the storenames
    allnames = allnames + event_name

    # instructions about how to save the storeslist file
    mark_down = pn.pane.Markdown(
        """


                ### Instructions to follow :

                - Check Storenames to repeat checkbox and see instructions in “Github Wiki” for duplicating storenames.
                Otherwise do not check the Storenames to repeat checkbox.<br>
                - Select storenames from list and click “Select Storenames” to populate area below.<br>
                - Enter names for storenames, in order, using the following naming convention:<br>
                    Isosbestic = “control_region” (ex: Dv1A= control_DMS)<br>
                    Signal= “signal_region” (ex: Dv2A= signal_DMS)<br>
                    TTLs can be named using any convention (ex: PrtR = RewardedPortEntries) but should be kept consistent for later group analysis

                ```
                {"storenames": ["Dv1A", "Dv2A",
                                "Dv3B", "Dv4B",
                                "LNRW", "LNnR",
                                "PrtN", "PrtR",
                                "RNPS"],
                "names_for_storenames": ["control_DMS", "signal_DMS",
                                         "control_DLS", "signal_DLS",
                                         "RewardedNosepoke", "UnrewardedNosepoke",
                                         "UnrewardedPort", "RewardedPort",
                                         "InactiveNosepoke"]}
                ```
                - If user has saved storenames before, clicking "Select Storenames" button will pop up a dialog box
                  showing previously used names for storenames. Select names for storenames by checking a checkbox and
                  click on "Show" to populate the text area in the Storenames GUI. Close the dialog box.

                - Select “create new” or “overwrite” to generate a new storenames list or replace a previous one
                - Click Save

                """,
        width=550,
    )

    # creating GUI template
    template = pn.template.BootstrapTemplate(title="Storenames GUI - {}".format(os.path.basename(filepath), mark_down))

    # creating different buttons and selectors for the GUI
    cross_selector = pn.widgets.CrossSelector(name="Store Names Selection", value=[], options=allnames, width=600)
    multi_choice = pn.widgets.MultiChoice(
        name="Select Storenames which you want more than once (multi-choice: multiple options selection)",
        value=[],
        options=allnames,
    )

    literal_input_1 = pn.widgets.LiteralInput(
        name="Number of times you want the above storename (list)", value=[], type=list
    )
    # literal_input_2 = pn.widgets.LiteralInput(name='Names for Storenames (list)', type=list)

    repeat_storenames = pn.widgets.Checkbox(name="Storenames to repeat", value=False)
    repeat_storename_wd = pn.WidgetBox("", width=600)

    def callback(target, event):
        if event.new == True:
            target.objects = [multi_choice, literal_input_1]
        elif event.new == False:
            target.clear()

    repeat_storenames.link(repeat_storename_wd, callbacks={"value": callback})
    # repeat_storename_wd = pn.WidgetBox('Storenames to repeat (leave blank if not needed)', multi_choice, literal_input_1, background="white", width=600)

    update_options = pn.widgets.Button(name="Select Storenames", width=600)
    save = pn.widgets.Button(name="Save", width=600)

    text = pn.widgets.LiteralInput(value=[], name="Selected Store Names", type=list, width=600)

    path = pn.widgets.TextInput(name="Location to Stores List file", width=600)

    mark_down_for_overwrite = pn.pane.Markdown(
        """ Select option from below if user wants to over-write a file or create a new file.
                                    **Creating a new file will make a new output folder and will get saved at that location.**
                                    If user selects to over-write a file **Select location of the file to over-write** will provide
                                    the existing options of the output folders where user needs to over-write the file""",
        width=600,
    )

    select_location = pn.widgets.Select(
        name="Select location of the file to over-write", value="None", options=["None"], width=600
    )

    overwrite_button = pn.widgets.MenuButton(
        name="over-write storeslist file or create a new one?  ",
        items=["over_write_file", "create_new_file"],
        button_type="default",
        split=True,
        width=600,
    )

    literal_input_2 = pn.widgets.CodeEditor(value="""{}""", theme="tomorrow", language="json", height=250, width=600)

    alert = pn.pane.Alert("#### No alerts !!", alert_type="danger", height=80, width=600)

    take_widgets = pn.WidgetBox(multi_choice, literal_input_1)

    change_widgets = pn.WidgetBox(text)

    storenames = []
    storename_dropdowns = {}
    storename_textboxes = {}

    if len(allnames) == 0:
        alert.object = (
            "####Alert !! \n No storenames found. There are not any TDT files or csv files to look for storenames."
        )

    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event):
        if event.new == "over_write_file":
            select_location.options = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
            # select_location.value = select_location.options[0]
        else:
            select_location.options = [show_dir(filepath)]
            # select_location.value = select_location.options[0]

    def fetchValues(event):
        global storenames
        alert.object = "#### No alerts !!"

        if not storename_dropdowns or not len(storenames) > 0:
            alert.object = "####Alert !! \n No storenames selected."
            return

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

        comboBoxValues, textBoxValues = [], []
        dropdown_keys = list(storename_dropdowns.keys())
        textbox_keys = list(storename_textboxes.keys()) if storename_textboxes else []

        # Get dropdown values
        for key in dropdown_keys:
            comboBoxValues.append(storename_dropdowns[key].value)

        # Get textbox values (matching with dropdown keys)
        for key in dropdown_keys:
            if key in storename_textboxes:
                textbox_value = storename_textboxes[key].value or ""
                textBoxValues.append(textbox_value)

                # Validation: Check for whitespace
                if len(textbox_value.split()) > 1:
                    alert.object = "####Alert !! \n Whitespace is not allowed in the text box entry."
                    return

                # Validation: Check for empty required fields
                dropdown_value = storename_dropdowns[key].value
                if (
                    not textbox_value
                    and dropdown_value not in storenames_cache
                    and dropdown_value in ["control", "signal", "event TTLs"]
                ):
                    alert.object = "####Alert !! \n One of the text box entry is empty."
                    return
            else:
                # For cached values, use the dropdown value directly
                textBoxValues.append(storename_dropdowns[key].value)

        if len(comboBoxValues) != len(textBoxValues):
            alert.object = "####Alert !! \n Number of entries in combo box and text box should be same."
            return

        names_for_storenames = []
        for i in range(len(comboBoxValues)):
            if comboBoxValues[i] == "control" or comboBoxValues[i] == "signal":
                if "_" in textBoxValues[i]:
                    alert.object = "####Alert !! \n Please do not use underscore in region name."
                    return
                names_for_storenames.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
            elif comboBoxValues[i] == "event TTLs":
                names_for_storenames.append(textBoxValues[i])
            else:
                names_for_storenames.append(comboBoxValues[i])

        d = dict()
        d["storenames"] = text.value
        d["names_for_storenames"] = names_for_storenames
        literal_input_2.value = str(json.dumps(d, indent=2))

    # Panel-based storename configuration (replaces Tkinter dialog)
    storename_config_widgets = pn.Column(visible=False)
    show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

    # on clicking 'Select Storenames' button, following function is executed
    def update_values(event):
        global storenames, vars_list
        arr = []
        for w in take_widgets:
            arr.append(w.value)

        new_arr = []

        for i in range(len(arr[1])):
            for j in range(arr[1][i]):
                new_arr.append(arr[0][i])

        if len(new_arr) > 0:
            storenames = cross_selector.value + new_arr
        else:
            storenames = cross_selector.value

        for w in change_widgets:
            w.value = storenames

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

        # Create Panel widgets for storename configuration
        config_widgets = []
        storename_dropdowns.clear()
        storename_textboxes.clear()

        if len(storenames) > 0:
            config_widgets.append(
                pn.pane.Markdown(
                    "## Configure Storenames\nSelect appropriate options for each storename and provide names as needed:"
                )
            )

            for i, storename in enumerate(storenames):
                # Create a row for each storename
                row_widgets = []

                # Label
                label = pn.pane.Markdown(f"**{storename}:**")
                row_widgets.append(label)

                # Dropdown options
                if storename in storenames_cache:
                    options = storenames_cache[storename]
                    default_value = options[0] if options else ""
                else:
                    options = ["", "control", "signal", "event TTLs"]
                    default_value = ""

                # Create unique key for widget
                widget_key = (
                    f"{storename}_{i}"
                    if f"{storename}_{i}" not in storename_dropdowns
                    else f"{storename}_{i}_{len(storename_dropdowns)}"
                )

                dropdown = pn.widgets.Select(name="Type", value=default_value, options=options, width=150)
                storename_dropdowns[widget_key] = dropdown
                row_widgets.append(dropdown)

                # Text input (only show if not cached or if control/signal/event TTLs selected)
                if storename not in storenames_cache or default_value in ["control", "signal", "event TTLs"]:
                    textbox = pn.widgets.TextInput(
                        name="Name", value="", placeholder="Enter region/event name", width=200
                    )
                    storename_textboxes[widget_key] = textbox
                    row_widgets.append(textbox)

                    # Add helper text based on selection
                    def create_help_function(dropdown_widget, help_pane_container):
                        @pn.depends(dropdown_widget.param.value, watch=True)
                        def update_help(dropdown_value):
                            if dropdown_value == "control":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                                )
                            elif dropdown_value == "signal":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                                )
                            elif dropdown_value == "event TTLs":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type event name for the TTLs*", styles={"color": "gray", "font-size": "12px"}
                                )
                            else:
                                help_pane_container[0] = pn.pane.Markdown(
                                    "", styles={"color": "gray", "font-size": "12px"}
                                )

                        return update_help

                    help_container = [pn.pane.Markdown("")]
                    help_function = create_help_function(dropdown, help_container)
                    help_function(dropdown.value)  # Initialize
                    row_widgets.append(help_container[0])

                # Add the row to config widgets
                config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))

            # Add show button
            config_widgets.append(pn.Spacer(height=20))
            config_widgets.append(show_config_button)
            config_widgets.append(
                pn.pane.Markdown(
                    "*Click 'Show Selected Configuration' to apply your selections.*",
                    styles={"font-size": "12px", "color": "gray"},
                )
            )

        # Update the configuration panel
        storename_config_widgets.objects = config_widgets
        storename_config_widgets.visible = len(storenames) > 0

    # on clicking save button, following function is executed
    def save_button(event=None):
        global storenames

        d = json.loads(literal_input_2.value)
        arr1, arr2 = np.asarray(d["storenames"]), np.asarray(d["names_for_storenames"])

        if np.where(arr2 == "")[0].size > 0:
            alert.object = "#### Alert !! \n Empty string in the list names_for_storenames."
            logger.error("Empty string in the list names_for_storenames.")
            raise Exception("Empty string in the list names_for_storenames.")
        else:
            alert.object = "#### No alerts !!"

        if arr1.shape[0] != arr2.shape[0]:
            alert.object = "#### Alert !! \n Length of list storenames and names_for_storenames is not equal."
            logger.error("Length of list storenames and names_for_storenames is not equal.")
            raise Exception("Length of list storenames and names_for_storenames is not equal.")
        else:
            alert.object = "#### No alerts !!"

        if not os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            storenames_cache = dict()

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
                json.dump(storenames_cache, f, indent=4)
        else:
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
                json.dump(storenames_cache, f, indent=4)

        arr = np.asarray([arr1, arr2])
        logger.info(arr)
        if not os.path.exists(select_location.value):
            os.mkdir(select_location.value)

        np.savetxt(os.path.join(select_location.value, "storesList.csv"), arr, delimiter=",", fmt="%s")
        path.value = os.path.join(select_location.value, "storesList.csv")
        logger.info(f"Storeslist file saved at {select_location.value}")
        logger.info("Storeslist : \n" + str(arr))

    # Connect button callbacks
    update_options.on_click(update_values)
    show_config_button.on_click(fetchValues)
    save.on_click(save_button)
    overwrite_button.on_click(overwrite_button_actions)

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = scanPortsAndFind(start_port=5000, end_port=5200)

    if "data_np_v2" in flag or "data_np" in flag or "event_np" in flag:
        widget_1 = pn.Column("# " + os.path.basename(filepath), mark_down, mark_down_np, plot_select, plot)
        widget_2 = pn.Column(
            repeat_storenames,
            repeat_storename_wd,
            pn.Spacer(height=20),
            cross_selector,
            update_options,
            storename_config_widgets,
            pn.Spacer(height=10),
            text,
            literal_input_2,
            alert,
            mark_down_for_overwrite,
            overwrite_button,
            select_location,
            save,
            path,
        )
        template.main.append(pn.Row(widget_1, widget_2))

    else:
        widget_1 = pn.Column("# " + os.path.basename(filepath), mark_down)
        widget_2 = pn.Column(
            repeat_storenames,
            repeat_storename_wd,
            pn.Spacer(height=20),
            cross_selector,
            update_options,
            storename_config_widgets,
            pn.Spacer(height=10),
            text,
            literal_input_2,
            alert,
            mark_down_for_overwrite,
            overwrite_button,
            select_location,
            save,
            path,
        )
        template.main.append(pn.Row(widget_1, widget_2))

    template.show(port=number)



# function to read input parameters and run the saveStorenames function
def execute(inputParameters):

    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    isosbestic_control = inputParameters["isosbestic_control"]
    num_ch = inputParameters["noChannels"]
    modality = inputParameters.get("modality", "tdt")

    logger.info(folderNames)

    try:
        for i in folderNames:
            filepath = os.path.join(inputParameters["abspath"], i)
            if modality == "tdt":
                data = readtsq(filepath)
                event_name, flag = [], []
            elif modality == "csv":
                data = 0
                event_name, flag = import_csv_step2(filepath)
            elif modality == "doric":
                data = 0
                event_name, flag = import_doric(filepath)
            elif modality == "npm":
                data = 0
                event_name, flag = import_npm(filepath, num_ch)
            else:
                raise ValueError("Modality not recognized. Please use 'tdt', 'csv', 'doric', or 'npm'.")
            
            saveStorenames(inputParameters, data, event_name, flag, filepath)
        logger.info("#" * 400)
    except Exception as e:
        logger.error(str(e))
        raise e
