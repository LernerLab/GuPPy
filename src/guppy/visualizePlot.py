import glob
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn

from .frontend.frontend_utils import scanPortsAndFind
from .frontend.visualization import Viewer, remove_cols
from .utils.utils import get_all_stores_for_combining_data, read_Df, takeOnlyDirs

pn.extension()

logger = logging.getLogger(__name__)


# helper function to create plots
def helper_plots(filepath, event, name, inputParameters):

    basename = os.path.basename(filepath)
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]

    # note when there are no behavior event TTLs
    if len(event) == 0:
        logger.warning("\033[1m" + "There are no behavior event TTLs present to visualize.".format(event) + "\033[0m")
        return 0

    if os.path.exists(os.path.join(filepath, "cross_correlation_output")):
        event_corr, frames = [], []
        if visualize_zscore_or_dff == "z_score":
            corr_fp = glob.glob(os.path.join(filepath, "cross_correlation_output", "*_z_score_*"))
        elif visualize_zscore_or_dff == "dff":
            corr_fp = glob.glob(os.path.join(filepath, "cross_correlation_output", "*_dff_*"))
        for i in range(len(corr_fp)):
            filename = os.path.basename(corr_fp[i]).split(".")[0]
            event_corr.append(filename)
            df = pd.read_hdf(corr_fp[i], key="df", mode="r")
            frames.append(df)
        if len(frames) > 0:
            df_corr = pd.concat(frames, keys=event_corr, axis=1)
        else:
            event_corr = []
            df_corr = []
    else:
        event_corr = []
        df_corr = None

    # combine all the event PSTH so that it can be viewed together
    if name:
        event_name, name = event, name
        new_event, frames, bins = [], [], {}
        for i in range(len(event_name)):

            for j in range(len(name)):
                new_event.append(event_name[i] + "_" + name[j].split("_")[-1])
                new_name = name[j]
                temp_df = read_Df(filepath, new_event[-1], new_name)
                cols = list(temp_df.columns)
                regex = re.compile("bin_[(]")
                bins[new_event[-1]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
                # bins.append(keep_cols)
                frames.append(temp_df)

        df = pd.concat(frames, keys=new_event, axis=1)
    else:
        new_event = list(np.unique(np.array(event)))
        frames, bins = [], {}
        for i in range(len(new_event)):
            temp_df = read_Df(filepath, new_event[i], "")
            cols = list(temp_df.columns)
            regex = re.compile("bin_[(]")
            bins[new_event[i]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
            frames.append(temp_df)

        df = pd.concat(frames, keys=new_event, axis=1)

    if isinstance(df_corr, pd.DataFrame):
        new_event.extend(event_corr)
        df = pd.concat([df, df_corr], axis=1, sort=False).reset_index()

    columns_dict = dict()
    for i in range(len(new_event)):
        df_1 = df[new_event[i]]
        columns = list(df_1.columns)
        columns.append("All")
        columns_dict[new_event[i]] = columns

    # make options array for different selectors
    multiple_plots_options = []
    heatmap_options = new_event
    bins_keys = list(bins.keys())
    if len(bins_keys) > 0:
        bins_new = bins
        for i in range(len(bins_keys)):
            arr = bins[bins_keys[i]]
            if len(arr) > 0:
                # heatmap_options.append('{}_bin'.format(bins_keys[i]))
                for j in arr:
                    multiple_plots_options.append("{}_{}".format(bins_keys[i], j))

        multiple_plots_options = new_event + multiple_plots_options
    else:
        multiple_plots_options = new_event
    x_min = float(inputParameters["nSecPrev"]) - 20
    x_max = float(inputParameters["nSecPost"]) + 20
    colormaps = plt.colormaps()
    new_colormaps = ["plasma", "plasma_r", "magma", "magma_r", "inferno", "inferno_r", "viridis", "viridis_r"]
    set_a = set(colormaps)
    set_b = set(new_colormaps)
    colormaps = new_colormaps + list(set_a.difference(set_b))
    x = [columns_dict[new_event[0]][-4]]
    y = remove_cols(columns_dict[new_event[0]])
    trial_no = range(1, len(remove_cols(columns_dict[heatmap_options[0]])[:-2]) + 1)
    trial_ts = [
        "{} - {}".format(i, j) for i, j in zip(trial_no, remove_cols(columns_dict[heatmap_options[0]])[:-2])
    ] + ["All"]

    view = Viewer(
        event_selector=new_event,
        event_selector_heatmap=heatmap_options,
        selector_for_multipe_events_plot=multiple_plots_options,
        columns_dict=columns_dict,
        df_new=df,
        x_min=x_min,
        x_max=x_max,
        color_map=colormaps,
        filepath=filepath,
        x=x,
        y=y,
        heatmap_y=trial_ts,
        psth_y=trial_ts[:-1],
    )

    # PSTH plot options
    psth_checkbox = pn.Param(
        view.param.select_trials_checkbox,
        widgets={
            "select_trials_checkbox": {
                "type": pn.widgets.CheckBoxGroup,
                "inline": True,
                "name": "Select mean and/or just trials",
            }
        },
    )
    parameters = pn.Param(
        view.param.selector_for_multipe_events_plot,
        widgets={
            "selector_for_multipe_events_plot": {"type": pn.widgets.CrossSelector, "width": 550, "align": "start"}
        },
    )
    heatmap_y_parameters = pn.Param(
        view.param.heatmap_y,
        widgets={
            "heatmap_y": {"type": pn.widgets.MultiSelect, "name": "Trial # - Timestamps", "width": 200, "size": 30}
        },
    )
    psth_y_parameters = pn.Param(
        view.param.psth_y,
        widgets={
            "psth_y": {
                "type": pn.widgets.MultiSelect,
                "name": "Trial # - Timestamps",
                "width": 200,
                "size": 15,
                "align": "start",
            }
        },
    )

    event_selector = pn.Param(
        view.param.event_selector, widgets={"event_selector": {"type": pn.widgets.Select, "width": 400}}
    )
    x_selector = pn.Param(view.param.x, widgets={"x": {"type": pn.widgets.Select, "width": 180}})
    y_selector = pn.Param(view.param.y, widgets={"y": {"type": pn.widgets.Select, "width": 180}})

    width_plot = pn.Param(view.param.Width_Plot, widgets={"Width_Plot": {"type": pn.widgets.Select, "width": 70}})
    height_plot = pn.Param(view.param.Height_Plot, widgets={"Height_Plot": {"type": pn.widgets.Select, "width": 70}})
    ylabel = pn.Param(view.param.Y_Label, widgets={"Y_Label": {"type": pn.widgets.Select, "width": 70}})
    save_opts = pn.Param(view.param.save_options, widgets={"save_options": {"type": pn.widgets.Select, "width": 70}})

    xlimit_plot = pn.Param(view.param.X_Limit, widgets={"X_Limit": {"type": pn.widgets.RangeSlider, "width": 180}})
    ylimit_plot = pn.Param(view.param.Y_Limit, widgets={"Y_Limit": {"type": pn.widgets.RangeSlider, "width": 180}})
    save_psth = pn.Param(view.param.save_psth, widgets={"save_psth": {"type": pn.widgets.Button, "width": 400}})

    options = pn.Column(
        event_selector,
        pn.Row(x_selector, y_selector),
        pn.Row(xlimit_plot, ylimit_plot),
        pn.Row(width_plot, height_plot, ylabel, save_opts),
        save_psth,
    )

    options_selectors = pn.Row(options, parameters)

    line_tab = pn.Column(
        "## " + basename,
        pn.Row(options_selectors, pn.Column(psth_checkbox, psth_y_parameters), width=1200),
        view.contPlot,
        view.update_selector,
        view.plot_specific_trials,
    )

    # Heatmap plot options
    event_selector_heatmap = pn.Param(
        view.param.event_selector_heatmap, widgets={"event_selector_heatmap": {"type": pn.widgets.Select, "width": 150}}
    )
    color_map = pn.Param(view.param.color_map, widgets={"color_map": {"type": pn.widgets.Select, "width": 150}})
    width_heatmap = pn.Param(
        view.param.width_heatmap, widgets={"width_heatmap": {"type": pn.widgets.Select, "width": 150}}
    )
    height_heatmap = pn.Param(
        view.param.height_heatmap, widgets={"height_heatmap": {"type": pn.widgets.Select, "width": 150}}
    )
    save_hm = pn.Param(view.param.save_hm, widgets={"save_hm": {"type": pn.widgets.Button, "width": 150}})
    save_options_heatmap = pn.Param(
        view.param.save_options_heatmap, widgets={"save_options_heatmap": {"type": pn.widgets.Select, "width": 150}}
    )

    hm_tab = pn.Column(
        "## " + basename,
        pn.Row(
            event_selector_heatmap,
            color_map,
            width_heatmap,
            height_heatmap,
            save_options_heatmap,
            pn.Column(pn.Spacer(height=25), save_hm),
        ),
        pn.Row(view.heatmap, heatmap_y_parameters),
    )  #
    logger.info("app")

    template = pn.template.MaterialTemplate(title="Visualization GUI")

    number = scanPortsAndFind(start_port=5000, end_port=5200)
    app = pn.Tabs(("PSTH", line_tab), ("Heat Map", hm_tab))

    template.main.append(app)

    template.show(port=number)


# function to combine all the output folders together and preprocess them to use them in helper_plots function
def createPlots(filepath, event, inputParameters):

    for i in range(len(event)):
        event[i] = event[i].replace("\\", "_")
        event[i] = event[i].replace("/", "_")

    average = inputParameters["visualizeAverageResults"]
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]

    if average == True:
        path = []
        for i in range(len(event)):
            if visualize_zscore_or_dff == "z_score":
                path.append(glob.glob(os.path.join(filepath, event[i] + "*_z_score_*")))
            elif visualize_zscore_or_dff == "dff":
                path.append(glob.glob(os.path.join(filepath, event[i] + "*_dff_*")))

        path = np.concatenate(path)
    else:
        if visualize_zscore_or_dff == "z_score":
            path = glob.glob(os.path.join(filepath, "z_score_*"))
        elif visualize_zscore_or_dff == "dff":
            path = glob.glob(os.path.join(filepath, "dff_*"))

    name_arr = []
    event_arr = []

    index = []
    for i in range(len(event)):
        if "control" in event[i].lower() or "signal" in event[i].lower():
            index.append(i)

    event = np.delete(event, index)

    for i in range(len(path)):
        name = (os.path.basename(path[i])).split(".")
        name = name[0]
        name_arr.append(name)

    if average == True:
        logger.info("average")
        helper_plots(filepath, name_arr, "", inputParameters)
    else:
        helper_plots(filepath, event, name_arr, inputParameters)


def visualizeResults(inputParameters):

    inputParameters = inputParameters

    average = inputParameters["visualizeAverageResults"]
    logger.info(average)

    folderNames = inputParameters["folderNames"]
    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    combine_data = inputParameters["combine_data"]

    if average == True and len(folderNamesForAvg) > 0:
        # folderNames = folderNamesForAvg
        filepath_avg = os.path.join(inputParameters["abspath"], "average")
        # filepath = os.path.join(inputParameters['abspath'], folderNames[0])
        storesListPath = []
        for i in range(len(folderNamesForAvg)):
            filepath = folderNamesForAvg[i]
            storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
        storesListPath = np.concatenate(storesListPath)
        storesList = np.asarray([[], []])
        for i in range(storesListPath.shape[0]):
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(
                        os.path.join(storesListPath[i], "storesList.csv"), dtype="str", delimiter=","
                    ).reshape(2, -1),
                ),
                axis=1,
            )
        storesList = np.unique(storesList, axis=1)

        createPlots(filepath_avg, np.unique(storesList[1, :]), inputParameters)

    else:
        if combine_data == True:
            storesListPath = []
            for i in range(len(folderNames)):
                filepath = folderNames[i]
                storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
            storesListPath = list(np.concatenate(storesListPath).flatten())
            op = get_all_stores_for_combining_data(storesListPath)
            for i in range(len(op)):
                storesList = np.asarray([[], []])
                for j in range(len(op[i])):
                    storesList = np.concatenate(
                        (
                            storesList,
                            np.genfromtxt(os.path.join(op[i][j], "storesList.csv"), dtype="str", delimiter=",").reshape(
                                2, -1
                            ),
                        ),
                        axis=1,
                    )
                storesList = np.unique(storesList, axis=1)
                filepath = op[i][0]
                createPlots(filepath, storesList[1, :], inputParameters)
        else:
            for i in range(len(folderNames)):

                filepath = folderNames[i]
                storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
                for j in range(len(storesListPath)):
                    filepath = storesListPath[j]
                    storesList = np.genfromtxt(
                        os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=","
                    ).reshape(2, -1)

                    createPlots(filepath, storesList[1, :], inputParameters)


# logger.info(sys.argv[1:])
# visualizeResults(sys.argv[1:][0])
