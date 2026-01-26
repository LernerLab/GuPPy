import json
import logging
import os
import subprocess
import sys
from threading import Thread

import numpy as np
import panel as pn

from .frontend.path_selection import get_folder_path
from .frontend.progress import readPBIncrementValues
from .saveStoresList import execute
from .visualizePlot import visualizeResults

logger = logging.getLogger(__name__)


def savingInputParameters():
    pn.extension()
    global folder_path
    folder_path = get_folder_path()
    current_dir = os.getcwd()

    def readRawData():
        inputParameters = getInputParameters()
        subprocess.call([sys.executable, "-m", "guppy.readTevTsq", json.dumps(inputParameters)])

    def extractTs():
        inputParameters = getInputParameters()
        subprocess.call([sys.executable, "-m", "guppy.preprocess", json.dumps(inputParameters)])

    def psthComputation():
        inputParameters = getInputParameters()
        inputParameters["curr_dir"] = current_dir
        subprocess.call([sys.executable, "-m", "guppy.computePsth", json.dumps(inputParameters)])

    def checkSameLocation(arr, abspath):
        # abspath = []
        for i in range(len(arr)):
            abspath.append(os.path.dirname(arr[i]))
        abspath = np.asarray(abspath)
        abspath = np.unique(abspath)
        if len(abspath) > 1:
            logger.error("All the folders selected should be at the same location")
            raise Exception("All the folders selected should be at the same location")

        return abspath

    def getAbsPath():
        arr_1, arr_2 = files_1.value, files_2.value
        if len(arr_1) == 0 and len(arr_2) == 0:
            logger.error("No folder is selected for analysis")
            raise Exception("No folder is selected for analysis")

        abspath = []
        if len(arr_1) > 0:
            abspath = checkSameLocation(arr_1, abspath)
        else:
            abspath = checkSameLocation(arr_2, abspath)

        abspath = np.unique(abspath)
        if len(abspath) > 1:
            logger.error("All the folders selected should be at the same location")
            raise Exception("All the folders selected should be at the same location")
        return abspath

    def onclickProcess(event=None):

        logger.debug("Saving Input Parameters file.")
        abspath = getAbsPath()
        analysisParameters = {
            "combine_data": combine_data.value,
            "isosbestic_control": isosbestic_control.value,
            "timeForLightsTurnOn": timeForLightsTurnOn.value,
            "filter_window": moving_avg_filter.value,
            "removeArtifacts": removeArtifacts.value,
            "noChannels": no_channels_np.value,
            "zscore_method": z_score_computation.value,
            "baselineWindowStart": baseline_wd_strt.value,
            "baselineWindowEnd": baseline_wd_end.value,
            "nSecPrev": nSecPrev.value,
            "nSecPost": nSecPost.value,
            "timeInterval": timeInterval.value,
            "bin_psth_trials": bin_psth_trials.value,
            "use_time_or_trials": use_time_or_trials.value,
            "baselineCorrectionStart": baselineCorrectionStart.value,
            "baselineCorrectionEnd": baselineCorrectionEnd.value,
            "peak_startPoint": list(df_widget.value["Peak Start time"]),  # startPoint.value,
            "peak_endPoint": list(df_widget.value["Peak End time"]),  # endPoint.value,
            "selectForComputePsth": computePsth.value,
            "selectForTransientsComputation": transients.value,
            "moving_window": moving_wd.value,
            "highAmpFilt": highAmpFilt.value,
            "transientsThresh": transientsThresh.value,
        }
        for folder in files_1.value:
            with open(os.path.join(folder, "GuPPyParamtersUsed.json"), "w") as f:
                json.dump(analysisParameters, f, indent=4)
            logger.info(f"Input Parameters file saved at {folder}")

        logger.info("#" * 400)

        # path.value = (os.path.join(op, 'inputParameters.json')).replace('\\', '/')
        logger.info("Input Parameters File Saved.")

    def onclickStoresList(event=None):
        inputParameters = getInputParameters()
        execute(inputParameters)

    def onclickVisualization(event=None):
        inputParameters = getInputParameters()
        visualizeResults(inputParameters)

    def onclickreaddata(event=None):
        thread = Thread(target=readRawData)
        thread.start()
        readPBIncrementValues(read_progress)
        thread.join()

    def onclickextractts(event=None):
        thread = Thread(target=extractTs)
        thread.start()
        readPBIncrementValues(extract_progress)
        thread.join()

    def onclickpsth(event=None):
        thread = Thread(target=psthComputation)
        thread.start()
        readPBIncrementValues(psth_progress)
        thread.join()

    mark_down_ip = pn.pane.Markdown("""**Step 1 : Save Input Parameters**""", width=300)
    mark_down_ip_note = pn.pane.Markdown(
        """***Note : ***<br>
                                            - Save Input Parameters will save input parameters used for the analysis
                                            in all the folders you selected for the analysis (useful for future
                                            reference). All analysis steps will run without saving input parameters.
                                        """,
        width=300,
    )
    save_button = pn.widgets.Button(name="Save to file...", button_type="primary", width=300, align="end")
    mark_down_storenames = pn.pane.Markdown("""**Step 2 : Open Storenames GUI <br> and save storenames**""", width=300)
    open_storesList = pn.widgets.Button(name="Open Storenames GUI", button_type="primary", width=300, align="end")
    mark_down_read = pn.pane.Markdown("""**Step 3 : Read Raw Data**""", width=300)
    read_rawData = pn.widgets.Button(name="Read Raw Data", button_type="primary", width=300, align="end")
    mark_down_extract = pn.pane.Markdown("""**Step 4 : Extract timestamps <br> and its correction**""", width=300)
    extract_ts = pn.widgets.Button(
        name="Extract timestamps and it's correction", button_type="primary", width=300, align="end"
    )
    mark_down_psth = pn.pane.Markdown("""**Step 5 : PSTH Computation**""", width=300)
    psth_computation = pn.widgets.Button(name="PSTH Computation", button_type="primary", width=300, align="end")
    mark_down_visualization = pn.pane.Markdown("""**Step 6 : Visualization**""", width=300)
    open_visualization = pn.widgets.Button(name="Open Visualization GUI", button_type="primary", width=300, align="end")
    open_terminal = pn.widgets.Button(name="Open Terminal", button_type="primary", width=300, align="end")

    save_button.on_click(onclickProcess)
    open_storesList.on_click(onclickStoresList)
    read_rawData.on_click(onclickreaddata)
    extract_ts.on_click(onclickextractts)
    psth_computation.on_click(onclickpsth)
    open_visualization.on_click(onclickVisualization)

    template.sidebar.append(mark_down_ip)
    template.sidebar.append(mark_down_ip_note)
    template.sidebar.append(save_button)
    # template.sidebar.append(path)
    template.sidebar.append(mark_down_storenames)
    template.sidebar.append(open_storesList)
    template.sidebar.append(mark_down_read)
    template.sidebar.append(read_rawData)
    template.sidebar.append(read_progress)
    template.sidebar.append(mark_down_extract)
    template.sidebar.append(extract_ts)
    template.sidebar.append(extract_progress)
    template.sidebar.append(mark_down_psth)
    template.sidebar.append(psth_computation)
    template.sidebar.append(psth_progress)
    template.sidebar.append(mark_down_visualization)
    template.sidebar.append(open_visualization)
    # template.sidebar.append(open_terminal)

    psth_baseline_param = pn.Column(zscore_param_wd, psth_param_wd, baseline_param_wd, peak_param_wd)

    widget = pn.Column(
        mark_down_1, files_1, explain_modality, modality_selector, pn.Row(individual_analysis_wd_2, psth_baseline_param)
    )

    # file_selector = pn.WidgetBox(files_1)
    styles = dict(background="WhiteSmoke")
    individual = pn.Card(widget, title="Individual Analysis", styles=styles, width=1000)
    group = pn.Card(group_analysis_wd_1, title="Group Analysis", styles=styles, width=1000)
    visualize = pn.Card(visualization_wd, title="Visualization Parameters", styles=styles, width=1000)

    # template.main.append(file_selector)
    template.main.append(individual)
    template.main.append(group)
    template.main.append(visualize)

    # Expose minimal hooks and widgets to enable programmatic testing
    template._hooks = {
        "onclickProcess": onclickProcess,
        "getInputParameters": getInputParameters,
    }
    template._widgets = {
        "files_1": files_1,
    }

    return template
