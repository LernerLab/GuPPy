import json
import logging
import os
import subprocess
import sys
from threading import Thread

import panel as pn

from ..frontend.input_parameters import ParameterForm
from ..frontend.path_selection import get_folder_path
from ..frontend.progress import readPBIncrementValues
from ..frontend.sidebar import Sidebar
from ..saveStoresList import execute
from ..visualizePlot import visualizeResults

logger = logging.getLogger(__name__)


def readRawData(parameter_form):
    inputParameters = parameter_form.getInputParameters()
    subprocess.call([sys.executable, "-m", "guppy.readTevTsq", json.dumps(inputParameters)])


def extractTs(parameter_form):
    inputParameters = parameter_form.getInputParameters()
    subprocess.call([sys.executable, "-m", "guppy.preprocess", json.dumps(inputParameters)])


def psthComputation(parameter_form, current_dir):
    inputParameters = parameter_form.getInputParameters()
    inputParameters["curr_dir"] = current_dir
    subprocess.call([sys.executable, "-m", "guppy.computePsth", json.dumps(inputParameters)])


def build_homepage():
    pn.extension()
    global folder_path
    folder_path = get_folder_path()
    current_dir = os.getcwd()

    template = pn.template.BootstrapTemplate(title="Input Parameters GUI")
    parameter_form = ParameterForm(folder_path=folder_path, template=template)
    sidebar = Sidebar(template=template)

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions for sidebar buttons
    def onclickProcess(event=None):
        inputParameters = parameter_form.getInputParameters()
        logger.debug("Saving Input Parameters file.")
        analysisParameters = {
            "combine_data": inputParameters["combine_data"],
            "isosbestic_control": inputParameters["isosbestic_control"],
            "timeForLightsTurnOn": inputParameters["timeForLightsTurnOn"],
            "filter_window": inputParameters["filter_window"],
            "removeArtifacts": inputParameters["removeArtifacts"],
            "noChannels": inputParameters["noChannels"],
            "zscore_method": inputParameters["zscore_method"],
            "baselineWindowStart": inputParameters["baselineWindowStart"],
            "baselineWindowEnd": inputParameters["baselineWindowEnd"],
            "nSecPrev": inputParameters["nSecPrev"],
            "nSecPost": inputParameters["nSecPost"],
            "timeInterval": inputParameters["timeInterval"],
            "bin_psth_trials": inputParameters["bin_psth_trials"],
            "use_time_or_trials": inputParameters["use_time_or_trials"],
            "baselineCorrectionStart": inputParameters["baselineCorrectionStart"],
            "baselineCorrectionEnd": inputParameters["baselineCorrectionEnd"],
            "peak_startPoint": inputParameters["peak_startPoint"],
            "peak_endPoint": inputParameters["peak_endPoint"],
            "selectForComputePsth": inputParameters["selectForComputePsth"],
            "selectForTransientsComputation": inputParameters["selectForTransientsComputation"],
            "moving_window": inputParameters["moving_window"],
            "highAmpFilt": inputParameters["highAmpFilt"],
            "transientsThresh": inputParameters["transientsThresh"],
        }
        for folder in inputParameters["folderNames"]:
            with open(os.path.join(folder, "GuPPyParamtersUsed.json"), "w") as f:
                json.dump(analysisParameters, f, indent=4)
            logger.info(f"Input Parameters file saved at {folder}")

        logger.info("#" * 400)

        # path.value = (os.path.join(op, 'inputParameters.json')).replace('\\', '/')
        logger.info("Input Parameters File Saved.")

    def onclickStoresList(event=None):
        inputParameters = parameter_form.getInputParameters()
        execute(inputParameters)

    def onclickVisualization(event=None):
        inputParameters = parameter_form.getInputParameters()
        visualizeResults(inputParameters)

    def onclickreaddata(event=None):
        thread = Thread(target=readRawData, args=(parameter_form,))
        thread.start()
        readPBIncrementValues(sidebar.read_progress)
        thread.join()

    def onclickextractts(event=None):
        thread = Thread(target=extractTs, args=(parameter_form,))
        thread.start()
        readPBIncrementValues(sidebar.extract_progress)
        thread.join()

    def onclickpsth(event=None):
        thread = Thread(target=psthComputation, args=(parameter_form, current_dir))
        thread.start()
        readPBIncrementValues(sidebar.psth_progress)
        thread.join()

    # ------------------------------------------------------------------------------------------------------------------

    button_name_to_onclick_fn = {
        "save_button": onclickProcess,
        "open_storesList": onclickStoresList,
        "read_rawData": onclickreaddata,
        "extract_ts": onclickextractts,
        "psth_computation": onclickpsth,
        "open_visualization": onclickVisualization,
    }
    sidebar.attach_callbacks(button_name_to_onclick_fn=button_name_to_onclick_fn)
    sidebar.add_to_template()

    # Expose minimal hooks and widgets to enable programmatic testing
    template._hooks = {
        "onclickProcess": onclickProcess,
        "getInputParameters": parameter_form.getInputParameters,
    }
    template._widgets = {
        "files_1": parameter_form.files_1,
    }

    return template
