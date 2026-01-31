#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os

import panel as pn

from guppy.extractors import (
    CsvRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    TdtRecordingExtractor,
)
from guppy.frontend.npm_gui_prompts import (
    get_multi_event_responses,
    get_timestamp_configuration,
)
from guppy.frontend.temp import saveStorenames

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


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
            folder_path = os.path.join(inputParameters["abspath"], i)
            if modality == "tdt":
                events, flags = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
            elif modality == "csv":
                events, flags = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)

            elif modality == "doric":
                events, flags = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)

            elif modality == "npm":
                headless = bool(os.environ.get("GUPPY_BASE_DIR"))
                if not headless:
                    # Resolve multiple event TTLs
                    multiple_event_ttls = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=folder_path)
                    responses = get_multi_event_responses(multiple_event_ttls)
                    inputParameters["npm_split_events"] = responses

                    # Resolve timestamp units and columns
                    ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(
                        folder_path=folder_path, num_ch=num_ch
                    )
                    ts_units, npm_timestamp_column_names = get_timestamp_configuration(ts_unit_needs, col_names_ts)
                    inputParameters["npm_time_units"] = ts_units if ts_units else None
                    inputParameters["npm_timestamp_column_names"] = (
                        npm_timestamp_column_names if npm_timestamp_column_names else None
                    )

                events, flags = NpmRecordingExtractor.discover_events_and_flags(
                    folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
                )
            else:
                raise ValueError("Modality not recognized. Please use 'tdt', 'csv', 'doric', or 'npm'.")

            saveStorenames(inputParameters, events, flags, folder_path)
        logger.info("#" * 400)
    except Exception as e:
        logger.error(str(e))
        raise e
