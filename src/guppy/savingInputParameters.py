import os
import sys
import time
import subprocess
import json
import panel as pn 
import numpy as np
import pandas as pd
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from threading import Thread
from pathlib import Path
from .visualizePlot import visualizeResults
from .saveStoresList import execute

def savingInputParameters():
    pn.extension()

    log_file = os.path.join(Path.home(), 'guppy.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    else:
        pass

    # Create the main window
    folder_selection = tk.Tk()
    folder_selection.title("Select the folder path where your data is located")
    folder_selection.geometry("700x200")
    def select_folder():
        global folder_path
        folder_path = filedialog.askdirectory(title="Select the folder path where your data is located")
        if folder_path:
            print(f"Folder path set to {folder_path}")
            folder_selection.destroy()
        else:
            folder_path = os.path.expanduser('~')
            print(f"Folder path set to {folder_path}")

    select_button = ttk.Button(folder_selection, text="Select a Folder", command=select_folder)
    select_button.pack(pady=5)
    folder_selection.mainloop()

    current_dir = os.getcwd()

    def insertLog(text, level):
        file = os.path.join(Path.home(), 'guppy.log')
        format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        infoLog = logging.FileHandler(file)
        infoLog.setFormatter(format)
        logger = logging.getLogger(file)
        logger.setLevel(level)
        
        if not logger.handlers:
            logger.addHandler(infoLog)
            if level == logging.DEBUG:
                logger.debug(text)
            if level == logging.INFO:
                logger.info(text)
            if level == logging.ERROR:
                logger.exception(text)
            if level == logging.WARNING:
                logger.warning(text)

        infoLog.close()
        logger.removeHandler(infoLog)

    def make_dir(filepath):
        op = os.path.join(filepath, 'inputParameters')
        if not os.path.exists(op):
            os.mkdir(op)
        return op

    def readRawData():
        inputParameters = getInputParameters()
        subprocess.call([sys.executable, "-m", "guppy.readTevTsq", json.dumps(inputParameters)])

    def extractTs():
        inputParameters = getInputParameters()
        subprocess.call([sys.executable, "-m", "guppy.preprocess", json.dumps(inputParameters)])

    def psthComputation():
        inputParameters = getInputParameters()
        inputParameters['curr_dir'] = current_dir
        subprocess.call([sys.executable, "-m", "guppy.computePsth", json.dumps(inputParameters)])


    def readPBIncrementValues(progressBar):
        print("Read progress bar increment values function started...")
        file_path = os.path.join(os.path.expanduser('~'), 'pbSteps.txt')
        if os.path.exists(file_path):
            os.remove(file_path)
        increment, maximum = 0, 100
        progressBar.value = increment
        progressBar.bar_color = 'success'
        while True:
            try:
                with open(file_path, 'r') as file:
                    content = file.readlines()
                    if len(content)==0:
                        pass
                    else:
                        maximum = int(content[0])
                        increment = int(content[-1])
                
                        if increment==-1:
                            progressBar.bar_color = 'danger'
                            os.remove(file_path)
                            break
                        progressBar.max = maximum
                        progressBar.value = increment
                time.sleep(0.001) 
            except FileNotFoundError:
                time.sleep(0.001)
            except PermissionError:
                time.sleep(0.001) 
            except Exception as e:
                # Handle other exceptions that may occur
                print(f"An error occurred while reading the file: {e}")
                break
            if increment==maximum:
                os.remove(file_path)
                break

        print("Read progress bar increment values stopped.")
        
    # progress bars = PB
    read_progress = pn.indicators.Progress(name='Progress', value=100, max=100, width=200, sizing_mode="stretch_width")
    extract_progress = pn.indicators.Progress(name='Progress', value=100, max=100, width=200, sizing_mode="stretch_width")
    psth_progress = pn.indicators.Progress(name='Progress', value=100, max=100, width=200, sizing_mode="stretch_width")


    template = pn.template.MaterialTemplate(title='Input Parameters GUI')

    mark_down_1 = pn.pane.Markdown("""**Select folders for the analysis from the file selector below**""", width=600)

    files_1 = pn.widgets.FileSelector(folder_path, name='folderNames', height=300, width=800)


    explain_time_artifacts = pn.pane.Markdown("""
                                                - ***Number of cores :*** Number of cores used for analysis. Try to 
                                                keep it less than the number of cores in your machine. 
                                                - ***Combine Data? :*** Make this parameter ``` True ``` if user wants to combine 
                                                the data, especially when there is two different 
                                                data files for the same recording session.<br>
                                                - ***Isosbestic Control Channel? :*** Make this parameter ``` False ``` if user
                                                does not want to use isosbestic control channel in the analysis.<br>
                                                - ***Eliminate first few seconds :*** It is the parameter to cut out first x seconds
                                                from the data. Default is 1 seconds.<br>
                                                - ***Window for Moving Average filter :*** The filtering of signals
                                                is done using moving average filter. Default window used for moving 
                                                average filter is 100 datapoints. Change it based on the requirement.<br>
                                                - ***Moving Window (transients detection) :*** Transients in the z-score 
                                                and/or \u0394F/F are detected using this moving window. 
                                                Default is 15 seconds. Change it based on the requirement.<br>
                                                - ***High Amplitude filtering threshold (HAFT) (transients detection) :*** High amplitude
                                                events greater than x times the MAD above the median are filtered out. Here, x is 
                                                high amplitude filtering threshold. Default is 2.
                                                - ***Transients detection threshold (TD Thresh):*** Peaks with local maxima greater than x times
                                                the MAD above the median of the trace (after filtering high amplitude events) are detected
                                                as transients. Here, x is transients detection threshold. Default is 3.
                                                - ***Number of channels (Neurophotometrics only) :*** Number of
                                                channels used while recording, when data files has no column names mentioning "Flags" 
                                                or "LedState".
                                                - ***removeArtifacts? :*** Make this parameter ``` True``` if there are 
                                                artifacts and user wants to remove the artifacts.
                                                - ***removeArtifacts method :*** Selecting ```concatenate``` will remove bad 
                                                chunks and concatenate the selected good chunks together.
                                                Selecting ```replace with NaN``` will replace bad chunks with NaN
                                                values.
                                                """)

    timeForLightsTurnOn = pn.widgets.LiteralInput(name='Eliminate first few seconds (int)', value=1, type=int, width=250)

    isosbestic_control = pn.widgets.Select(name='Isosbestic Control Channel? (bool)', value=True, options=[True, False], width=250)

    numberOfCores = pn.widgets.LiteralInput(name='# of cores (int)', value=2, type=int, width=100)

    combine_data = pn.widgets.Select(name='Combine Data? (bool)', value=False, options=[True, False], width=125)

    computePsth = pn.widgets.Select(name='z_score and/or \u0394F/F? (psth)', options=['z_score', 'dff', 'Both'], width=250)

    transients = pn.widgets.Select(name='z_score and/or \u0394F/F? (transients)', options=['z_score', 'dff', 'Both'], width=250)

    plot_zScore_dff = pn.widgets.Select(name='z-score plot and/or \u0394F/F plot?', options=['z_score', 'dff', 'Both', 'None'], value='None', width=250)

    moving_wd = pn.widgets.LiteralInput(name='Moving Window for transients detection (s) (int)', value=15, type=int, width=250)

    highAmpFilt = pn.widgets.LiteralInput(name='HAFT (int)', value=2, type=int, width=120)

    transientsThresh = pn.widgets.LiteralInput(name='TD Thresh (int)', value=3, type=int, width=120)

    moving_avg_filter = pn.widgets.LiteralInput(name='Window for Moving Average filter (int)', value=100, type=int, width=250)

    removeArtifacts = pn.widgets.Select(name='removeArtifacts? (bool)', value=False, options=[True, False], width=125)

    artifactsRemovalMethod = pn.widgets.Select(name='removeArtifacts method', 
                                            value='concatenate', 
                                            options=['concatenate', 'replace with NaN'],
                                            width=100)

    no_channels_np = pn.widgets.LiteralInput(name='Number of channels (Neurophotometrics only)',
                                            value=2, type=int, width=250)

    z_score_computation = pn.widgets.Select(name='z-score computation Method', 
                                            options=['standard z-score', 'baseline z-score', 'modified z-score'], 
                                            value='standard z-score', width=200)
    baseline_wd_strt = pn.widgets.LiteralInput(name='Baseline Window Start Time (s) (int)', value=0, type=int, width=200)
    baseline_wd_end = pn.widgets.LiteralInput(name='Baseline Window End Time (s) (int)', value=0, type=int, width=200)

    explain_z_score = pn.pane.Markdown("""
                                    ***Note :***<br>
                                    - Details about z-score computation methods are explained in Github wiki.<br>
                                    - The details will make user understand what computation method to use for 
                                    their data.<br>
                                    - Baseline Window Parameters should be kept 0 unless you are using baseline<br> 
                                    z-score computation method. The parameters are in seconds.
                                    """, width=500)

    explain_nsec = pn.pane.Markdown("""
                                    - ***Time Interval :*** To omit bursts of event timestamps, user defined time interval
                                    is set so that if the time difference between two timestamps is less than this defined time
                                    interval, it will be deleted for the calculation of PSTH.
                                    - ***Compute Cross-correlation :*** Make this parameter ```True```, when user wants
                                    to compute cross-correlation between PSTHs of two different signals or signals 
                                    recorded from different brain regions.
                                    """, width=500)

    nSecPrev = pn.widgets.LiteralInput(name='Seconds before 0 (int)', value=-10, type=int,  width=120)

    nSecPost = pn.widgets.LiteralInput(name='Seconds after 0 (int)', value=20, type=int,  width=120)

    computeCorr = pn.widgets.Select(name='Compute Cross-correlation (bool)', 
                                            options=[True, False], 
                                            value=False, width=160)

    timeInterval = pn.widgets.LiteralInput(name='Time Interval (s)', value=2, type=int,  width=120)

    use_time_or_trials = pn.widgets.Select(name='Bin PSTH trials (str)', 
                                        options = ['Time (min)', '# of trials'],
                                        value='Time (min)', width=120)

    bin_psth_trials = pn.widgets.LiteralInput(name='Time(min) / # of trials \n for binning? (int)', value=0, type=int,  width=160)

    explain_baseline = pn.pane.Markdown("""
                                        ***Note :***<br>
                                        - If user does not want to do baseline correction, 
                                        put both parameters 0.<br>
                                        - If the first event timestamp is less than the length of baseline
                                        window, it will be rejected in the PSTH computation step.<br>
                                        - Baseline parameters must be within the PSTH parameters 
                                        set in the PSTH parameters section.
                                        """, width=500)

    baselineCorrectionStart = pn.widgets.LiteralInput(name='Baseline Correction Start time(int)', value=-5, type=int, width=200)

    baselineCorrectionEnd = pn.widgets.LiteralInput(name='Baseline Correction End time(int)', value=0, type=int, width=200)

    zscore_param_wd = pn.WidgetBox("### Z-score Parameters", explain_z_score,
                                                            z_score_computation,
                                                            pn.Row(baseline_wd_strt, baseline_wd_end),
                                                            width=500, height=350)

    psth_param_wd = pn.WidgetBox("### PSTH Parameters", explain_nsec, 
                                                        pn.Row(nSecPrev, nSecPost, computeCorr), 
                                                        pn.Row(timeInterval, use_time_or_trials, bin_psth_trials), 
                                                        width=500, height=350)

    baseline_param_wd = pn.WidgetBox("### Baseline Parameters", explain_baseline, 
                                    pn.Row(baselineCorrectionStart, baselineCorrectionEnd), 
                                    width=500, height=300)

    peak_explain = pn.pane.Markdown("""
                                    ***Note :***<br>
                                    - Peak and area are computed between the window set below.<br>
                                    - Peak and AUC parameters must be within the PSTH parameters set in the PSTH parameters section.<br>
                                    - Please make sure when user changes the parameters in the table below, click on any other cell after 
                                    changing a value in a particular cell.
                                    """, width=500)


    start_end_point_df = pd.DataFrame({'Peak Start time': [-5, 0, 5, np.nan, np.nan, 
                                                        np.nan, np.nan, np.nan, np.nan, np.nan], 
                                    'Peak End time': [0, 3, 10, np.nan, np.nan, 
                                                    np.nan, np.nan, np.nan, np.nan, np.nan]})

    df_widget = pn.widgets.Tabulator(start_end_point_df, name='DataFrame', show_index=False, row_height=20, width=450)


    peak_param_wd = pn.WidgetBox("### Peak and AUC Parameters", 
                                peak_explain, df_widget,
                                height=400)



    mark_down_2 = pn.pane.Markdown("""**Select folders for the average analysis from the file selector below**""", width=600)

    files_2 = pn.widgets.FileSelector(folder_path, name='folderNamesForAvg', height=300, width=800)

    averageForGroup = pn.widgets.Select(name='Average Group? (bool)', value=False, options=[True, False], width=400)

    visualizeAverageResults = pn.widgets.Select(name='Visualize Average Results? (bool)', 
                                                value=False, options=[True, False], width=400)

    visualize_zscore_or_dff = pn.widgets.Select(name='z-score or \u0394F/F? (for visualization)', options=['z_score', 'dff'], width=400)

    individual_analysis_wd_2 = pn.Column(
                                        explain_time_artifacts, pn.Row(numberOfCores, combine_data), 
                                        isosbestic_control, timeForLightsTurnOn,
                                        moving_avg_filter, computePsth, transients, plot_zScore_dff, 
                                        moving_wd, pn.Row(highAmpFilt, transientsThresh),
                                        no_channels_np, pn.Row(removeArtifacts, artifactsRemovalMethod)
                                        )

    group_analysis_wd_1 = pn.Column(mark_down_2, files_2, averageForGroup, width=800)

    visualization_wd = pn.Row(visualize_zscore_or_dff, visualizeAverageResults, width=800)


    def getInputParameters():
        abspath = getAbsPath()
        inputParameters = {
            "abspath": abspath[0],
            "folderNames": files_1.value,
            "numberOfCores": numberOfCores.value,
            "combine_data": combine_data.value,
            "isosbestic_control": isosbestic_control.value,
            "timeForLightsTurnOn": timeForLightsTurnOn.value,
            "filter_window": moving_avg_filter.value,
            "removeArtifacts": removeArtifacts.value,
            "artifactsRemovalMethod": artifactsRemovalMethod.value,
            "noChannels": no_channels_np.value,
            "zscore_method": z_score_computation.value,
            "baselineWindowStart": baseline_wd_strt.value,
            "baselineWindowEnd": baseline_wd_end.value,
            "nSecPrev": nSecPrev.value,
            "nSecPost": nSecPost.value,
            "computeCorr": computeCorr.value,
            "timeInterval": timeInterval.value,
            "bin_psth_trials": bin_psth_trials.value,
            "use_time_or_trials": use_time_or_trials.value,
            "baselineCorrectionStart": baselineCorrectionStart.value,
            "baselineCorrectionEnd": baselineCorrectionEnd.value,
            "peak_startPoint": list(df_widget.value['Peak Start time']), #startPoint.value,
            "peak_endPoint": list(df_widget.value['Peak End time']), #endPoint.value,
            "selectForComputePsth": computePsth.value,
            "selectForTransientsComputation": transients.value,
            "moving_window": moving_wd.value,
            "highAmpFilt": highAmpFilt.value,
            "transientsThresh": transientsThresh.value,
            "plot_zScore_dff": plot_zScore_dff.value,
            "visualize_zscore_or_dff": visualize_zscore_or_dff.value,
            "folderNamesForAvg": files_2.value,
            "averageForGroup": averageForGroup.value,
            "visualizeAverageResults": visualizeAverageResults.value
        }
        return inputParameters

    def checkSameLocation(arr, abspath):
        #abspath = []
        for i in range(len(arr)):
            abspath.append(os.path.dirname(arr[i]))
        abspath = np.asarray(abspath)
        abspath = np.unique(abspath)
        if len(abspath)>1:
            insertLog('All the folders selected should be at the same location', 
                    logging.ERROR)
            raise Exception('All the folders selected should be at the same location')
        
        return abspath

    def getAbsPath():
        arr_1, arr_2 = files_1.value, files_2.value 
        if len(arr_1)==0 and len(arr_2)==0:
            insertLog('No folder is selected for analysis',
                    logging.ERROR)
            raise Exception('No folder is selected for analysis')
        
        abspath = []
        if len(arr_1)>0:
            abspath = checkSameLocation(arr_1, abspath)
        else:
            abspath = checkSameLocation(arr_2, abspath)
        
        abspath = np.unique(abspath)
        if len(abspath)>1:
            insertLog('All the folders selected should be at the same location',
                    logging.ERROR)
            raise Exception('All the folders selected should be at the same location')
        return abspath

    def onclickProcess(event=None):
        
        insertLog('Saving Input Parameters file.',
                logging.DEBUG)
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
            "peak_startPoint": list(df_widget.value['Peak Start time']), #startPoint.value,
            "peak_endPoint": list(df_widget.value['Peak End time']), #endPoint.value,
            "selectForComputePsth": computePsth.value,
            "selectForTransientsComputation": transients.value,
            "moving_window": moving_wd.value,
            "highAmpFilt": highAmpFilt.value,
            "transientsThresh": transientsThresh.value   
        }
        for folder in files_1.value:
            with open(os.path.join(folder, 'GuPPyParamtersUsed.json'), 'w') as f:
                json.dump(analysisParameters, f, indent=4)
            insertLog(f"Input Parameters file saved at {folder}",
                    logging.INFO)
        
        insertLog('#'*400, logging.INFO)
                
        #path.value = (os.path.join(op, 'inputParameters.json')).replace('\\', '/')
        print('Input Parameters File Saved.')

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
        

        
    mark_down_ip = pn.pane.Markdown("""**Step 1 : Save Input Parameters**""", width=500)
    mark_down_ip_note = pn.pane.Markdown("""***Note : ***<br>
                                            - Save Input Parameters will save input parameters used for the analysis
                                            in all the folders you selected for the analysis (useful for future
                                            reference). All analysis steps will run without saving input parameters.
                                        """, width=500, sizing_mode="stretch_width")
    save_button = pn.widgets.Button(name='Save to file...', button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    mark_down_storenames = pn.pane.Markdown("""**Step 2 : Open Storenames GUI <br> and save storenames**""", width=500)
    open_storesList = pn.widgets.Button(name='Open Storenames GUI', button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    mark_down_read = pn.pane.Markdown("""**Step 3 : Read Raw Data**""", width=500)
    read_rawData = pn.widgets.Button(name='Read Raw Data', button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    mark_down_extract = pn.pane.Markdown("""**Step 4 : Extract timestamps <br> and its correction**""", width=500)
    extract_ts = pn.widgets.Button(name="Extract timestamps and it's correction", button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    mark_down_psth = pn.pane.Markdown("""**Step 5 : PSTH Computation**""", width=500)
    psth_computation = pn.widgets.Button(name="PSTH Computation", button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    mark_down_visualization = pn.pane.Markdown("""**Step 6 : Visualization**""", width=500)
    open_visualization = pn.widgets.Button(name='Open Visualization GUI', button_type='primary', width=500, sizing_mode="stretch_width", align='end')
    open_terminal = pn.widgets.Button(name='Open Terminal', button_type='primary', width=500, sizing_mode="stretch_width", align='end')


    save_button.on_click(onclickProcess)
    open_storesList.on_click(onclickStoresList)
    read_rawData.on_click(onclickreaddata)
    extract_ts.on_click(onclickextractts)
    psth_computation.on_click(onclickpsth)
    open_visualization.on_click(onclickVisualization)


    template.sidebar.append(mark_down_ip)
    template.sidebar.append(mark_down_ip_note)
    template.sidebar.append(save_button)
    #template.sidebar.append(path)
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
    #template.sidebar.append(open_terminal)


    psth_baseline_param = pn.Column(zscore_param_wd, psth_param_wd, baseline_param_wd, peak_param_wd)

    widget = pn.Column(mark_down_1, files_1, pn.Row(individual_analysis_wd_2, psth_baseline_param))

    #file_selector = pn.WidgetBox(files_1)
    styles = dict(background='WhiteSmoke')
    individual = pn.Card(widget, title='Individual Analysis', styles=styles, width=850)
    group = pn.Card(group_analysis_wd_1, title='Group Analysis', styles=styles, width=850)
    visualize = pn.Card(visualization_wd, title='Visualization Parameters', styles=styles, width=850)

    #template.main.append(file_selector)
    template.main.append(individual)
    template.main.append(group)
    template.main.append(visualize)

    return template
