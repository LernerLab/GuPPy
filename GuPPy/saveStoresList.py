#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import glob
import h5py
import numpy as np
import pandas as pd
from numpy import int32, uint32, uint8, uint16, float64, int64, int32, float32
import panel as pn
from random import randint
from pathlib import Path
import holoviews as hv
import warnings
import logging
import tkinter as tk
from tkinter import ttk, StringVar, messagebox

#hv.extension()
pn.extension()

def insertLog(text, level):
    file = os.path.join('.','..','guppy.log')
    format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    infoLog = logging.FileHandler(file)
    infoLog.setFormatter(format)
    infoLog
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

# function to show location for over-writing or creating a new stores list file.
def show_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename+'_output_'+str(i))
        if not os.path.exists(op):
            break
        i += 1
    return op

def make_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename+'_output_'+str(i))
        if not os.path.exists(op):
            os.mkdir(op)
            break
        i += 1

    return op

def check_header(df):
    arr = list(df.columns)
    check_float = []
    for i in arr:
        try:
            check_float.append(float(i))
        except:
            pass
    
    return arr, check_float


# function to read 'tsq' file
def readtsq(filepath):
    names = ('size', 'type', 'name', 'chan', 'sort_code', 'timestamp',
            'fp_loc', 'strobe', 'format', 'frequency')
    formats = (int32, int32, 'S4', uint16, uint16, float64, int64,
               float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({'names': names, 'formats': formats,
                          'offsets': offsets}, align=True)
    path = glob.glob(os.path.join(filepath, '*.tsq'))
    if len(path)>1:
        insertLog('Two tsq files are present at the location.',
                  logging.ERROR)
        raise Exception('Two tsq files are present at the location.')
    elif len(path)==0:
        return 0
    else:
        path = path[0]
    tsq = np.fromfile(path, dtype=tsq_dtype)
    df = pd.DataFrame(tsq)
    return df


# function to show GUI and save 
def saveStorenames(inputParameters, data, event_name, flag, filepath):
    
    insertLog('Saving stores list file.',
                  logging.DEBUG)
    # getting input parameters
    inputParameters = inputParameters

    # reading storenames from the data fetched using 'readtsq' function
    if isinstance(data, pd.DataFrame):
        data['name'] = np.asarray(data['name'], dtype=str)
        allnames = np.unique(data['name'])
        index = []
        for i in range(len(allnames)):
            length = len(str(allnames[i]))
            if length<4:
                index.append(i)
        allnames = np.delete(allnames, index, 0)
        allnames = list(allnames)

    else:
        allnames = []


    if 'data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag:
        path_chev = glob.glob(os.path.join(filepath, '*chev*'))
        path_chod = glob.glob(os.path.join(filepath, '*chod*'))
        path_chpr = glob.glob(os.path.join(filepath, '*chpr*'))
        combine_paths = path_chev + path_chod + path_chpr
        d = dict()
        for i in range(len(combine_paths)):
            basename = (os.path.basename(combine_paths[i])).split('.')[0]
            df = pd.read_csv(combine_paths[i])
            d[basename] = {
                    'x': np.array(df['timestamps']),
                    'y': np.array(df['data'])
            }
        keys = list(d.keys())
        mark_down_np = pn.pane.Markdown("""
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

                                            """)
        plot_select = pn.widgets.Select(name='Select channel to see correspondings channels', options=keys, value=keys[0])
        
        @pn.depends(plot_select=plot_select)
        def plot(plot_select):
            return hv.Curve((d[plot_select]['x'], d[plot_select]['y'])).opts(width=550)
    else:
        pass

    # finalizing all the storenames 
    allnames = allnames + event_name


    # instructions about how to save the storeslist file
    mark_down = pn.pane.Markdown("""


                ### Instructions to follow : 

                - Check Storenames to repeat checkbox and see instructions in “Github Wiki” for duplicating storenames. 
                Otherwise do not check the Storenames to repeat checkbox.<br>
                - Select storenames from list and click “Select Storenames” to populate area below.<br>
                - Enter names for storenames, in order, using the following naming convention:<br>
                    Isosbestic = “control_region” (ex: Dv1A= control_DMS)<br>
                    Signal= “signal_region” (ex: Dv2A= signal_DMS)<br>
                    TTLs can be named using any convention (ex: PrtR = RewardedPortEntries) but should be kept consistent for later group analysis
                    
                ```{.python}
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

                """, width=550)


    # creating GUI template
    template = pn.template.MaterialTemplate(title='Storenames GUI - {}'.format(os.path.basename(filepath), mark_down))

    

    # creating different buttons and selectors for the GUI
    cross_selector = pn.widgets.CrossSelector(name='Store Names Selection', value=[], options=allnames)
    multi_choice = pn.widgets.MultiChoice(name='Select Storenames which you want more than once (multi-choice: multiple options selection)', value=[], options=allnames)

    literal_input_1 = pn.widgets.LiteralInput(name='Number of times you want the above storename (list)', value=[], type=list)
    #literal_input_2 = pn.widgets.LiteralInput(name='Names for Storenames (list)', type=list)

    repeat_storenames = pn.widgets.Checkbox(name='Storenames to repeat', value=False)
    repeat_storename_wd = pn.WidgetBox('', background='white', width=600)
    def callback(target, event):
        if event.new==True:
            target.objects = [multi_choice, literal_input_1]
        elif event.new==False:
            target.clear()
    repeat_storenames.link(repeat_storename_wd, callbacks={'value': callback})
    #repeat_storename_wd = pn.WidgetBox('Storenames to repeat (leave blank if not needed)', multi_choice, literal_input_1, background="white", width=600)

    update_options = pn.widgets.Button(name='Select Storenames')
    save = pn.widgets.Button(name='Save')

    text = pn.widgets.LiteralInput(value=[], name='Selected Store Names', type=list)

    path = pn.widgets.TextInput(name='Location to Stores List file', width=500, sizing_mode="stretch_width")

    mark_down_for_overwrite = pn.pane.Markdown(""" Select option from below if user wants to over-write a file or create a new file. 
                                    **Creating a new file will make a new ouput folder and will get saved at that location.**
                                    If user selects to over-write a file **Select location of the file to over-write** will provide 
                                    the existing options of the ouput folders where user needs to over-write the file""")

    select_location = pn.widgets.Select(name='Select location of the file to over-write', value='None', options=['None'], align='end')


    overwrite_button = pn.widgets.MenuButton(name='over-write storeslist file or create a new one?  ', items=['over_write_file', 'create_new_file'], button_type='default', split=True, align='end')
    
    literal_input_2 = pn.widgets.Ace(value="""{}""", sizing_mode='stretch_both', theme='tomorrow', language='json', height=250)

    alert = pn.pane.Alert('#### No alerts !!', alert_type='danger', height=80)


    take_widgets = pn.WidgetBox(
        multi_choice,
        literal_input_1   
    )

    change_widgets = pn.WidgetBox(
        text
    )

    
    storenames = []
    
    if len(allnames)==0:
        alert.object = '####Alert !! \n No storenames found. There are not any TDT files or csv files to look for storenames.'

    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event):
        if event.new=='over_write_file':
            select_location.options = glob.glob(os.path.join(filepath, '*_output_*'))
            #select_location.value = select_location.options[0]
        else:
            select_location.options = [show_dir(filepath)]
            #select_location.value = select_location.options[0]
    
    def fetchValues():
        alert.object = '#### No alerts !!'
        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), '.storesList.json')):
            with open(os.path.join(Path.home(), '.storesList.json')) as f:
                storenames_cache = json.load(f)
        comboBox_keys = list(hold_comboBoxValues.keys())
        textBox_keys = list(hold_textBoxValues.keys())
        
        comboBoxValues, textBoxValues = [], []
        for i in range(len(comboBox_keys)):
            comboBoxValues.append(hold_comboBoxValues[comboBox_keys[i]].get())
        
        for i in range(len(textBox_keys)):
            textBoxValues.append(hold_textBoxValues[textBox_keys[i]].get())
            if len(textBoxValues[i].split())>1:
                alert.object = '####Alert !! \n Whitespace is not allowed in the text box entry.'
            if textBoxValues[i]==None and comboBoxValues[i] not in storenames_cache:
                print(textBoxValues[i], comboBoxValues[i])
                alert.object = '####Alert !! \n One of the text box entry is empty.'
    
        if len(comboBoxValues)!=len(textBoxValues):
            alert.object = '####Alert !! \n Number of entries in combo box and text box should be same.'
        
        names_for_storenames = []
        for i in range(len(comboBoxValues)):
            if comboBoxValues[i]=='control' or comboBoxValues[i]=="signal":
                if '_' in textBoxValues[i]:
                    messagebox.showwarning("Warning", "Please do not use underscore in region name")
                names_for_storenames.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
            elif comboBoxValues[i]=='event TTLs':
                names_for_storenames.append(textBoxValues[i])
            else:
                names_for_storenames.append(comboBoxValues[i])
        
        d = dict()
        print(text.value)
        d["storenames"] = text.value
        d["names_for_storenames"] = names_for_storenames
        literal_input_2.value = str(json.dumps(d))
    
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

        if len(new_arr)>0:
            storenames = cross_selector.value + new_arr
        else:
            storenames = cross_selector.value
        
        for w in change_widgets:
            w.value = storenames

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), '.storesList.json')):
            with open(os.path.join(Path.home(), '.storesList.json')) as f:
                storenames_cache = json.load(f)
        

        def comboBoxSelected(event):
            row, col = event.widget.grid_info()['row'], event.widget.grid_info()['column']
            if event.widget.get()=="control":
                label = ttk.Label(root, 
                                text="Type appropriate region name in the text box below :").grid(row=row, column=col+1)
            elif event.widget.get()=="signal":
                label = ttk.Label(root, 
                                text="Type appropriate region name in the text box below :").grid(row=row, column=col+1)
            elif event.widget.get()=="event TTLs":
                label = ttk.Label(root, 
                                text="Type event name for the TTLs in the text box below :").grid(row=row, column=col+1)
            else:
                pass
        
        global hold_comboBoxValues, hold_textBoxValues
        root = tk.Tk()
        root.title('Select options for storenames and give appropriate names (if asked)')
        root.geometry('1200x1000')
        hold_comboBoxValues = dict()
        hold_textBoxValues = dict()

        def stayOnTop():
            root.lift()
            root.after(2000, stayOnTop)

        for i in range(len(storenames)):
            if storenames[i] in storenames_cache:
                T = ttk.Label(root, text="Select appropriate option for {} : ".format(storenames[i])).grid(row=i+1, column=1)
                if storenames[i] in hold_comboBoxValues and storenames[i] in hold_textBoxValues:
                    hold_comboBoxValues[storenames[i]+'_'+str(i)] = StringVar()
                    hold_textBoxValues[storenames[i]+'_'+str(i)] = StringVar()
                    myCombo = ttk.Combobox(root, 
                                        textvariable=hold_comboBoxValues[storenames[i]+'_'+str(i)],
                                        value=storenames_cache[storenames[i]], 
                                        width=20)
                else:
                    hold_comboBoxValues[storenames[i]] = StringVar()
                    hold_textBoxValues[storenames[i]] = StringVar()
                    myCombo = ttk.Combobox(root, 
                                        textvariable=hold_comboBoxValues[storenames[i]],
                                        value=storenames_cache[storenames[i]], 
                                        width=20)
                myCombo.grid(row=i+1, column=2)
                myCombo.current(0)
                myCombo.bind("<<ComboboxSelected>>", comboBoxSelected)
            else:
                T = ttk.Label(root, text="Select appropriate option for {} : ".format(storenames[i])).grid(row=i+1, column=1)
                if storenames[i] in hold_comboBoxValues and storenames[i] in hold_textBoxValues:
                    hold_comboBoxValues[storenames[i]+'_'+str(i)] = StringVar()
                    hold_textBoxValues[storenames[i]+'_'+str(i)] = StringVar()
                    myCombo = ttk.Combobox(root, 
                                    textvariable=hold_comboBoxValues[storenames[i]+'_'+str(i)],
                                    value=['', 'control', 'signal', 'event TTLs'], 
                                    width=12)
                    textBox = tk.Entry(root, 
                                    textvariable=hold_textBoxValues[storenames[i]+'_'+str(i)])
                else:
                    hold_comboBoxValues[storenames[i]] = StringVar()
                    hold_textBoxValues[storenames[i]] = StringVar()
                    myCombo = ttk.Combobox(root, 
                                        textvariable=hold_comboBoxValues[storenames[i]],
                                        value=['', 'control', 'signal', 'event TTLs'], 
                                        width=12)
                    textBox = tk.Entry(root, 
                                    textvariable=hold_textBoxValues[storenames[i]])
                myCombo.grid(row=i+1, column=2)
                textBox.grid(row=i+1, column=4)
                myCombo.current(0)
                myCombo.bind("<<ComboboxSelected>>", comboBoxSelected)

        note = ttk.Label(root, text="Note : Click on Show button after appropriate selections and close the window.").grid(row=(len(storenames)*2)+2, column=2)
        button = ttk.Button(root, text='Show', command=fetchValues).grid(row=(len(storenames)*2)+4, column=2)
        stayOnTop()   
        root.mainloop()



    # on clicking save button, following function is executed
    def save_button(event=None):
        global storenames
        
        d = json.loads(literal_input_2.value)
        arr1, arr2 = np.asarray(d["storenames"]), np.asarray(d["names_for_storenames"])

        if np.where(arr2=="")[0].size>0:
            alert.object = '#### Alert !! \n Empty string in the list names_for_storenames.'
            insertLog('Empty string in the list names_for_storenames.',
                      logging.ERROR)
            raise Exception('Empty string in the list names_for_storenames.')
        else:
            alert.object = '#### No alerts !!'

        if arr1.shape[0]!=arr2.shape[0]:
            alert.object = '#### Alert !! \n Length of list storenames and names_for_storenames is not equal.'
            insertLog('Length of list storenames and names_for_storenames is not equal.',
                      logging.ERROR)
            raise Exception('Length of list storenames and names_for_storenames is not equal.')
        else:
            alert.object = '#### No alerts !!'


        if not os.path.exists(os.path.join(Path.home(), '.storesList.json')):
            storenames_cache = dict()

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), '.storesList.json'), 'w') as f:
                json.dump(storenames_cache, f, indent=4) 
        else:
            with open(os.path.join(Path.home(), '.storesList.json')) as f:
                storenames_cache = json.load(f)

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), '.storesList.json'), 'w') as f:
                json.dump(storenames_cache, f, indent=4)

        arr = np.asarray([arr1, arr2])
        print(arr)
        if not os.path.exists(select_location.value):
            os.mkdir(select_location.value)
            
        np.savetxt(os.path.join(select_location.value, 'storesList.csv'), arr, delimiter=",", fmt='%s')
        path.value = os.path.join(select_location.value, 'storesList.csv')
        insertLog(f"Storeslist file saved at {select_location.value}",
                      logging.INFO)
        insertLog('Storeslist : \n'+str(arr), logging.INFO)
    

    update_options.on_click(update_values)
    save.on_click(save_button)
    overwrite_button.on_click(overwrite_button_actions)

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = randint(5000,5200)

    if 'data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag:
        widget_1 = pn.Column('# '+os.path.basename(filepath), mark_down, mark_down_np, plot_select, plot)
        widget_2 = pn.Column(repeat_storenames, repeat_storename_wd, pn.Spacer(height=20), 
                             cross_selector, update_options, 
                             text, literal_input_2, alert, mark_down_for_overwrite, 
                             overwrite_button, select_location, save, path)
        template.main.append(pn.Row(widget_1, widget_2))

    else:
        widget_1 = pn.Column('# '+os.path.basename(filepath), mark_down)
        widget_2 = pn.Column(repeat_storenames, repeat_storename_wd, pn.Spacer(height=20), 
                             cross_selector, update_options, 
                             text, literal_input_2, alert, mark_down_for_overwrite, 
                             overwrite_button, select_location, save, path)
        template.main.append(pn.Row(widget_1, widget_2))

    template.show(port=number)


# check flag consistency in neurophotometrics data
def check_channels(state):
    state = state.astype(int)
    unique_state = np.unique(state[2:12])
    if unique_state.shape[0]>3:
        insertLog("Looks like there are more than 3 channels in the file. Reading of these files\
                        are not supported. Reach out to us if you get this error message.",
                        logging.ERROR)
        raise Exception("Looks like there are more than 3 channels in the file. Reading of these files\
                        are not supported. Reach out to us if you get this error message.")

    return unique_state.shape[0], unique_state
    

# function to decide indices of interleaved channels
# in neurophotometrics data
def decide_indices(file, df, flag, num_ch=2):
    ch_name = [file+'chev', file+'chod', file+'chpr']
    if len(ch_name)<num_ch:
        insertLog('Number of channels parameters in Input Parameters GUI is more than 3. \
                    Looks like there are more than 3 channels in the file. Reading of these files\
                    are not supported. Reach out to us if you get this error message.',
                    logging.ERROR)
        raise Exception('Number of channels parameters in Input Parameters GUI is more than 3. \
                         Looks like there are more than 3 channels in the file. Reading of these files\
                         are not supported. Reach out to us if you get this error message.')
    if flag=='data_np':
        indices_dict = dict()
        for i in range(num_ch):
            indices_dict[ch_name[i]] = np.arange(i, df.shape[0], num_ch)

    else:
        cols = np.array(list(df.columns))
        if 'flags' in np.char.lower(np.array(cols)):
            arr = ['FrameCounter', 'Flags']
            state = np.array(df['Flags'])
        elif 'ledstate' in np.char.lower(np.array(cols)):
            arr = ['FrameCounter', 'LedState']
            state = np.array(df['LedState'])
        else:
            insertLog("File type shows Neurophotometrics newer version \
                    data but column names does not have Flags or LedState",
                    logging.ERROR)
            raise Exception("File type shows Neurophotometrics newer version \
                            data but column names does not have Flags or LedState")

        num_ch, ch = check_channels(state)
        indices_dict = dict()
        for i in range(num_ch):
            first_occurrence = np.where(state==ch[i])[0]
            indices_dict[ch_name[i]] = np.arange(first_occurrence[0], df.shape[0], num_ch)
        
        df = df.drop(arr, axis=1)

    return df, indices_dict, num_ch

def separate_last_element(arr):
    l = arr[-1]
    return arr[:-1], l

def access_keys_doricV6(doric_file):
    data = [doric_file["DataAcquisition"]]
    res = []
    while len(data) != 0:
        members = len(data)
        while members != 0:
            members -= 1
            data, last_element = separate_last_element(data)
            if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                res.append(last_element.name)
            elif isinstance(last_element, h5py.Group):
                data.extend(reversed([last_element[k] for k in last_element.keys()])) 

    keys = []
    for element in res:
        sep_values = element.split('/')
        if sep_values[-1]=='Values':
            keys.append(sep_values[-2])
        else:
            keys.append(sep_values[-1])
    
    return keys

def access_keys_doricV1(doric_file):
    keys = list(doric_file['Traces']['Console'].keys())
    keys.remove('Time(s)')
    
    return keys

def read_doric(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'Traces' in list(f.keys()):
            keys = access_keys_doricV1(f)
        elif list(f.keys())==['Configurations', 'DataAcquisition']:
            keys = access_keys_doricV6(f)
    
    return keys

# function to see if there are 'csv' files present
# and recognize type of 'csv' files either from
# Neurophotometrics, Doric systems or custom made 'csv' files
# and read data accordingly
def import_np_doric_csv(filepath, isosbestic_control, num_ch):

    insertLog("If it exists, importing either NPM or Doric or csv file based on the structure of file",
              logging.DEBUG)
    path = sorted(glob.glob(os.path.join(filepath, '*.csv'))) + \
           sorted(glob.glob(os.path.join(filepath, '*.doric')))
    path_chev = glob.glob(os.path.join(filepath, '*chev*'))
    path_chod = glob.glob(os.path.join(filepath, '*chod*'))
    path_chpr = glob.glob(os.path.join(filepath, '*chpr*'))
    path_event = glob.glob(os.path.join(filepath, 'event*'))
    #path_sig = glob.glob(os.path.join(filepath, 'sig*'))
    path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

    path = list(set(path)-set(path_chev_chod_event))
    flag = 'None'
    event_from_filename = []
    flag_arr = []
    for i in range(len(path)):
        dirname = os.path.dirname(path[i])
        ext = os.path.basename(path[i]).split('.')[-1]
        if ext=='doric':
            key_names = read_doric(path[i])
            event_from_filename.extend(key_names)
            flag = 'doric_doric'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    df = pd.read_csv(path[i], index_col=False, dtype=float)
                except:
                    df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)   # to make process faster reading just first 10 rows
                    df = df.drop(['Time(s)'], axis=1)
                    event_from_filename.extend(list(df.columns))
                    flag = 'doric_csv'
        if flag=='doric_csv' or flag=='doric_doric':
            continue
        else:
            colnames, value = check_header(df)
            #print(len(colnames), len(value))

            # check dataframe structure and read data accordingly
            if len(value)>0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                cols = np.array(list(df.columns), dtype=str)
            else:
                df = df
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(cols)==1:
                if cols[0].lower()!='timestamps':
                    insertLog("\033[1m"+"Column name should be timestamps (all lower-cases)"+"\033[0m",
                              logging.ERROR)
                    raise Exception("\033[1m"+"Column name should be timestamps (all lower-cases)"+"\033[0m")
                else:
                    flag = 'event_csv'
            elif len(cols)==3:
                arr1 = np.array(['timestamps', 'data', 'sampling_rate'])
                arr2 = np.char.lower(np.array(cols))
                if (np.sort(arr1)==np.sort(arr2)).all()==False:
                    insertLog("\033[1m"+"Column names should be timestamps, data and sampling_rate (all lower-cases)"+"\033[0m",
                              logging.ERROR)
                    raise Exception("\033[1m"+"Column names should be timestamps, data and sampling_rate (all lower-cases)"+"\033[0m")
                else:
                    flag = 'data_csv'
            elif len(cols)==2:
                flag = 'event_or_data_np'
            elif len(cols)>=2:
                flag  = 'data_np'
            else:
                insertLog('Number of columns in csv file does not make sense.',
                          logging.ERROR)
                raise Exception('Number of columns in csv file does not make sense.')


            if columns_isstr == True and ('flags' in np.char.lower(np.array(cols)) or 'ledstate' in np.char.lower(np.array(cols))):
                flag = flag+'_v2'
            else:
                flag = flag

            print(flag)
            # used assigned flags to process the files and read the data
            if flag=='event_or_data_np':
                arr = list(df.iloc[:,1])
                check_float = [True for i in arr if type(i)==np.float]
                if len(arr)==len(check_float) and columns_isstr == False:
                    flag = 'data_np'
                elif columns_isstr == True and ('value' in np.char.lower(np.array(cols))):
                    flag = 'event_np'
                else:
                    flag = 'event_np'
            else:
                pass
            
            flag_arr.append(flag)
            if flag=='event_csv' or flag=='data_csv':
                name = os.path.basename(path[i]).split('.')[0]
                event_from_filename.append(name)
            elif flag=='data_np':
                file = f'file{str(i)}_'
                df, indices_dict, num_channels = decide_indices(file, df, flag, num_ch)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j==0:
                            timestamps = df.iloc[:,j][indices_dict[keys[k]]]
                            #timestamps_odd = df.iloc[:,j][odd_indices]
                        else:
                            d = dict()
                            d['timestamps'] = timestamps 
                            d['data'] = df.iloc[:,j][indices_dict[keys[k]]]
                            
                            df_ch = pd.DataFrame(d)
                            df_ch.to_csv(os.path.join(dirname, keys[k]+str(j)+'.csv'), index=False)
                            event_from_filename.append(keys[k]+str(j))
                        
            elif flag=='event_np':
                type_val = np.array(df.iloc[:,1])
                type_val_unique = np.unique(type_val)
                if len(type_val_unique)>1:
                    response = messagebox.askyesno('Multiple event TTLs', 'Based on the TTL file,\
                                                                            it looks like TTLs \
                                                                            belongs to multipe behavior type. \
                                                                            Do you want to create multiple files for each \
                                                                            behavior type ?')
                else:
                    response = 0
                if response==1:
                    timestamps = np.array(df.iloc[:,0])
                    for j in range(len(type_val_unique)):
                        idx = np.where(type_val==type_val_unique[j])
                        d = dict()
                        d['timestamps'] = timestamps[idx]
                        df_new = pd.DataFrame(d)
                        df_new.to_csv(os.path.join(dirname, 'event'+str(j)+'.csv'), index=False)
                        event_from_filename.append('event'+str(j))
                else:
                    timestamps = np.array(df.iloc[:,0])
                    d = dict()
                    d['timestamps'] = timestamps
                    df_new = pd.DataFrame(d)
                    df_new.to_csv(os.path.join(dirname, 'event'+str(0)+'.csv'), index=False)
                    event_from_filename.append('event'+str(0))
            else:
                file = f'file{str(i)}_'
                df, indices_dict, num_channels = decide_indices(file, df, flag)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j==0:
                            timestamps = df.iloc[:,j][indices_dict[keys[k]]]
                            #timestamps_odd = df.iloc[:,j][odd_indices]
                        else:
                            d = dict()
                            d['timestamps'] = timestamps
                            d['data'] = df.iloc[:,j][indices_dict[keys[k]]]
                            
                            df_ch = pd.DataFrame(d)
                            df_ch.to_csv(os.path.join(dirname, keys[k]+str(j)+'.csv'), index=False)
                            event_from_filename.append(keys[k]+str(j))

            path_chev = glob.glob(os.path.join(filepath, '*chev*'))
            path_chod = glob.glob(os.path.join(filepath, '*chod*'))
            path_chpr = glob.glob(os.path.join(filepath, '*chpr*'))
            path_event = glob.glob(os.path.join(filepath, 'event*'))
            #path_sig = glob.glob(os.path.join(filepath, 'sig*'))
            path_chev_chod_chpr = [path_chev, path_chod, path_chpr]
            if ('data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag): # i==len(path)-1 and 
                num_path_chev, num_path_chod, num_path_chpr = len(path_chev), len(path_chod), len(path_chpr)
                arr_len, no_ch = [], []
                for i in range(len(path_chev_chod_chpr)):
                    if len(path_chev_chod_chpr[i])>0:
                        arr_len.append(len(path_chev_chod_chpr[i]))
                    else:
                        continue

                unique_arr_len = np.unique(np.array(arr_len))
                if 'data_np_v2' in flag_arr:
                    divisor = 1
                else:
                    divisor = 1000

                for j in range(len(path_event)):
                    df_event = pd.read_csv(path_event[j])
                    df_chev = pd.read_csv(path_chev[0])
                    df_event['timestamps'] = (df_event['timestamps']-df_chev['timestamps'][0])/divisor
                    df_event.to_csv(path_event[j], index=False)
                if unique_arr_len.shape[0]==1:
                    for j in range(len(path_chev)):
                        if file+'chev' in indices_dict.keys():
                            df_chev = pd.read_csv(path_chev[j])
                            df_chev['timestamps'] = (df_chev['timestamps']-df_chev['timestamps'][0])/divisor
                            df_chev['sampling_rate'] = np.full(df_chev.shape[0], np.nan)
                            df_chev['sampling_rate'][0] = df_chev.shape[0]/(df_chev['timestamps'].iloc[-1] - df_chev['timestamps'].iloc[0])
                            df_chev.to_csv(path_chev[j], index=False)

                        if file+'chod' in indices_dict.keys():
                            df_chod = pd.read_csv(path_chod[j])
                            df_chod['timestamps'] = df_chev['timestamps']
                            df_chod['sampling_rate'] = np.full(df_chod.shape[0], np.nan)
                            df_chod['sampling_rate'][0] = df_chev['sampling_rate'][0]
                            df_chod.to_csv(path_chod[j], index=False)

                        if file+'chpr' in indices_dict.keys():
                            df_chpr = pd.read_csv(path_chpr[j])
                            df_chpr['timestamps'] = df_chev['timestamps']
                            df_chpr['sampling_rate'] = np.full(df_chpr.shape[0], np.nan)
                            df_chpr['sampling_rate'][0] = df_chev['sampling_rate'][0]
                            df_chpr.to_csv(path_chpr[j], index=False)
                else:
                    insertLog('Number of channels should be same for all regions.',
                              logging.ERROR)
                    raise Exception('Number of channels should be same for all regions.')
            else:
                pass
    insertLog('Importing of either NPM or Doric or csv file is done.',
              logging.INFO)
    return event_from_filename, flag_arr


# function to read input parameters and run the saveStorenames function
def execute(inputParameters):
    
    
    inputParameters = inputParameters
    folderNames = inputParameters['folderNames']
    isosbestic_control = inputParameters['isosbestic_control']
    num_ch = inputParameters['noChannels']

    print(folderNames)

    try:
        for i in folderNames:
            filepath = os.path.join(inputParameters['abspath'], i)
            data = readtsq(filepath)
            event_name, flag = import_np_doric_csv(filepath, isosbestic_control, num_ch)
            saveStorenames(inputParameters, data, event_name, flag, filepath)
        insertLog('#'*400, logging.INFO)
    except Exception as e:
        insertLog(str(e), logging.ERROR)
        raise e





