#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from numpy import int32, uint32, uint8, uint16, float64, int64, int32, float32
import panel as pn
from collections import OrderedDict
from random import randint
from pathlib import Path
import holoviews as hv
from holoviews import opts
import tkinter as tk
from tkinter import ttk, StringVar

#hv.extension()
pn.extension()
# In[2]:


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
        raise Exception('Two tsq files are present at the location.')
    elif len(path)==0:
        return 0
    else:
        path = path[0]
    tsq = np.fromfile(path, dtype=tsq_dtype)
    df = pd.DataFrame(tsq)
    return df


# In[3]:


# function to show GUI and save 
def saveStorenames(inputParametersPath, data, event_name, flag, filepath):
    
    # reading input parameters file
    with open(inputParametersPath) as f:
        inputParameters = json.load(f)

    # reading storenames from the data fetched using 'readtsq' function
    if isinstance(data, pd.DataFrame):
        data['name'] = np.asarray(data['name'], dtype=np.str)
        allnames = np.unique(data['name'])
        index = []
        for i in range(len(allnames)):
            length = len(np.str(allnames[i]))
            if length<4:
                index.append(i)
        allnames = np.delete(allnames, index, 0)
        allnames = list(allnames)

    else:
        allnames = []


    if 'data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag:
        path_chev = glob.glob(os.path.join(filepath, 'chev*'))
        path_chod = glob.glob(os.path.join(filepath, 'chod*'))
        path_chpr = glob.glob(os.path.join(filepath, 'chpr*'))
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

                - See instructions in “Github Wiki” for duplicating storenames. Otherwise leave these selections blank.<br>
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
    multi_choice = pn.widgets.MultiChoice(name='Select Storenames which you want more than once', value=[], options=allnames)

    literal_input_1 = pn.widgets.LiteralInput(name='Number of times you want the above storename', value=[], type=list)
    #literal_input_2 = pn.widgets.LiteralInput(name='Names for Storenames (list)', type=list)

    repeat_storename_wd = pn.WidgetBox('Storenames to repeat (leave blank if not needed)', multi_choice, literal_input_1, width=600)

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

    alert = pn.pane.Alert('#### No alerts !!', alert_type='danger', height=60)


    take_widgets = pn.WidgetBox(
        multi_choice,
        literal_input_1   
    )

    change_widgets = pn.WidgetBox(
        text
    )

    
    #storenames = []
    
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
                names_for_storenames.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
            elif comboBoxValues[i]=='event TTLs':
                names_for_storenames.append(textBoxValues[i])
            else:
                names_for_storenames.append(comboBoxValues[i])

        d = dict()
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
                                    value=['control', 'signal', 'event TTLs'], 
                                    width=12)
                    textBox = tk.Entry(root, 
                                    textvariable=hold_textBoxValues[storenames[i]+'_'+str(i)])
                else:
                    hold_comboBoxValues[storenames[i]] = StringVar()
                    hold_textBoxValues[storenames[i]] = StringVar()
                    myCombo = ttk.Combobox(root, 
                                        textvariable=hold_comboBoxValues[storenames[i]],
                                        value=['control', 'signal', 'event TTLs'], 
                                        width=12)
                    textBox = tk.Entry(root, 
                                    textvariable=hold_textBoxValues[storenames[i]])
                myCombo.grid(row=i+1, column=2)
                textBox.grid(row=i+1, column=4)
                myCombo.current(0)
                myCombo.bind("<<ComboboxSelected>>", comboBoxSelected)

        button = ttk.Button(root, text='Show', command=fetchValues).grid(row=len(storenames)*2, column=2)   
        root.mainloop()
    
    # on clicking save button, following function is executed
    def save_button(event=None):
        global storenames
        
        d = json.loads(literal_input_2.value)
        arr1, arr2 = np.asarray(d["storenames"]), np.asarray(d["names_for_storenames"])

        if np.where(arr2=="")[0].size>0:
            alert.object = '#### Alert !! \n Empty string in the list names_for_storenames.'
            raise Exception('Empty string in the list names_for_storenames.')
        else:
            alert.object = '#### No alerts !!'

        if arr1.shape[0]!=arr2.shape[0]:
            alert.object = '#### Alert !! \n Length of list storenames and names_for_storenames is not equal.'
            raise Exception('Length of list storenames and names_for_storenames is not equal.')
        else:
            alert.object = '#### No alerts !!'


        if not os.path.exists(os.path.join(Path.home(), '.storesList.json')):
            global_storenames = dict()

            for i in range(arr1.shape[0]):
                if arr1[i] in global_storenames:
                    global_storenames[arr1[i]].append(arr2[i])
                    global_storenames[arr1[i]] = list(set(global_storenames[arr1[i]]))
                else:
                    global_storenames[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), '.storesList.json'), 'w') as f:
                json.dump(global_storenames, f, indent=4) 
        else:
            with open(os.path.join(Path.home(), '.storesList.json')) as f:
                global_storenames = json.load(f)

            for i in range(arr1.shape[0]):
                if arr1[i] in global_storenames:
                    global_storenames[arr1[i]].append(arr2[i])
                    global_storenames[arr1[i]] = list(set(global_storenames[arr1[i]]))
                else:
                    global_storenames[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), '.storesList.json'), 'w') as f:
                json.dump(global_storenames, f, indent=4)

        arr = np.asarray([arr1, arr2])
        print(arr)
        if not os.path.exists(select_location.value):
            os.mkdir(select_location.value)
            
        np.savetxt(os.path.join(select_location.value, 'storesList.csv'), arr, delimiter=",", fmt='%s')
        path.value = os.path.join(select_location.value, 'storesList.csv')

    

    update_options.on_click(update_values)
    save.on_click(save_button)
    overwrite_button.on_click(overwrite_button_actions)

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = randint(5000,5200)

    if 'data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag:
        widget_1 = pn.Column('# '+os.path.basename(filepath), mark_down, mark_down_np, plot_select, plot)
        widget_2 = pn.Column(repeat_storename_wd, cross_selector, update_options, 
                             text, literal_input_2, alert, mark_down_for_overwrite, 
                             overwrite_button, select_location, save, path)
        template.main.append(pn.Row(widget_1, widget_2))

    else:
        widget_1 = pn.Column('# '+os.path.basename(filepath), mark_down)
        widget_2 = pn.Column(repeat_storename_wd, cross_selector, update_options, 
                             text, literal_input_2, alert, mark_down_for_overwrite, 
                             overwrite_button, select_location, save, path)
        template.main.append(pn.Row(widget_1, widget_2))

    template.show(port=number)


# check flag consistency in neurophotometrics data
def check_channels(state):
    state = state.astype(int)
    unique_state = np.unique(state[2:12])
    if unique_state.shape[0]>3:
        raise Exception("Looks like there are more than 3 channels in the file. Reading of these files\
                        are not supported. Reach out to us if you get this error message.")

    return unique_state.shape[0], unique_state
    

# function to decide indices of interleaved channels
# in neurophotometrics data
def decide_indices(df, flag, num_ch=2):
    ch_name = ['chev', 'chod', 'chpr']
    if len(ch_name)<num_ch:
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
            raise Exception("File type shows Neurophotometrics newer version \
                            data but column names does not have Flags or LedState")

        num_ch, ch = check_channels(state)
        indices_dict = dict()
        for i in range(num_ch):
            first_occurrence = np.where(state==ch[i])[0]
            indices_dict[ch_name[i]] = np.arange(first_occurrence[0], df.shape[0], num_ch)
        
        df = df.drop(arr, axis=1)

    return df, indices_dict, num_ch

# In[4]:
# function to see if there are 'csv' files present
# and recognize type of 'csv' files either from
# Neurophotometrics or custom made 'csv' files
# and read data accordingly
def import_np_csv(filepath, isosbestic_control, num_ch):
    path = sorted(glob.glob(os.path.join(filepath, '*.csv')))
    path_chev = glob.glob(os.path.join(filepath, 'chev*'))
    path_chod = glob.glob(os.path.join(filepath, 'chod*'))
    path_chpr = glob.glob(os.path.join(filepath, 'chpr*'))
    path_event = glob.glob(os.path.join(filepath, 'event*'))
    #path_sig = glob.glob(os.path.join(filepath, 'sig*'))
    path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

    path = list(set(path)-set(path_chev_chod_event))
    event_from_filename = []
    flag_arr = []
    for i in range(len(path)):
        dirname = os.path.dirname(path[i])

        df = pd.read_csv(path[i], dtype=float)

        colnames, value = check_header(df)
        #print(len(colnames), len(value))

        # check dataframe structure and read data accordingly
        if len(value)>0:
            columns_isstr = False
            df = pd.read_csv(path[i], header=None)
            cols = np.array(list(df.columns), dtype=np.str)
        else:
            df = df
            columns_isstr = True
            cols = np.array(list(df.columns), dtype=np.str)

        # check the structure of dataframe and assign flag to the type of file
        if len(cols)==1:
            if cols[0].lower()!='timestamps':
                raise Exception("\033[1m"+"Column name should be timestamps (all lower-cases)"+"\033[0m")
            else:
                flag = 'event_csv'
        elif len(cols)==3:
            arr1 = np.array(['timestamps', 'data', 'sampling_rate'])
            arr2 = np.char.lower(np.array(cols))
            if (np.sort(arr1)==np.sort(arr2)).all()==False:
                raise Exception("\033[1m"+"Column names should be timestamps, data and sampling_rate (all lower-cases)"+"\033[0m")
            else:
                flag = 'data_csv'
        elif len(cols)==2:
            flag = 'event_or_data_np'
        elif len(cols)>=2:
            flag  = 'data_np'
        else:
            raise Exception('Number of columns in csv file does not make sense.')


        if columns_isstr == True and ('flags' in np.char.lower(np.array(cols)) or 'ledstate' in np.char.lower(np.array(cols))):
            flag = flag+'_v2'
        else:
            flag = flag


        # used assigned flags to process the files and read the data
        if flag=='event_or_data_np':
            arr = list(df.iloc[:,1])
            check_float = [True for i in arr if type(i)==np.float]
            if len(arr)==len(check_float):
                flag = 'data_np'
            else:
                flag = 'event_np'
        else:
            pass

        flag_arr.append(flag)
        print(flag)
        if flag=='event_csv' or flag=='data_csv':
            name = os.path.basename(path[i]).split('.')[0]
            event_from_filename.append(name)
        elif flag=='data_np':
            df, indices_dict, num_channels = decide_indices(df, flag, num_ch)
            
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
            timestamps = np.array(df.iloc[:,0])
            for j in range(len(type_val_unique)):
                idx = np.where(type_val==type_val_unique[j])
                d = dict()
                d['timestamps'] = timestamps[idx]
                df_new = pd.DataFrame(d)
                df_new.to_csv(os.path.join(dirname, 'event'+str(j)+'.csv'), index=False)
                event_from_filename.append('event'+str(j))
        else:
            df, indices_dict, num_channels = decide_indices(df, flag)

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

        path_chev = glob.glob(os.path.join(filepath, 'chev*'))
        path_chod = glob.glob(os.path.join(filepath, 'chod*'))
        path_chpr = glob.glob(os.path.join(filepath, 'chpr*'))
        path_event = glob.glob(os.path.join(filepath, 'event*'))
        #path_sig = glob.glob(os.path.join(filepath, 'sig*'))
        path_chev_chod_chpr = [path_chev, path_chod, path_chpr]

        if i==len(path)-1 and ('data_np_v2' in flag or 'data_np' in flag or 'event_np' in flag):
            num_path_chev, num_path_chod, num_path_chpr = len(path_chev), len(path_chod), len(path_chpr)
            arr_len, no_ch = [], []
            for i in range(len(path_chev_chod_chpr)):
                if len(path_chev_chod_chpr[i])>0:
                    arr_len.append(len(path_chev_chod_chpr[i]))
                else:
                    continue

            unique_arr_len = np.unique(np.array(arr_len))
            print(unique_arr_len)
            print(unique_arr_len.shape[0])
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
                    if 'chev' in indices_dict.keys():
                        df_chev = pd.read_csv(path_chev[j])
                        df_chev['timestamps'] = (df_chev['timestamps']-df_chev['timestamps'][0])/divisor
                        df_chev['sampling_rate'] = np.full(df_chev.shape[0], np.nan)
                        df_chev['sampling_rate'][0] = df_chev.shape[0]/(df_chev['timestamps'].iloc[-1] - df_chev['timestamps'].iloc[0])
                        df_chev.to_csv(path_chev[j], index=False)

                    if 'chod' in indices_dict.keys():
                        df_chod = pd.read_csv(path_chod[j])
                        df_chod['timestamps'] = df_chev['timestamps']
                        df_chod['sampling_rate'] = np.full(df_chod.shape[0], np.nan)
                        df_chod['sampling_rate'][0] = df_chev['sampling_rate'][0]
                        df_chod.to_csv(path_chod[j], index=False)

                    if 'chpr' in indices_dict.keys():
                        df_chpr = pd.read_csv(path_chpr[j])
                        df_chpr['timestamps'] = df_chev['timestamps']
                        df_chpr['sampling_rate'] = np.full(df_chpr.shape[0], np.nan)
                        df_chpr['sampling_rate'][0] = df_chev['sampling_rate'][0]
                        df_chpr.to_csv(path_chpr[j], index=False)
            else:
                raise Exception('Number of channels should be same for all regions.')
        else:
            pass
    
    return event_from_filename, flag_arr


# function to read input parameters and run the saveStorenames function
def execute(inputParametersPath):
    inputParametersPath = inputParametersPath
    with open(inputParametersPath) as f:
        inputParameters = json.load(f)

    folderNames = inputParameters['folderNames']
    isosbestic_control = inputParameters['isosbestic_control']
    num_ch = inputParameters['noChannels']

    print(folderNames)
    # In[5]:

    for i in folderNames:
        filepath = os.path.join(inputParameters['abspath'], i)
        data = readtsq(filepath)
        event_name, flag = import_np_csv(filepath, isosbestic_control, num_ch)
        saveStorenames(inputParametersPath, data, event_name, flag, filepath)


#execute(sys.argv[1:][0])



# In[ ]:




