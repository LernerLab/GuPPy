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
    else:
        path = path[0]
    tsq = np.fromfile(path, dtype=tsq_dtype)
    df = pd.DataFrame(tsq)
    return df


# In[3]:


# function to show GUI and save 
def saveStorenames(inputParametersPath, data, event_name, filepath):
    
    # reading input parameters file
    with open(inputParametersPath) as f:
        inputParameters = json.load(f)


    # reading storenames from the data fetched using 'readtsq' function
    data['name'] = np.asarray(data['name'], dtype=np.str)
    allnames = np.unique(data['name'])
    index = []
    for i in range(len(allnames)):
        length = len(np.str(allnames[i]))
        if length<4:
            index.append(i)
    allnames = np.delete(allnames, index, 0)

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

                - Select “create new” or “overwrite” to generate a new storenames list or replace a previous one
                - Click Save

                """, width=550)

    # finalizing all the storenames
    allnames = list(allnames)
    allnames = allnames + event_name

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

    text = pn.widgets.TextInput(value="[]", name='Selected Store Names')

    path = pn.widgets.TextInput(name='Location to Stores List file', width=500, sizing_mode="stretch_width")

    mark_down_for_overwrite = pn.pane.Markdown(""" Select option from below if user wants to over-write a file or create a new file. 
                                    **Creating a new file will make a new ouput folder and will get saved at that location.**
                                    If user selects to over-write a file **Select location of the file to over-write** will provide 
                                    the existing options of the ouput folders where user needs to over-write the file""")

    select_location = pn.widgets.Select(name='Select location of the file to over-write', value='None', options=['None'], align='end')


    overwrite_button = pn.widgets.MenuButton(name='over-write storeslist file or create a new one?  ', items=['over_write_file', 'create_new_file'], button_type='default', split=True, align='end')
    
    literal_input_2 = pn.widgets.Ace(value="""{}""", sizing_mode='stretch_both', theme='tomorrow', language='json', height=150)
    take_widgets = pn.WidgetBox(
        multi_choice,
        literal_input_1   
    )

    change_widgets = pn.WidgetBox(
        text
    )

    
    storenames = []
    

    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event):
        if event.new=='over_write_file':
            select_location.options = glob.glob(os.path.join(filepath, '*_output_*'))
            #select_location.value = select_location.options[0]
        else:
            select_location.options = [show_dir(filepath)]
            #select_location.value = select_location.options[0]

    # on clicking 'Select Storenames' button, following function is executed
    def update_values(event):
        global storenames
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
            w.value = str(storenames)

        d = dict()
        d["storenames"] = storenames
        d["names_for_storenames"] = []
        literal_input_2.value = str(json.dumps(d))




    # on clicking save button, following functino is executed
    def save_button(event=None):
        global storenames
        
        d = json.loads(literal_input_2.value)
        arr1, arr2 = np.asarray(d["storenames"]), np.asarray(d["names_for_storenames"])

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

    widget_1 = pn.Column('# '+os.path.basename(filepath), mark_down)

    widget_2 = pn.Column(repeat_storename_wd, cross_selector, update_options, text, literal_input_2, mark_down_for_overwrite, overwrite_button, select_location, save, path)
    #pn.Column(mark_down_for_overwrite, pn.Row(overwrite_button, select_location))
    #box = pn.Column('# '+os.path.basename(filepath), mark_down, multi_choice, literal_input_1, cross_selector, update_options, text, literal_input_2, save) # '# '+os.path.basename(filepath), 
    template.main.append(pn.Row(widget_1, widget_2))
    #box = pn.Row(widget_1, widget_2)
    template.show(port=number)


# In[4]:
# function to import event name to make it appear in the storenames selection 
# (if event timestamps csv file is present at the same location where TDT data is present)
def import_csv(filepath):
    path = glob.glob(os.path.join(filepath, '*.csv'))

    event_from_filename = []
    for i in range(len(path)):
        name = os.path.basename(path[i]).split('.')[0]
        event_from_filename.append(name)

    return event_from_filename


# function to read input parameters and run the saveStorenames function
def execute(inputParametersPath):
    inputParametersPath = inputParametersPath
    with open(inputParametersPath) as f:
        inputParameters = json.load(f)

    folderNames = inputParameters['folderNames']

    print(folderNames)
    # In[5]:

    for i in folderNames:
        filepath = os.path.join(inputParameters['abspath'], i)
        data = readtsq(filepath)
        event_name = import_csv(filepath)
        saveStorenames(inputParametersPath, data, event_name, filepath)


#execute(sys.argv[1:][0])



# In[ ]:




