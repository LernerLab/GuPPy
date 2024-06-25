import tkinter as tk
from tkinter import ttk, StringVar, messagebox


def comboBoxSelected(event):
    print(event.widget.get())

window = tk.Tk()
window.title('Select appropriate options for timestamps')
window.geometry('500x200')
holdComboboxValues = dict()

timestamps_label = ttk.Label(window, 
                             text="Select which timetamps to use : ").grid(row=0, column=1, pady=25, padx=25)
holdComboboxValues['timestamps'] = StringVar()
timestamps_combo = ttk.Combobox(window, 
                                values=['', 'SystemTimestamp', "ComputerTimestamp"], 
                                textvariable=holdComboboxValues['timestamps'])
#timestamps_combo.pack()
#timestamps_combo.current()
timestamps_combo.grid(row=0, column=2, pady=25, padx=25)
timestamps_combo.current(0)
timestamps_combo.bind("<<ComboboxSelected>>", comboBoxSelected)

time_unit_label = ttk.Label(window, text="Select timetamps unit : ").grid(row=1, column=1, pady=25, padx=25)
holdComboboxValues['time_unit'] = StringVar()
time_unit_combo = ttk.Combobox(window, 
                               values=['', 'seconds', 'milliseconds', 'microseconds'],
                               textvariable=holdComboboxValues['time_unit'])
time_unit_combo.grid(row=1, column=2, pady=25, padx=25)
time_unit_combo.current(0)
time_unit_combo.bind("<<ComboboxSelected>>", comboBoxSelected)
window.mainloop()

print(holdComboboxValues)
print(holdComboboxValues['timestamps'].get())
print(holdComboboxValues['time_unit'].get())

if holdComboboxValues['timestamps'].get():
    print('yes')
else:
    print('no')
if holdComboboxValues['time_unit'].get():
    print('yes')
else:
    print('no')
