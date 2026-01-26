import logging
import os
import tkinter as tk
from tkinter import filedialog, ttk

logger = logging.getLogger(__name__)


def get_folder_path():
    # Determine base folder path (headless-friendly via env var)
    base_dir_env = os.environ.get("GUPPY_BASE_DIR")
    is_headless = base_dir_env and os.path.isdir(base_dir_env)
    if is_headless:
        folder_path = base_dir_env
        logger.info(f"Folder path set to {folder_path} (from GUPPY_BASE_DIR)")
        return folder_path

    # Create the main window
    folder_selection = tk.Tk()
    folder_selection.title("Select the folder path where your data is located")
    folder_selection.geometry("700x200")

    selected_path = {"value": None}

    def select_folder():
        selected = filedialog.askdirectory(title="Select the folder path where your data is located")
        if selected:
            logger.info(f"Folder path set to {selected}")
            selected_path["value"] = selected
        else:
            default_path = os.path.expanduser("~")
            logger.info(f"Folder path set to {default_path}")
            selected_path["value"] = default_path
        folder_selection.destroy()

    select_button = ttk.Button(folder_selection, text="Select a Folder", command=select_folder)
    select_button.pack(pady=5)
    folder_selection.mainloop()

    return selected_path["value"]
