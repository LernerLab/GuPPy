import logging
import os
import time

logger = logging.getLogger(__name__)

PB_STEPS_FILE = os.path.join(os.path.expanduser("~"), "pbSteps.txt")
PB_ERROR_FILE = os.path.join(os.path.expanduser("~"), "guppyError.txt")


def writeErrorToFile(message: str, *, file_path=PB_ERROR_FILE):
    """Write an error message to a file for the GUI to read and display."""
    with open(file_path, "w") as file:
        file.write(message)


def readPBIncrementValues(progressBar, *, file_path, error_pane=None, error_file_path=PB_ERROR_FILE):
    logger.info("Read progress bar increment values function started...")
    if os.path.exists(file_path):
        os.remove(file_path)
    if error_pane is not None and os.path.exists(error_file_path):
        os.remove(error_file_path)
    increment, maximum = 0, 100
    progressBar.value = increment
    progressBar.bar_color = "success"
    if error_pane is not None:
        error_pane.object = ""
    while True:
        try:
            with open(file_path, "r") as file:
                content = file.readlines()
                if len(content) == 0:
                    pass
                else:
                    maximum = int(content[0])
                    increment = int(content[-1])

                    if increment == -1:
                        progressBar.bar_color = "danger"
                        if error_pane is not None:
                            try:
                                with open(error_file_path, "r") as ef:
                                    msg = ef.read().strip()
                                if msg:
                                    error_pane.object = f"**Error:** {msg}"
                            except FileNotFoundError:
                                pass
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
            logger.info(f"An error occurred while reading the file: {e}")
            break
        if increment == maximum:
            os.remove(file_path)
            break

    logger.info("Read progress bar increment values stopped.")


def writeToFile(value: str, *, file_path):
    with open(file_path, "a") as file:
        file.write(value)
