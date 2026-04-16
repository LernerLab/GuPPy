import logging
import os
import time

logger = logging.getLogger(__name__)

PB_STEPS_FILE = os.path.join(os.path.expanduser("~"), "pbSteps.txt")
PB_ERROR_FILE = os.path.join(os.path.expanduser("~"), "pbError.txt")


def readPBIncrementValues(progressBar, *, file_path, error_file_path=PB_ERROR_FILE):  # pragma: no cover
    """Read progress bar values from file and update the progress bar widget.

    Returns the error message string if the subprocess reported a failure,
    or ``None`` on success.

    Note: excluded from coverage because this function uses a tight polling loop
    that reads from a file written by a subprocess; the threading and file-lock
    behaviour cannot be tested reliably on Windows in CI (see issue #286).
    """
    logger.info("Read progress bar increment values function started...")
    if os.path.exists(file_path):
        os.remove(file_path)
    # Always remove stale error file at the start, regardless of whether an error pane is provided.
    if os.path.exists(error_file_path):
        os.remove(error_file_path)
    increment, maximum = 0, 100
    progressBar.value = increment
    progressBar.bar_color = "success"
    error_message = None
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
                        if os.path.exists(error_file_path):
                            with open(error_file_path, "r") as ef:
                                error_message = ef.read().strip()
                            os.remove(error_file_path)
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
    return error_message


def writeToFile(value: str, *, file_path):
    with open(file_path, "a") as file:
        file.write(value)
