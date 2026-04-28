import functools
import logging
import os
import time

logger = logging.getLogger(__name__)

PB_STEPS_FILE = os.path.join(os.path.expanduser("~"), "pbSteps.txt")
PB_ERROR_FILE = os.path.join(os.path.expanduser("~"), "pbError.txt")


def subprocess_main_handler(func):
    """Decorate an orchestration subprocess entry point with shared error reporting.

    On success, logs the banner separator. On exception, writes the error message
    to ``PB_ERROR_FILE`` and a ``-1`` sentinel to ``PB_STEPS_FILE`` so that
    ``readPBIncrementValues`` (running in the parent Panel process) can stop
    polling and surface the message to the user, then re-raises.
    """

    @functools.wraps(func)
    def wrapper(input_parameters):
        try:
            result = func(input_parameters)
            logger.info("#" * 400)
            return result
        except Exception as e:
            with open(PB_ERROR_FILE, "w") as error_file:
                error_file.write(str(e))
            writeToFile(str(-1) + "\n", file_path=PB_STEPS_FILE)
            logger.error(str(e))
            raise

    return wrapper


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
