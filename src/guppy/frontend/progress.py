import logging
import os
import time

logger = logging.getLogger(__name__)

PB_STEPS_FILE = os.path.join(os.path.expanduser("~"), "pbSteps.txt")


def readPBIncrementValues(progressBar, *, file_path):
    logger.info("Read progress bar increment values function started...")
    if os.path.exists(file_path):
        os.remove(file_path)
    increment, maximum = 0, 100
    progressBar.value = increment
    progressBar.bar_color = "success"
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
