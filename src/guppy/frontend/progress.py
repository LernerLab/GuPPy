import functools
import logging
import os

logger = logging.getLogger(__name__)

PB_STEPS_FILE = os.path.join(os.path.expanduser("~"), "pbSteps.txt")
PB_ERROR_FILE = os.path.join(os.path.expanduser("~"), "pbError.txt")


def subprocess_main_handler(func: object) -> object:
    """Decorate an orchestration subprocess entry point with shared error reporting.

    On success, logs the banner separator. On exception, writes the error message
    to ``PB_ERROR_FILE`` and a ``-1`` sentinel to ``PB_STEPS_FILE`` so that
    ``readPBIncrementValues`` (running in the parent Panel process) can stop
    polling and surface the message to the user, then re-raises.
    """

    @functools.wraps(func)
    def wrapper(input_parameters: dict[str, object]) -> object:
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


def read_progress_snapshot(
    *, file_path: str, error_file_path: str = PB_ERROR_FILE
) -> tuple[int | None, int | None, str | None, bool]:
    """Read the shared progress file once, without blocking.

    Parameters
    ----------
    file_path : str
        Path to the progress-steps file written by the worker subprocess.
    error_file_path : str
        Path to the file holding the subprocess failure message.

    Returns
    -------
    maximum : int or None
        The progress-bar maximum (first line), or None when nothing is readable yet.
    increment : int or None
        The latest progress value (last line), or None when nothing is readable yet.
    error_message : str or None
        The subprocess failure message when ``increment`` is ``-1``, else None. The
        error file is consumed (deleted) once read.
    done : bool
        True when the subprocess reported completion (``increment == maximum``) or
        failure (``increment == -1``).
    """
    if not os.path.exists(file_path):
        return None, None, None, False
    # The worker subprocess writes this file concurrently, so a partial read can
    # yield an empty or half-written line; treat those as "nothing new this poll".
    try:
        with open(file_path, "r") as file:
            content = file.readlines()
    except (FileNotFoundError, PermissionError):
        return None, None, None, False
    if len(content) == 0:
        return None, None, None, False
    try:
        maximum = int(content[0])
        increment = int(content[-1])
    except ValueError:
        return None, None, None, False

    if increment == -1:
        error_message = None
        if os.path.exists(error_file_path):
            with open(error_file_path, "r") as error_file:
                error_message = error_file.read().strip()
            os.remove(error_file_path)
        return maximum, increment, error_message, True

    return maximum, increment, None, increment == maximum


def poll_progress_step(
    progressBar: object, *, file_path: str = PB_STEPS_FILE, error_file_path: str = PB_ERROR_FILE
) -> tuple[bool, str | None]:
    """Apply one progress-file read to the progress bar and report completion.

    Non-blocking: intended to be driven repeatedly by a Panel periodic callback on
    the server IOLoop so the main app never freezes while a step (or an interactive
    page it serves) runs.

    Parameters
    ----------
    progressBar : object
        Progress indicator with ``value``, ``max``, and ``bar_color`` attributes.
    file_path : str
        Path to the progress-steps file written by the worker subprocess.
    error_file_path : str
        Path to the file holding the subprocess failure message.

    Returns
    -------
    done : bool
        True when the subprocess reported completion or failure.
    error_message : str or None
        The subprocess failure message, or None on success / still-running.
    """
    maximum, increment, error_message, done = read_progress_snapshot(
        file_path=file_path, error_file_path=error_file_path
    )
    if increment == -1:
        progressBar.bar_color = "danger"
    elif maximum is not None:
        progressBar.max = maximum
        progressBar.value = increment
    return done, error_message


def writeToFile(value: str, *, file_path: str) -> None:
    """Append ``value`` to the file at ``file_path``, creating it if necessary.

    Parameters
    ----------
    value : str
        Text to append.
    file_path : str
        Absolute path to the target file.
    """
    with open(file_path, "a") as file:
        file.write(value)
