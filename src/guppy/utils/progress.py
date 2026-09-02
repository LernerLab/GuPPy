"""Progress reporting channel for pipeline steps.

Modelled on :mod:`logging`: deep pipeline code emits progress by calling the module-level
functions here, with no reporter threaded through its signature, and the top level decides
where those emissions go. Emitting when nothing is bound is a no-op, so the workers can be
driven headlessly (``guppy.testing.api``) without side effects.

The sink is a :class:`StepProgress` bound to a :class:`~contextvars.ContextVar` by
``guppy.orchestration.home`` for the duration of one step run. Only that module binds; only
the pipeline steps emit.
"""

import functools
import logging
import threading
from contextvars import ContextVar
from typing import Callable

logger = logging.getLogger(__name__)


class StepProgress:
    """Mutable progress state for a single pipeline-step run.

    Written by the step running on a background thread and read by the Panel poller on the
    server IOLoop. ``completed`` is guarded by a lock because steps advance it from pool
    callbacks as well as the step thread; ``total`` and ``error_message`` are single
    assignments of immutable values, which are atomic under the GIL.

    ``warnings`` collects messages about work the step declined to do but that did not stop
    it, so the poller can surface them once the step finishes.
    """

    def __init__(self) -> None:
        self.total = 0
        self.error_message = None
        self.warnings = []
        self._completed = 0
        self._source = None
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        """Units of work completed so far.

        Reads from the tracked source when one is attached, so that progress driven by a
        shared counter in a worker pool needs no thread to copy it across.
        """
        if self._source is not None:
            return int(self._source())
        return self._completed

    def start(self, total: int) -> None:
        """Set the denominator and reset progress to zero."""
        self.total = total
        with self._lock:
            self._completed = 0

    def advance(self, count: int = 1) -> None:
        """Add ``count`` completed units."""
        with self._lock:
            self._completed += count

    def track(self, source: Callable[[], int]) -> None:
        """Take the completed count from ``source`` instead of from ``advance`` calls."""
        self._source = source

    def fail(self, message: str) -> None:
        """Record the step's failure message."""
        self.error_message = message

    def warn(self, message: str) -> None:
        """Record a message about work the step skipped without failing."""
        with self._lock:
            self.warnings.append(message)


_current_step: ContextVar[StepProgress | None] = ContextVar("guppy_step_progress", default=None)


def start(total: int) -> None:
    """Declare how many units of work the running step will complete."""
    step = _current_step.get()
    if step is not None:
        step.start(total)


def advance(count: int = 1) -> None:
    """Report ``count`` further units of work completed."""
    step = _current_step.get()
    if step is not None:
        step.advance(count)


def track(source: Callable[[], int]) -> None:
    """Report progress by pulling from ``source`` rather than by calling :func:`advance`."""
    step = _current_step.get()
    if step is not None:
        step.track(source)


def fail(message: str) -> None:
    """Report that the running step failed with ``message``."""
    step = _current_step.get()
    if step is not None:
        step.fail(message)


def warn(message: str) -> None:
    """Report that the running step skipped something the user should know about."""
    step = _current_step.get()
    if step is not None:
        step.warn(message)


def step_error_handler(func: object) -> object:
    """Decorate a pipeline step's top-level function with shared error reporting.

    The step runs on a background thread, where an uncaught exception would otherwise be
    swallowed by the threading excepthook. On success, logs the banner separator. On
    exception, reports the message through the progress channel so the poller can surface it
    to the user, then re-raises.
    """

    @functools.wraps(func)
    def wrapper(input_parameters: dict[str, object]) -> object:
        try:
            result = func(input_parameters)
            logger.info("#" * 400)
            return result
        except Exception as e:
            fail(str(e))
            logger.error(str(e))
            raise

    return wrapper
