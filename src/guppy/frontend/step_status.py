"""The status line under each sidebar step."""

from datetime import datetime

import panel as pn

from .header import LOGO_GRADIENT

_TEXT_STYLE = "font-size: 12px;"
_NOT_RUN_COLOR = "#6C757D"
_RUNNING_COLOR = "#6C757D"
_DONE_COLOR = "#1e7e34"
_FAILED_COLOR = "#c0392b"


def _progress_stylesheet(*, value: int, maximum: int) -> str:
    """Return the CSS that fills a progress bar with the logo's gradient.

    The gradient spans the whole track, so a bar part-way through shows only its violet end and
    turns green as it completes. The filled part is only ``value / maximum`` of the track wide, so
    the gradient is drawn that many times wider than it.

    Parameters
    ----------
    value : int
        The bar's progress.
    maximum : int
        The value at which the bar is full.

    Returns
    -------
    str
        A stylesheet for the ``Progress`` widget.
    """
    filled_percent = max(100 * value / maximum, 1)
    gradient_width = f"{100 * 100 / filled_percent:.2f}% 100%"
    return f"""
progress[value]::-webkit-progress-value {{ background-image: {LOGO_GRADIENT}; background-size: {gradient_width}; }}
progress[value]::-moz-progress-bar {{ background-image: {LOGO_GRADIENT}; background-size: {gradient_width}; }}
"""


class StepStatus:
    """A step's status under its sidebar button: not run yet, running, done, failed, or opened.

    Parameters
    ----------
    has_progress : bool
        Whether the step runs in the background, showing a progress bar while it runs.
    has_results : bool
        Whether the step has a results page, opened from a View results button once it is done.
    """

    def __init__(self, *, has_progress: bool, has_results: bool) -> None:
        self.message = pn.pane.HTML(margin=(4, 10))
        self.progress = None
        if has_progress:
            self.progress = pn.indicators.Progress(
                value=0, max=100, width=300, visible=False, stylesheets=[_progress_stylesheet(value=0, maximum=100)]
            )
            self.progress.param.watch(self._repaint_progress, ["value", "max"])
        self.results_button = (
            pn.widgets.Button(
                name="View results →",
                button_type="primary",
                button_style="outline",
                height=28,
                width=120,
                margin=(0, 10),
                align="center",
                visible=False,
            )
            if has_results
            else None
        )
        status_row = pn.Row(self.message, width=320)
        if self.results_button is not None:
            status_row.extend([pn.HSpacer(), self.results_button])
        self.layout = pn.Column(*([self.progress] if has_progress else []), status_row, margin=0)
        self._show(text="Not run yet", color=_NOT_RUN_COLOR)

    def _repaint_progress(self, *events: object) -> None:
        self.progress.stylesheets = [_progress_stylesheet(value=self.progress.value, maximum=self.progress.max)]

    def _show(self, *, text: str, color: str) -> None:
        self.message.object = f'<span style="{_TEXT_STYLE} color: {color};">{text}</span>'

    def _now(self) -> str:
        return datetime.now().astimezone().strftime("%H:%M")

    def set_running(self) -> None:
        """Show the progress bar, empty, and say the step is running."""
        self.progress.value = 0
        self.progress.visible = True
        self._show(text="Running…", color=_RUNNING_COLOR)

    def set_done(self) -> None:
        """Hide the progress bar and say when the step finished."""
        self.progress.visible = False
        self._show(text=f"✓ Done at {self._now()}", color=_DONE_COLOR)

    def set_failed(self) -> None:
        """Hide the progress bar and say when the step failed."""
        self.progress.visible = False
        self._show(text=f"✗ Failed at {self._now()}", color=_FAILED_COLOR)

    def set_opened(self) -> None:
        """Say when the step's page was last opened."""
        self._show(text=f"✓ Opened at {self._now()}", color=_DONE_COLOR)
