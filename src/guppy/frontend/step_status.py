"""The status line under each sidebar step."""

from datetime import datetime

import panel as pn

_TEXT_STYLE = "font-size: 13px;"
_NOT_RUN_COLOR = "#8a96a3"
_RUNNING_COLOR = "#55616f"
_DONE_COLOR = "#1e7e34"
_FAILED_COLOR = "#c0392b"


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
        self.progress = pn.indicators.Progress(value=0, max=100, width=300, visible=False) if has_progress else None
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
