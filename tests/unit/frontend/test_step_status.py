import re

import pytest

from guppy.frontend.step_status import StepStatus


@pytest.fixture
def status(panel_extension):
    return StepStatus(has_progress=True, has_results=True)


def message_text(status):
    return re.sub(r"<[^>]+>", "", status.message.object)


class TestStepStatus:
    def test_starts_as_not_run_yet(self, status):
        assert message_text(status) == "Not run yet"
        assert status.progress.visible is False
        assert status.results_button.visible is False

    def test_a_step_without_progress_or_results_has_neither(self, panel_extension):
        status = StepStatus(has_progress=False, has_results=False)

        assert status.progress is None
        assert status.results_button is None
        assert message_text(status) == "Not run yet"

    def test_running_shows_an_empty_progress_bar(self, status):
        status.progress.value = 7

        status.set_running()

        assert message_text(status) == "Running…"
        assert status.progress.visible is True
        assert status.progress.value == 0

    def test_done_hides_the_progress_bar_and_says_when(self, status):
        status.set_running()

        status.set_done()

        assert re.fullmatch(r"✓ Done at \d\d:\d\d", message_text(status))
        assert status.progress.visible is False

    def test_failed_hides_the_progress_bar_and_says_when(self, status):
        status.set_running()

        status.set_failed()

        assert re.fullmatch(r"✗ Failed at \d\d:\d\d", message_text(status))
        assert status.progress.visible is False

    def test_opened_says_when(self, panel_extension):
        status = StepStatus(has_progress=False, has_results=False)

        status.set_opened()

        assert re.fullmatch(r"✓ Opened at \d\d:\d\d", message_text(status))
