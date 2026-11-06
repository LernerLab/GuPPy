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


class TestProgressGradient:
    def test_an_empty_bar_starts_at_the_violet_end_of_the_logo_gradient(self, status):
        (stylesheet,) = status.progress.stylesheets

        assert "linear-gradient(90deg, #6E56CF, #1F7FC4, #14A39A, #5DB45C)" in stylesheet
        assert "background-size: 10000.00% 100%" in stylesheet

    def test_the_gradient_spans_the_track_as_the_bar_fills(self, status):
        # At 60%, the filled part is 60% of the track, so the gradient is drawn 100 / 60 times as wide.
        status.progress.value = 60

        (stylesheet,) = status.progress.stylesheets
        assert "background-size: 166.67% 100%" in stylesheet

    def test_a_full_bar_shows_the_whole_gradient(self, status):
        status.progress.value = 100

        (stylesheet,) = status.progress.stylesheets
        assert "background-size: 100.00% 100%" in stylesheet
