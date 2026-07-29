import pytest

from guppy.utils import progress
from guppy.utils.progress import StepProgress, _current_step, step_error_handler


@pytest.fixture
def bound_step():
    """Bind a StepProgress for the duration of one test, as ``home.py`` does per step run."""
    step = StepProgress()
    token = _current_step.set(step)
    yield step
    _current_step.reset(token)


class TestStepProgress:
    def test_starts_empty(self):
        step = StepProgress()
        assert (step.total, step.value, step.error_message) == (0, 0, None)

    def test_start_sets_total_and_zeroes_value(self):
        step = StepProgress()
        step.advance(7)
        step.start(20)
        assert (step.total, step.value) == (20, 0)

    def test_advance_accumulates(self):
        step = StepProgress()
        step.start(10)
        step.advance()
        step.advance(4)
        assert step.value == 5

    def test_track_pulls_from_source_instead_of_advance(self):
        step = StepProgress()
        step.start(100)
        counter = {"samples": 0}
        step.track(lambda: counter["samples"])

        assert step.value == 0
        counter["samples"] = 64
        assert step.value == 64

        # A tracked step ignores pushed units; the source is the single truth.
        step.advance(5)
        assert step.value == 64

    def test_fail_records_message(self):
        step = StepProgress()
        step.fail("baselineWindowStart=-1 is before the signal start 0s")
        assert step.error_message == "baselineWindowStart=-1 is before the signal start 0s"


class TestModuleFunctions:
    def test_emit_to_the_bound_step(self, bound_step):
        progress.start(6)
        progress.advance()
        progress.advance(2)
        progress.fail("boom")

        assert (bound_step.total, bound_step.value, bound_step.error_message) == (6, 3, "boom")

    def test_track_reaches_the_bound_step(self, bound_step):
        progress.start(50)
        counter = {"value": 12}
        progress.track(lambda: counter["value"])

        assert bound_step.value == 12

    def test_emitting_with_nothing_bound_is_a_no_op(self):
        """Headless callers (``guppy.testing.api``) run the workers with no sink bound."""
        assert _current_step.get() is None

        progress.start(10)
        progress.advance()
        progress.track(lambda: 3)
        progress.fail("ignored")

        assert _current_step.get() is None


class TestStepErrorHandler:
    def test_returns_value_and_reports_no_failure_on_success(self, bound_step):
        @step_error_handler
        def worker(input_parameters):
            return input_parameters["x"] + 1

        assert worker({"x": 41}) == 42
        assert bound_step.error_message is None

    def test_reports_failure_and_reraises_on_exception(self, bound_step):
        @step_error_handler
        def worker(input_parameters):
            raise ValueError("bad parameter foo=3; valid range is [0, 1]")

        with pytest.raises(ValueError, match="bad parameter foo=3"):
            worker({})

        assert bound_step.error_message == "bad parameter foo=3; valid range is [0, 1]"

    def test_reraises_when_no_step_is_bound(self):
        @step_error_handler
        def worker(input_parameters):
            raise ValueError("still raised")

        with pytest.raises(ValueError, match="still raised"):
            worker({})
