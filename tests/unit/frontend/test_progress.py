import pytest

from guppy.frontend import progress as progress_module
from guppy.frontend.progress import (
    poll_progress_step,
    read_progress_snapshot,
    subprocess_main_handler,
    writeToFile,
)


def test_write_to_file_creates_file_with_content(tmp_path):
    file_path = tmp_path / "pbSteps.txt"
    writeToFile("50\n", file_path=file_path)
    assert file_path.exists()
    assert file_path.read_text() == "50\n"


def test_write_to_file_appends_multiple_calls(tmp_path):
    file_path = tmp_path / "pbSteps.txt"
    writeToFile("100\n", file_path=file_path)
    writeToFile("200\n", file_path=file_path)
    content = file_path.read_text()
    assert "100\n" in content
    assert "200\n" in content


class _MockProgressBar:
    def __init__(self):
        self.value = 0
        self.max = 100
        self.bar_color = "success"


class TestReadProgressSnapshot:
    def test_missing_file_reports_nothing(self, tmp_path):
        assert read_progress_snapshot(file_path=str(tmp_path / "absent.txt")) == (None, None, None, False)

    def test_in_progress(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        steps_file.write_text("40\n10\n20\n")
        # max=40, latest=20, not done.
        assert read_progress_snapshot(file_path=str(steps_file)) == (40, 20, None, False)

    def test_done_when_increment_reaches_max(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        steps_file.write_text("40\n40\n")
        assert read_progress_snapshot(file_path=str(steps_file)) == (40, 40, None, True)

    def test_failure_returns_error_message_and_consumes_error_file(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        error_file = tmp_path / "pbError.txt"
        steps_file.write_text("40\n-1\n")
        error_file.write_text("Cross-correlation requires two recording sites")

        maximum, increment, error_message, done = read_progress_snapshot(
            file_path=str(steps_file), error_file_path=str(error_file)
        )
        assert (maximum, increment, done) == (40, -1, True)
        assert error_message == "Cross-correlation requires two recording sites"
        assert not error_file.exists()

    def test_failure_without_error_file_returns_none_message(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        steps_file.write_text("40\n-1\n")
        _, increment, error_message, done = read_progress_snapshot(
            file_path=str(steps_file), error_file_path=str(tmp_path / "absent.txt")
        )
        assert increment == -1 and done is True and error_message is None


class TestPollProgressStep:
    def test_updates_bar_while_running(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        steps_file.write_text("50\n30\n")
        bar = _MockProgressBar()
        done, error_message = poll_progress_step(bar, file_path=str(steps_file))
        assert (bar.max, bar.value) == (50, 30)
        assert done is False and error_message is None

    def test_reports_done_at_max(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        steps_file.write_text("50\n50\n")
        bar = _MockProgressBar()
        done, error_message = poll_progress_step(bar, file_path=str(steps_file))
        assert bar.value == 50 and done is True and error_message is None

    def test_failure_sets_danger_and_returns_message(self, tmp_path):
        steps_file = tmp_path / "pbSteps.txt"
        error_file = tmp_path / "pbError.txt"
        steps_file.write_text("50\n-1\n")
        error_file.write_text("Step failed in subprocess")
        bar = _MockProgressBar()
        done, error_message = poll_progress_step(bar, file_path=str(steps_file), error_file_path=str(error_file))
        assert bar.bar_color == "danger"
        assert done is True and error_message == "Step failed in subprocess"


class TestSubprocessMainHandler:
    @pytest.fixture
    def redirect_progress_files(self, tmp_path, monkeypatch):
        steps_file = tmp_path / "pbSteps.txt"
        error_file = tmp_path / "pbError.txt"
        monkeypatch.setattr(progress_module, "PB_STEPS_FILE", str(steps_file))
        monkeypatch.setattr(progress_module, "PB_ERROR_FILE", str(error_file))
        return steps_file, error_file

    def test_returns_value_and_writes_no_files_on_success(self, redirect_progress_files):
        steps_file, error_file = redirect_progress_files

        @subprocess_main_handler
        def worker(input_parameters):
            return input_parameters["x"] + 1

        result = worker({"x": 41})

        assert result == 42
        assert not steps_file.exists()
        assert not error_file.exists()

    def test_writes_error_files_and_reraises_on_exception(self, redirect_progress_files):
        steps_file, error_file = redirect_progress_files

        @subprocess_main_handler
        def worker(input_parameters):
            raise ValueError("bad parameter foo=3; valid range is [0, 1]")

        with pytest.raises(ValueError, match="bad parameter foo=3"):
            worker({})

        assert error_file.read_text() == "bad parameter foo=3; valid range is [0, 1]"
        assert steps_file.read_text() == "-1\n"
