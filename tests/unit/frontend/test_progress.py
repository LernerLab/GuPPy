import threading
import time

import pytest

from guppy.frontend import progress as progress_module
from guppy.frontend.progress import (
    readPBIncrementValues,
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


@pytest.mark.progress_bar
def test_readpbincrementvalues_returns_error_message_on_failure(tmp_path):
    """When the subprocess writes -1 and an error file exists, the error message is returned."""
    steps_file = tmp_path / "pbSteps.txt"
    error_file = tmp_path / "pbError.txt"

    expected_message = "Cross-correlation requires at least two distinct signal recording sites"
    error_file.write_text(expected_message)

    def _write_failure():
        time.sleep(0.05)
        steps_file.write_text("-1\n")

    t = threading.Thread(target=_write_failure)
    t.start()

    pb = _MockProgressBar()
    result = readPBIncrementValues(pb, file_path=str(steps_file), error_file_path=str(error_file))
    t.join(timeout=2)

    assert not t.is_alive(), "Writer thread did not finish in time"
    assert result == expected_message
    assert pb.bar_color == "danger"
    assert not error_file.exists()


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


@pytest.mark.progress_bar
def test_readpbincrementvalues_returns_none_when_no_error_file(tmp_path):
    """When the subprocess writes -1 but no error file exists, None is returned."""
    steps_file = tmp_path / "pbSteps.txt"
    error_file = tmp_path / "pbError.txt"  # intentionally absent

    def _write_failure():
        time.sleep(0.05)
        steps_file.write_text("-1\n")

    t = threading.Thread(target=_write_failure)
    t.start()

    pb = _MockProgressBar()
    result = readPBIncrementValues(pb, file_path=str(steps_file), error_file_path=str(error_file))
    t.join(timeout=2)

    assert not t.is_alive(), "Writer thread did not finish in time"
    assert result is None
    assert pb.bar_color == "danger"
