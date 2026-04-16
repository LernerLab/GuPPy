import threading
import time

import pytest

from guppy.frontend.progress import readPBIncrementValues, writeToFile


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

    expected_message = "Cross-correlation requires at least two distinct signal regions"
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
