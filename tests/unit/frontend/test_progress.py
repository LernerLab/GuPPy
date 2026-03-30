import time
from threading import Thread

import panel as pn

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


def test_read_pb_increment_values_terminates_on_completion(tmp_path, panel_extension):
    """readPBIncrementValues terminates when increment equals maximum."""
    file_path = tmp_path / "pbSteps.txt"
    progress_bar = pn.indicators.Progress(value=0, max=100)

    def write_completion():
        time.sleep(0.01)
        file_path.write_text("3\n3\n")

    writer = Thread(target=write_completion)
    writer.start()
    readPBIncrementValues(progress_bar, file_path=file_path)
    writer.join()

    assert progress_bar.value == 3
    assert progress_bar.max == 3


def test_read_pb_increment_values_sets_danger_on_error_sentinel(tmp_path, panel_extension):
    """readPBIncrementValues sets bar_color to danger when sentinel -1 is written."""
    file_path = tmp_path / "pbSteps.txt"
    progress_bar = pn.indicators.Progress(value=0, max=100)

    def write_error_sentinel():
        time.sleep(0.01)
        file_path.write_text("5\n-1\n")

    writer = Thread(target=write_error_sentinel)
    writer.start()
    readPBIncrementValues(progress_bar, file_path=file_path)
    writer.join()

    assert progress_bar.bar_color == "danger"
