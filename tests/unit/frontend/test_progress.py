import panel as pn

from guppy.frontend.progress import readPBIncrementValues, writeToFile


def test_write_to_file_creates_file_with_content(tmp_path, monkeypatch):
    monkeypatch.setattr("guppy.frontend.progress.os.path.expanduser", lambda _: str(tmp_path))
    writeToFile("50\n")
    file_path = tmp_path / "pbSteps.txt"
    assert file_path.exists()
    assert file_path.read_text() == "50\n"


def test_write_to_file_appends_multiple_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("guppy.frontend.progress.os.path.expanduser", lambda _: str(tmp_path))
    writeToFile("100\n")
    writeToFile("200\n")
    file_path = tmp_path / "pbSteps.txt"
    content = file_path.read_text()
    assert "100\n" in content
    assert "200\n" in content


def test_read_pb_increment_values_terminates_on_completion(tmp_path, monkeypatch, panel_extension):
    """readPBIncrementValues terminates when increment equals maximum.

    Patch os.path.exists to return False so the startup delete is skipped,
    allowing the pre-written terminal progress file to be found on the first
    polling iteration.
    """
    monkeypatch.setattr("guppy.frontend.progress.os.path.expanduser", lambda _: str(tmp_path))
    monkeypatch.setattr("guppy.frontend.progress.os.path.exists", lambda _: False)

    file_path = tmp_path / "pbSteps.txt"
    file_path.write_text("3\n3\n")

    progress_bar = pn.indicators.Progress(value=0, max=100)
    readPBIncrementValues(progress_bar)

    assert progress_bar.value == 3
    assert progress_bar.max == 3


def test_read_pb_increment_values_sets_danger_on_error_sentinel(tmp_path, monkeypatch, panel_extension):
    """readPBIncrementValues sets bar_color to danger when sentinel -1 is written."""
    monkeypatch.setattr("guppy.frontend.progress.os.path.expanduser", lambda _: str(tmp_path))
    monkeypatch.setattr("guppy.frontend.progress.os.path.exists", lambda _: False)

    file_path = tmp_path / "pbSteps.txt"
    file_path.write_text("5\n-1\n")

    progress_bar = pn.indicators.Progress(value=0, max=100)
    readPBIncrementValues(progress_bar)

    assert progress_bar.bar_color == "danger"
