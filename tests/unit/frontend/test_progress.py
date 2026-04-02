from guppy.frontend.progress import writeToFile


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
