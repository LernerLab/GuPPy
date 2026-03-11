from pathlib import Path

from guppy.extractors import TdtRecordingExtractor

_REPO_ROOT = Path(__file__).parent


def main():
    source_folder_path = _REPO_ROOT / "testing_data" / "SampleData_Clean" / "Photo_63_207-181030-103332"
    stub_folder_path = _REPO_ROOT / "stubbed_testing_data" / "tdt" / "Photo_63_207-181030-103332"
    TdtRecordingExtractor(folder_path=source_folder_path).stub(folder_path=stub_folder_path, duration_in_seconds=60.0)


if __name__ == "__main__":
    main()
