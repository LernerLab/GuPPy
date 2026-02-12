"""
Temporary script to manually verify compare_output_folders.

Run from the repo root:
    python check_consistency.py
"""

from guppy.testing import compare_output_folders

ACTUAL = "testing_data/SampleData_Clean/Photo_63_207-181030-103332" "/Photo_63_207-181030-103332_output_1"
EXPECTED = "testing_data/StandardOutputs_Clean/Photo_63_207-181030-103332" "/Photo_63_207-181030-103332_output_1"

try:
    compare_output_folders(actual_dir=ACTUAL, expected_dir=EXPECTED)
    print("PASS: all expected files are present and numerically identical")
except AssertionError as exc:
    print("DIFFERENCES FOUND:\n")
    print(exc)
