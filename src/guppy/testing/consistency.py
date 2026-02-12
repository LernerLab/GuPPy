"""
Consistency utilities for comparing GuPPy output folders across versions.

The comparison is one-directional: every file present in ``expected_dir`` must
exist in ``actual_dir`` and be numerically identical. Extra files in
``actual_dir`` that are absent from ``expected_dir`` are silently ignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# Subdirectories to skip entirely (plots are not numerically reproducible).
_SKIP_DIRS = {"saved_plots"}


def compare_output_folders(*, actual_dir: str, expected_dir: str) -> None:
    """
    Assert that every file in ``expected_dir`` exists in ``actual_dir`` and is
    numerically identical.

    Extra files present in ``actual_dir`` but absent from ``expected_dir`` are
    silently ignored — they represent intentional additions and are not failures.

    All mismatches are collected before raising so that a single run surfaces
    every discrepancy rather than stopping at the first one.

    Parameters
    ----------
    actual_dir : str
        Path to the output folder produced by the current code under test.
    expected_dir : str
        Path to the reference output folder (e.g. from GuPPy v1.3.0).

    Raises
    ------
    AssertionError
        If any expected file is missing from ``actual_dir`` or if the content
        of a shared file does not match. The error message lists every
        discrepancy found.
    """
    actual_dir = os.path.abspath(actual_dir)
    expected_dir = os.path.abspath(expected_dir)

    expected_files = _collect_relative_paths(expected_dir)

    mismatches: list[str] = []

    for rel_path in sorted(expected_files):
        actual_path = os.path.join(actual_dir, rel_path)
        expected_path = os.path.join(expected_dir, rel_path)

        if not os.path.exists(actual_path):
            mismatches.append(f"MISSING in actual: {rel_path}")
            continue

        ext = Path(rel_path).suffix.lower()
        if ext in {".hdf5", ".h5"}:
            _compare_hdf5(actual_path, expected_path, rel_path, mismatches)
        elif ext == ".csv":
            _compare_csv(actual_path, expected_path, rel_path, mismatches)
        elif ext == ".json":
            _compare_json(actual_path, expected_path, rel_path, mismatches)
        # Unknown extensions are skipped silently.

    if mismatches:
        summary = "\n".join(mismatches)
        raise AssertionError(f"Output folder comparison failed:\n{summary}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_relative_paths(root: str) -> list[str]:
    """Return all file paths under *root* as paths relative to *root*, skipping _SKIP_DIRS."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk does not descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            result.append(os.path.relpath(full, root))
    return result


def _compare_hdf5(
    actual_path: str,
    expected_path: str,
    rel_path: str,
    mismatches: list[str],
) -> None:
    """Compare all datasets in two HDF5 files, accumulating mismatches."""
    with h5py.File(actual_path, "r") as actual_f, h5py.File(expected_path, "r") as expected_f:
        _walk_hdf5_group(actual_f, expected_f, rel_path, "", mismatches)


def _walk_hdf5_group(
    actual_group: h5py.Group,
    expected_group: h5py.Group,
    rel_path: str,
    group_path: str,
    mismatches: list[str],
) -> None:
    """Recursively walk HDF5 groups, comparing all datasets."""
    for key in expected_group.keys():
        item_path = f"{group_path}/{key}" if group_path else key
        if key not in actual_group:
            mismatches.append(f"{rel_path}: missing HDF5 key '{item_path}' in actual")
            continue

        expected_item = expected_group[key]
        actual_item = actual_group[key]

        if isinstance(expected_item, h5py.Group):
            if not isinstance(actual_item, h5py.Group):
                mismatches.append(f"{rel_path}: '{item_path}' is a group in expected but a dataset in actual")
            else:
                _walk_hdf5_group(actual_item, expected_item, rel_path, item_path, mismatches)
        elif isinstance(expected_item, h5py.Dataset):
            if not isinstance(actual_item, h5py.Dataset):
                mismatches.append(f"{rel_path}: '{item_path}' is a dataset in expected but a group in actual")
            else:
                _compare_hdf5_dataset(actual_item, expected_item, rel_path, item_path, mismatches)


def _compare_hdf5_dataset(
    actual_ds: h5py.Dataset,
    expected_ds: h5py.Dataset,
    rel_path: str,
    item_path: str,
    mismatches: list[str],
) -> None:
    """Compare two HDF5 datasets, handling numeric and string dtypes."""
    actual_data = actual_ds[()]
    expected_data = expected_ds[()]

    # Scalar datasets (0-d) are returned as Python scalars by h5py.
    actual_data = np.asarray(actual_data)
    expected_data = np.asarray(expected_data)

    if actual_data.shape != expected_data.shape:
        mismatches.append(
            f"{rel_path}: '{item_path}' shape mismatch: " f"actual={actual_data.shape} expected={expected_data.shape}"
        )
        return

    # String / bytes datasets: exact equality.
    if expected_data.dtype.kind in {"S", "U", "O"}:
        if not np.array_equal(actual_data, expected_data):
            mismatches.append(f"{rel_path}: '{item_path}' string data differs")
        return

    # Numeric datasets: tolerance-based comparison with NaN == NaN.
    if not np.allclose(actual_data, expected_data, rtol=1e-5, atol=1e-8, equal_nan=True):
        mismatches.append(f"{rel_path}: '{item_path}' numeric data differs")


def _compare_csv(
    actual_path: str,
    expected_path: str,
    rel_path: str,
    mismatches: list[str],
) -> None:
    """Compare two CSV files as DataFrames."""
    actual_df = pd.read_csv(actual_path, index_col=0)
    expected_df = pd.read_csv(expected_path, index_col=0)
    try:
        pd.testing.assert_frame_equal(actual_df, expected_df, check_exact=False, rtol=1e-5, atol=1e-8, check_names=True)
    except AssertionError as exc:
        mismatches.append(f"{rel_path}: CSV content differs — {exc}")


def _compare_json(
    actual_path: str,
    expected_path: str,
    rel_path: str,
    mismatches: list[str],
) -> None:
    """Compare two JSON files with NaN-safe value comparison."""
    with open(actual_path) as f:
        actual_data = json.load(f)
    with open(expected_path) as f:
        expected_data = json.load(f)

    json_mismatches: list[str] = []
    _compare_json_values(actual_data, expected_data, rel_path, "", json_mismatches)
    mismatches.extend(json_mismatches)


def _compare_json_values(
    actual: Any,
    expected: Any,
    rel_path: str,
    key_path: str,
    mismatches: list[str],
) -> None:
    """Recursively compare JSON values with NaN-aware scalar comparison."""
    location = f"{rel_path} @ '{key_path}'" if key_path else rel_path

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(f"{location}: expected dict, got {type(actual).__name__}")
            return
        for k in expected:
            if k not in actual:
                mismatches.append(f"{location}: missing key '{k}' in actual")
            else:
                _compare_json_values(actual[k], expected[k], rel_path, f"{key_path}.{k}" if key_path else k, mismatches)

    elif isinstance(expected, list):
        if not isinstance(actual, list):
            mismatches.append(f"{location}: expected list, got {type(actual).__name__}")
            return
        if len(actual) != len(expected):
            mismatches.append(f"{location}: list length mismatch: actual={len(actual)} expected={len(expected)}")
            return
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare_json_values(a, e, rel_path, f"{key_path}[{i}]", mismatches)

    else:
        # Scalar: handle NaN
        try:
            actual_is_nan = actual is None or (isinstance(actual, float) and np.isnan(actual))
            expected_is_nan = expected is None or (isinstance(expected, float) and np.isnan(expected))
        except (TypeError, ValueError):
            actual_is_nan = False
            expected_is_nan = False

        if expected_is_nan and not actual_is_nan:
            mismatches.append(f"{location}: expected NaN/None, got {actual!r}")
        elif not expected_is_nan and actual_is_nan:
            mismatches.append(f"{location}: expected {expected!r}, got NaN/None")
        elif not expected_is_nan and not actual_is_nan and actual != expected:
            mismatches.append(f"{location}: expected {expected!r}, got {actual!r}")
