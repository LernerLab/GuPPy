import os

import h5py
import numpy as np
import pytest

from guppy.orchestration.read_raw_data import orchestrate_read_raw_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INPUT_PARAMETERS = {
    "folderNames": [],  # overridden per test
    "numberOfCores": 1,
    "noChannels": 2,
}


def _make_hdf5(path):
    """Create a minimal HDF5 file at *path*."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("timestamps", data=np.zeros(10))


def _mock_extractor_map(storesList):
    """Return a dict mapping each event in storesList[0] to a sentinel object."""
    return {name: object() for name in storesList[0, :]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_with_cache(tmp_path):
    """
    Session output directory whose storesList has changed since the last run.

    .cache_storesList.csv  — OLD names (Dv2A / Di1i  →  signal_DMS / event1)
    storesList.csv         — NEW names (Dv3A / Di1i  →  signal_NAc / event1)

    HDF5 files exist for all OLD names; only Dv2A and signal_DMS are orphans.
    """
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    output_dir = session_dir / "session1_output_1"
    output_dir.mkdir()

    old_storesList = np.array([["Dv2A", "Di1i"], ["signal_DMS", "event1"]])
    new_storesList = np.array([["Dv3A", "Di1i"], ["signal_NAc", "event1"]])

    np.savetxt(str(output_dir / ".cache_storesList.csv"), old_storesList, delimiter=",", fmt="%s")
    np.savetxt(str(output_dir / "storesList.csv"), new_storesList, delimiter=",", fmt="%s")

    # Orphan files (old names absent from new storesList)
    _make_hdf5(output_dir / "Dv2A.hdf5")
    _make_hdf5(output_dir / "signal_DMS.hdf5")

    # Shared files (names that appear in both old and new storesList)
    _make_hdf5(output_dir / "Di1i.hdf5")
    _make_hdf5(output_dir / "event1.hdf5")

    return session_dir, output_dir


# ---------------------------------------------------------------------------
# Tests — cache-driven orphan cleanup
# ---------------------------------------------------------------------------


def test_orphan_hdf5_deleted_for_names_absent_from_new_storeslist(session_with_cache, monkeypatch):
    """HDF5 files for names in .cache_storesList but absent from storesList.csv are deleted."""
    session_dir, output_dir = session_with_cache

    monkeypatch.setattr("guppy.orchestration.read_raw_data.read_and_save_all_events", lambda *a, **kw: None)
    monkeypatch.setattr(
        "guppy.orchestration.read_raw_data._build_event_to_extractor",
        lambda **kw: _mock_extractor_map(kw["storesList"]),
    )
    monkeypatch.setattr("guppy.orchestration.read_raw_data.writeToFile", lambda *a, **kw: None)

    params = {**_INPUT_PARAMETERS, "folderNames": [str(session_dir)]}
    orchestrate_read_raw_data(params)

    assert not (output_dir / "Dv2A.hdf5").exists(), "Orphan Dv2A.hdf5 should have been deleted"
    assert not (output_dir / "signal_DMS.hdf5").exists(), "Orphan signal_DMS.hdf5 should have been deleted"


def test_shared_hdf5_files_preserved_during_orphan_cleanup(session_with_cache, monkeypatch):
    """HDF5 files for names present in both old and new storesList are not deleted."""
    session_dir, output_dir = session_with_cache

    monkeypatch.setattr("guppy.orchestration.read_raw_data.read_and_save_all_events", lambda *a, **kw: None)
    monkeypatch.setattr(
        "guppy.orchestration.read_raw_data._build_event_to_extractor",
        lambda **kw: _mock_extractor_map(kw["storesList"]),
    )
    monkeypatch.setattr("guppy.orchestration.read_raw_data.writeToFile", lambda *a, **kw: None)

    params = {**_INPUT_PARAMETERS, "folderNames": [str(session_dir)]}
    orchestrate_read_raw_data(params)

    assert (output_dir / "Di1i.hdf5").exists(), "Di1i.hdf5 (shared name) must not be deleted"
    assert (output_dir / "event1.hdf5").exists(), "event1.hdf5 (shared name) must not be deleted"


def test_no_orphan_cleanup_without_cache(tmp_path, monkeypatch):
    """When .cache_storesList.csv is absent, no HDF5 files are deleted."""
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    output_dir = session_dir / "session1_output_1"
    output_dir.mkdir()

    storesList = np.array([["Dv2A", "Di1i"], ["signal_DMS", "event1"]])
    np.savetxt(str(output_dir / "storesList.csv"), storesList, delimiter=",", fmt="%s")

    # Pre-existing file that is NOT in the current storesList
    _make_hdf5(output_dir / "OldName.hdf5")

    monkeypatch.setattr("guppy.orchestration.read_raw_data.read_and_save_all_events", lambda *a, **kw: None)
    monkeypatch.setattr(
        "guppy.orchestration.read_raw_data._build_event_to_extractor",
        lambda **kw: _mock_extractor_map(kw["storesList"]),
    )
    monkeypatch.setattr("guppy.orchestration.read_raw_data.writeToFile", lambda *a, **kw: None)

    params = {**_INPUT_PARAMETERS, "folderNames": [str(session_dir)]}
    orchestrate_read_raw_data(params)

    assert (output_dir / "OldName.hdf5").exists(), "Without a cache file, OldName.hdf5 must not be deleted"


def test_orphan_deletion_logged(session_with_cache, monkeypatch, caplog):
    """Each deleted orphan HDF5 file should produce an INFO log entry."""
    import logging

    session_dir, output_dir = session_with_cache

    monkeypatch.setattr("guppy.orchestration.read_raw_data.read_and_save_all_events", lambda *a, **kw: None)
    monkeypatch.setattr(
        "guppy.orchestration.read_raw_data._build_event_to_extractor",
        lambda **kw: _mock_extractor_map(kw["storesList"]),
    )
    monkeypatch.setattr("guppy.orchestration.read_raw_data.writeToFile", lambda *a, **kw: None)

    params = {**_INPUT_PARAMETERS, "folderNames": [str(session_dir)]}
    with caplog.at_level(logging.INFO, logger="guppy.orchestration.read_raw_data"):
        orchestrate_read_raw_data(params)

    deleted_log_messages = [r.message for r in caplog.records if "Deleted orphan HDF5 file" in r.message]
    assert len(deleted_log_messages) == 2, f"Expected 2 deletion log entries, got: {deleted_log_messages}"
