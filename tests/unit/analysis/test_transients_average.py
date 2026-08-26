import numpy as np
import pytest

from guppy.analysis.standard_io import (
    read_freq_and_amp_from_hdf5,
    write_freq_and_amp_to_hdf5,
)
from guppy.analysis.transients_average import average_transients_for_group

# ── average_transients_for_group ──────────────────────────────────────────────


@pytest.fixture
def group_folder(tmp_path):
    folder = tmp_path / "saline_group"
    folder.mkdir()
    return folder


def test_average_transients_for_group_stacks_the_members_freq_and_amp(tmp_path, group_folder):
    session1 = tmp_path / "session1"
    session2 = tmp_path / "session2"
    session1.mkdir()
    session2.mkdir()

    # Stub files so glob("z_score_*") finds them
    (session1 / "z_score_dms.hdf5").touch()
    (session2 / "z_score_dms.hdf5").touch()

    # Write per-session freq/amp HDF5 files
    # session1: freq=2.0, amp=1.5; session2: freq=3.0, amp=2.5
    write_freq_and_amp_to_hdf5(
        str(session1),
        np.array([[2.0, 1.5]]),
        "z_score_dms",
        index=["session1"],
        columns=["freq (events/min)", "amplitude"],
    )
    write_freq_and_amp_to_hdf5(
        str(session2),
        np.array([[3.0, 2.5]]),
        "z_score_dms",
        index=["session2"],
        columns=["freq (events/min)", "amplitude"],
    )

    input_parameters = {"selectForTransientsComputation": "z_score"}
    average_transients_for_group(
        member_run_folders=[str(session1), str(session2)],
        group_folder=str(group_folder),
        inputParameters=input_parameters,
    )

    assert (group_folder / "freqAndAmp_z_score_dms.h5").exists()
    assert (group_folder / "freqAndAmp_z_score_dms.csv").exists()
    # Nothing is written to an "average" directory beside the members any more.
    assert not (tmp_path / "average").exists()

    df = read_freq_and_amp_from_hdf5(str(group_folder), "z_score_dms")
    np.testing.assert_allclose(df["freq (events/min)"].values, np.array([2.0, 3.0]))
    np.testing.assert_allclose(df["amplitude"].values, np.array([1.5, 2.5]))


def test_average_transients_for_group_handles_non_overlapping_stores_without_indexerror(tmp_path, group_folder):
    """Non-overlapping store_ids across sessions must not cause an IndexError.

    Regression test for issue #274.
    """
    session1 = tmp_path / "session1"
    session2 = tmp_path / "session2"
    session1.mkdir()
    session2.mkdir()

    (session1 / "z_score_regionA.hdf5").touch()
    (session2 / "z_score_regionB.hdf5").touch()

    write_freq_and_amp_to_hdf5(
        str(session1),
        np.array([[2.0, 1.5]]),
        "z_score_regionA",
        index=["session1"],
        columns=["freq (events/min)", "amplitude"],
    )
    write_freq_and_amp_to_hdf5(
        str(session2),
        np.array([[3.0, 2.5]]),
        "z_score_regionB",
        index=["session2"],
        columns=["freq (events/min)", "amplitude"],
    )

    # Must not raise IndexError
    average_transients_for_group(
        member_run_folders=[str(session1), str(session2)],
        group_folder=str(group_folder),
        inputParameters={"selectForTransientsComputation": "z_score"},
    )

    assert (group_folder / "freqAndAmp_z_score_regionA.h5").exists()
    assert (group_folder / "freqAndAmp_z_score_regionB.h5").exists()
