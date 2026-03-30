import numpy as np

from guppy.analysis.standard_io import (
    read_freq_and_amp_from_hdf5,
    write_freq_and_amp_to_hdf5,
)
from guppy.analysis.transients_average import averageForGroup

# ── averageForGroup ───────────────────────────────────────────────────────────


def test_average_for_group_creates_combined_freq_amp_file(tmp_path):
    # Two session folders each with z_score_dms.hdf5 stub and freqAndAmp_z_score_dms.h5
    # Expected output: tmp_path/average/freqAndAmp_z_score_dms.h5
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

    input_parameters = {
        "abspath": str(tmp_path),
        "selectForTransientsComputation": "z_score",
    }
    averageForGroup([str(session1), str(session2)], input_parameters)

    assert (tmp_path / "average" / "freqAndAmp_z_score_dms.h5").exists()
    assert (tmp_path / "average" / "freqAndAmp_z_score_dms.csv").exists()

    df = read_freq_and_amp_from_hdf5(str(tmp_path / "average"), "z_score_dms")
    np.testing.assert_allclose(df["freq (events/min)"].values, np.array([2.0, 3.0]))
    np.testing.assert_allclose(df["amplitude"].values, np.array([1.5, 2.5]))
