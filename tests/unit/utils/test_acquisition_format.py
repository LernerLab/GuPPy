"""Unit tests for resolving a session folder's acquisition format for NWB export.

NWB export reads a session through one format's interfaces, so it has to collapse the set of
formats GuPPy detects in a folder into a single choice -- and refuse the folders where it cannot.
"""

import pytest

from guppy.utils.acquisition_format import (
    acquisition_supplies_session_start_time,
    resolve_acquisition_format,
)


@pytest.fixture
def session_path(tmp_path):
    """An empty session folder to drop format marker files into."""
    session = tmp_path / "Photo_session"
    session.mkdir()
    return session


def write_tdt(session_path):
    (session_path / "Photo_session.tsq").write_bytes(b"\x00")


def write_doric_hdf5(session_path):
    (session_path / "Photo_session.doric").write_bytes(b"\x00")


def write_doric_csv(session_path):
    # A Doric CSV export is recognized by its two all-string header rows.
    (session_path / "traces.csv").write_text("Console,AIn-1,DI--O-1\nTime(s),Values,Values\n0.0,1.0,0\n")


def write_npm(session_path):
    (session_path / "raw.csv").write_text("Timestamp,LedState,Region0G,Region1G\n0.0,1,10.0,11.0\n")


def write_csv_data(session_path):
    (session_path / "signal_dms.csv").write_text("timestamps,data,sampling_rate\n0.0,1.0,100.0\n")


def write_csv_event(session_path):
    (session_path / "port_entries.csv").write_text("timestamps\n1.5\n2.5\n")


def write_nwb(session_path):
    (session_path / "session.nwb").write_bytes(b"\x00")


class TestResolveAcquisitionFormat:
    @pytest.mark.parametrize(
        "write_session, expected_format",
        [
            (write_tdt, "tdt"),
            (write_doric_hdf5, "doric"),
            (write_doric_csv, "doric"),
            (write_npm, "npm"),
            (write_csv_data, "csv"),
            (write_csv_event, "csv"),
        ],
    )
    def test_single_format_resolves(self, session_path, write_session, expected_format):
        write_session(session_path)
        assert resolve_acquisition_format(str(session_path)) == expected_format

    def test_empty_folder_names_the_folder(self, session_path):
        with pytest.raises(ValueError) as excinfo:
            resolve_acquisition_format(str(session_path))
        assert "No acquisition data was found" in str(excinfo.value)
        assert str(session_path) in str(excinfo.value)

    def test_nwb_source_is_rejected(self, session_path):
        write_nwb(session_path)
        with pytest.raises(ValueError) as excinfo:
            resolve_acquisition_format(str(session_path))
        message = str(excinfo.value)
        assert "does not support the 'nwb' acquisition format" in message
        assert "no raw acquisition to bundle" in message

    def test_tdt_with_a_custom_event_csv_is_rejected_as_mixed(self, session_path):
        # Custom events are written as single-column CSVs into the session folder, which makes a TDT
        # session a 'csv' source as well -- and the converter reads traces and events as one format.
        write_tdt(session_path)
        write_csv_event(session_path)

        with pytest.raises(ValueError) as excinfo:
            resolve_acquisition_format(str(session_path))
        message = str(excinfo.value)
        assert "more than one acquisition format" in message
        assert "['csv', 'tdt']" in message
        assert "https://github.com/LernerLab/GuPPy/issues/new" in message

    def test_two_photometry_formats_are_rejected_as_mixed(self, session_path):
        write_doric_hdf5(session_path)
        write_npm(session_path)

        with pytest.raises(ValueError) as excinfo:
            resolve_acquisition_format(str(session_path))
        message = str(excinfo.value)
        assert "['doric', 'npm']" in message


class TestAcquisitionSuppliesSessionStartTime:
    def test_tdt_supplies_it(self, session_path):
        write_tdt(session_path)
        assert acquisition_supplies_session_start_time(session_folder_path=str(session_path), acquisition_format="tdt")

    @pytest.mark.parametrize("acquisition_format", ["doric", "npm", "csv"])
    def test_every_other_format_leaves_it_to_the_form(self, session_path, acquisition_format):
        # A .doric HDF5 export carries a creation timestamp only when the acquisition software wrote
        # one, so Doric cannot be counted on either.
        write_doric_hdf5(session_path)
        assert not acquisition_supplies_session_start_time(
            session_folder_path=str(session_path), acquisition_format=acquisition_format
        )
