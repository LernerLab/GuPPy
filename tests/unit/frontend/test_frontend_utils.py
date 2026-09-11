import socket
from unittest.mock import patch

from guppy.frontend.frontend_utils import scanPortsAndFind


def test_scan_ports_and_find_returns_integer_in_range():
    port = scanPortsAndFind()
    assert isinstance(port, int)
    assert 5000 <= port <= 5200


def test_scan_ports_and_find_retries_occupied_port():
    """A drawn port that is already bound is rejected, and the search draws again."""
    # The draw is patched rather than left random: the function redraws uniformly from the whole
    # range, so a real retry may legitimately land on the port it just rejected. Fixing the
    # sequence is what makes "it kept going" assertable at all.
    drawn_ports = iter([5000, 5000, 5001])
    occupied_ports = {5000}

    def mock_connect_ex(self, address):
        return 0 if address[1] in occupied_ports else 1  # 0 means the port answered, so it is taken

    with (
        patch("guppy.frontend.frontend_utils.randint", lambda start, end: next(drawn_ports)),
        patch.object(socket.socket, "connect_ex", mock_connect_ex),
    ):
        result = scanPortsAndFind()

    assert result == 5001
    assert next(drawn_ports, None) is None, "expected all three draws to be consumed"
