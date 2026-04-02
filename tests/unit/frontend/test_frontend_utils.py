import socket
from unittest.mock import patch

from guppy.frontend.frontend_utils import scanPortsAndFind


def test_scan_ports_and_find_returns_integer_in_range():
    port = scanPortsAndFind()
    assert isinstance(port, int)
    assert 5000 <= port <= 5200


def test_scan_ports_and_find_retries_occupied_port():
    """When the first drawn port is occupied (connect_ex returns 0), the
    function should keep trying and return a different port."""
    call_count = 0
    first_port = None

    original_connect_ex = socket.socket.connect_ex

    def mock_connect_ex(self, address):
        nonlocal call_count, first_port
        call_count += 1
        if call_count == 1:
            first_port = address[1]
            return 0  # Simulate occupied port
        return 1  # Simulate free port

    with patch.object(socket.socket, "connect_ex", mock_connect_ex):
        result = scanPortsAndFind()

    assert call_count >= 2
    assert result != first_port
