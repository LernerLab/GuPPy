import logging
import socket
from random import randint

logger = logging.getLogger(__name__)


# Ports that all major browsers refuse to connect to (ERR_UNSAFE_PORT).
# Defined in the WHATWG Fetch Standard (https://fetch.spec.whatwg.org/#bad-port),
# a cross-browser spec maintained by Apple, Google, Mozilla, and Microsoft.
_CHROME_UNSAFE_PORTS = {5060, 5061}


def scanPortsAndFind(start_port=5000, end_port=5200, host="127.0.0.1"):
    while True:
        port = randint(start_port, end_port)
        if port in _CHROME_UNSAFE_PORTS:
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.001)  # Set timeout to avoid long waiting on closed ports
        result = sock.connect_ex((host, port))
        if result == 0:  # If the connection is successful, the port is open
            continue
        else:
            break

    return port
