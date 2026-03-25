import logging
import socket
from random import randint

logger = logging.getLogger(__name__)


def scanPortsAndFind(start_port=5000, end_port=5200, host="127.0.0.1"):
    while True:
        port = randint(start_port, end_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.001)  # Set timeout to avoid long waiting on closed ports
        result = sock.connect_ex((host, port))
        if result == 0:  # If the connection is successful, the port is open
            continue
        else:
            break

    return port
