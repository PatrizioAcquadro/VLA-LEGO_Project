"""Local sensor/action transport between incompatible native Python runtimes."""

from __future__ import annotations

import json
import socket
import struct


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Policy runtime disconnected.")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, header: dict, payload: bytes = b"") -> None:
    encoded = json.dumps(header, allow_nan=False).encode()
    connection.sendall(struct.pack("!II", len(encoded), len(payload)) + encoded + payload)


def receive_message(connection: socket.socket) -> tuple[dict, bytes]:
    header_size, payload_size = struct.unpack("!II", _receive_exact(connection, 8))
    if header_size > 65536 or payload_size > 8 * 1024**2:
        raise ValueError("Unexpected sensor/action message size.")
    header = json.loads(_receive_exact(connection, header_size))
    return header, _receive_exact(connection, payload_size)
