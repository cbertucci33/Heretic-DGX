# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import ipaddress
import json
import queue
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import TypedDict

from .distributed_protocol import (
    LoadAcknowledgement,
    LoadCommand,
    ModelLoadIdentity,
    validate_load_acknowledgement,
)

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
_FRAME_HEADER = struct.Struct("!I")


class _WireIdentity(TypedDict):
    base_model_id: str
    base_model_revision: str
    dtype: str
    quantization_config: str


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_canonical_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("wire payload must be exactly bytes")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("wire payload must be UTF-8 JSON") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("wire payload must be JSON") from error
    if type(value) is not dict:
        raise ValueError("wire payload must be a JSON object")
    if _canonical_json_bytes(value) != payload:
        raise ValueError("wire payload must use canonical JSON")
    return value


def _require_exact_keys(
    value: dict[str, object],
    expected: frozenset[str],
    object_name: str,
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{object_name} has missing or unknown fields")


def _identity_to_wire(identity: ModelLoadIdentity) -> _WireIdentity:
    revalidated = ModelLoadIdentity(
        base_model_id=identity.base_model_id,
        base_model_revision=identity.base_model_revision,
        dtype=identity.dtype,
        quantization_config=identity.quantization_config,
    )
    return {
        "base_model_id": revalidated.base_model_id,
        "base_model_revision": revalidated.base_model_revision,
        "dtype": revalidated.dtype,
        "quantization_config": revalidated.quantization_config,
    }


def _identity_from_wire(value: object) -> ModelLoadIdentity:
    if type(value) is not dict:
        raise TypeError("identity must be a JSON object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "base_model_id",
                "base_model_revision",
                "dtype",
                "quantization_config",
            }
        ),
        "identity",
    )
    return ModelLoadIdentity(
        base_model_id=value["base_model_id"],  # type: ignore[arg-type]
        base_model_revision=value["base_model_revision"],  # type: ignore[arg-type]
        dtype=value["dtype"],  # type: ignore[arg-type]
        quantization_config=value["quantization_config"],  # type: ignore[arg-type]
    )


def encode_load_command(command: LoadCommand) -> bytes:
    """Encode one validated rank-0 LOAD command as canonical JSON bytes."""
    if type(command) is not LoadCommand:
        raise TypeError("command must be exactly LoadCommand")
    revalidated = LoadCommand(
        command_id=command.command_id,
        identity=ModelLoadIdentity(**_identity_to_wire(command.identity)),
    )
    return _canonical_json_bytes(
        {
            "command_id": revalidated.command_id,
            "identity": _identity_to_wire(revalidated.identity),
            "kind": "LOAD",
            "version": PROTOCOL_VERSION,
        }
    )


def decode_load_command(payload: bytes) -> LoadCommand:
    """Decode and validate one canonical rank-0 LOAD command."""
    value = _decode_canonical_object(payload)
    _require_exact_keys(
        value,
        frozenset({"command_id", "identity", "kind", "version"}),
        "LOAD command",
    )
    if value["kind"] != "LOAD" or type(value["kind"]) is not str:
        raise ValueError("wire message kind must be LOAD")
    if type(value["version"]) is not int or value["version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported LOAD protocol version")
    return LoadCommand(
        command_id=value["command_id"],  # type: ignore[arg-type]
        identity=_identity_from_wire(value["identity"]),
    )


def encode_load_acknowledgement(acknowledgement: LoadAcknowledgement) -> bytes:
    """Encode one validated rank-1 LOAD acknowledgement."""
    if type(acknowledgement) is not LoadAcknowledgement:
        raise TypeError("acknowledgement must be exactly LoadAcknowledgement")
    revalidated = LoadAcknowledgement(
        command_id=acknowledgement.command_id,
        rank=acknowledgement.rank,
        identity=ModelLoadIdentity(**_identity_to_wire(acknowledgement.identity)),
    )
    return _canonical_json_bytes(
        {
            "command_id": revalidated.command_id,
            "identity": _identity_to_wire(revalidated.identity),
            "kind": "LOAD_ACK",
            "rank": revalidated.rank,
            "version": PROTOCOL_VERSION,
        }
    )


def decode_load_acknowledgement(payload: bytes) -> LoadAcknowledgement:
    """Decode and validate one canonical rank-1 LOAD acknowledgement."""
    value = _decode_canonical_object(payload)
    _require_exact_keys(
        value,
        frozenset({"command_id", "identity", "kind", "rank", "version"}),
        "LOAD acknowledgement",
    )
    if value["kind"] != "LOAD_ACK" or type(value["kind"]) is not str:
        raise ValueError("wire message kind must be LOAD_ACK")
    if type(value["version"]) is not int or value["version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported LOAD protocol version")
    return LoadAcknowledgement(
        command_id=value["command_id"],  # type: ignore[arg-type]
        rank=value["rank"],  # type: ignore[arg-type]
        identity=_identity_from_wire(value["identity"]),
    )


def _set_remaining_timeout(connection: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise socket.timeout("transport deadline expired")
    connection.settimeout(remaining)


def _require_before_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise socket.timeout("transport deadline expired")


def _require_loopback_address(address: object, endpoint: str) -> None:
    if type(address) is not str:
        raise TypeError(f"{endpoint} address must be exactly str")
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as error:
        raise ValueError(f"{endpoint} address must be a loopback IP literal") from error
    if not parsed.is_loopback:
        raise ValueError(f"{endpoint} address must be loopback")


def _load_with_deadline(
    load_identity: Callable[[ModelLoadIdentity], ModelLoadIdentity],
    requested_identity: ModelLoadIdentity,
    deadline: float,
) -> ModelLoadIdentity:
    result_queue: queue.Queue[ModelLoadIdentity | BaseException] = queue.Queue(
        maxsize=1
    )

    def invoke_loader() -> None:
        try:
            result_queue.put(load_identity(requested_identity))
        except BaseException as error:
            result_queue.put(error)

    loader_thread = threading.Thread(target=invoke_loader, daemon=True)
    loader_thread.start()
    loader_thread.join(max(deadline - time.monotonic(), 0.0))
    if loader_thread.is_alive():
        raise socket.timeout("rank-1 loader deadline expired")
    result = result_queue.get_nowait()
    if isinstance(result, BaseException):
        raise result
    return result


def _receive_exact(
    connection: socket.socket,
    byte_count: int,
    *,
    deadline: float | None = None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        if deadline is not None:
            _set_remaining_timeout(connection, deadline)
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("truncated transport frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(connection: socket.socket, payload: bytes) -> None:
    """Send one bounded length-prefixed payload."""
    if type(payload) is not bytes:
        raise TypeError("frame payload must be exactly bytes")
    if not payload:
        raise ValueError("transport frame must not be empty")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("transport frame exceeds maximum size")
    connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)


def receive_frame(
    connection: socket.socket,
    *,
    deadline: float | None = None,
) -> bytes:
    """Receive one bounded length-prefixed payload or fail closed."""
    header = _receive_exact(connection, _FRAME_HEADER.size, deadline=deadline)
    (payload_size,) = _FRAME_HEADER.unpack(header)
    if payload_size == 0:
        raise ValueError("transport frame must not be empty")
    if payload_size > MAX_FRAME_BYTES:
        raise ValueError("transport frame exceeds maximum size")
    return _receive_exact(connection, payload_size, deadline=deadline)


def coordinate_load_once(
    listener: socket.socket,
    command: LoadCommand,
    *,
    timeout_seconds: float,
) -> LoadAcknowledgement:
    """Perform one fail-closed rank-0 LOAD exchange on an existing listener."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    _require_loopback_address(listener.getsockname()[0], "listener")
    deadline = time.monotonic() + timeout_seconds
    _set_remaining_timeout(listener, deadline)
    try:
        connection, _peer = listener.accept()
    except socket.timeout as error:
        raise TimeoutError("timed out waiting for rank-1 worker") from error
    with connection:
        _require_loopback_address(_peer[0], "peer")
        try:
            command_payload = encode_load_command(command)
            _require_before_deadline(deadline)
            _set_remaining_timeout(connection, deadline)
            send_frame(connection, command_payload)
            _require_before_deadline(deadline)
            acknowledgement_payload = receive_frame(
                connection,
                deadline=deadline,
            )
            _require_before_deadline(deadline)
            acknowledgement = decode_load_acknowledgement(acknowledgement_payload)
            _require_before_deadline(deadline)
            validate_load_acknowledgement(command, acknowledgement)
            _require_before_deadline(deadline)
        except socket.timeout as error:
            raise TimeoutError("timed out during rank-1 LOAD exchange") from error
    return acknowledgement


def run_load_worker_once(
    host: str,
    port: int,
    load_identity: Callable[[ModelLoadIdentity], ModelLoadIdentity],
    *,
    timeout_seconds: float,
) -> LoadAcknowledgement:
    """Connect rank 1, load the requested identity, and acknowledge exactly once."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    _require_loopback_address(host, "worker host")
    deadline = time.monotonic() + timeout_seconds
    try:
        _require_before_deadline(deadline)
        connection = socket.create_connection(
            (host, port), timeout=max(deadline - time.monotonic(), 0.0)
        )
    except socket.timeout as error:
        raise TimeoutError("timed out connecting rank-1 worker") from error
    with connection:
        try:
            _require_before_deadline(deadline)
            command_payload = receive_frame(connection, deadline=deadline)
            _require_before_deadline(deadline)
            command = decode_load_command(command_payload)
            _require_before_deadline(deadline)
            loaded_identity = _load_with_deadline(
                load_identity,
                command.identity,
                deadline,
            )
            _require_before_deadline(deadline)
            acknowledgement = LoadAcknowledgement(
                command_id=command.command_id,
                rank=1,
                identity=loaded_identity,
            )
            acknowledgement_payload = encode_load_acknowledgement(acknowledgement)
            _require_before_deadline(deadline)
            _set_remaining_timeout(connection, deadline)
            send_frame(connection, acknowledgement_payload)
            _require_before_deadline(deadline)
        except socket.timeout as error:
            raise TimeoutError("timed out during rank-1 LOAD exchange") from error
    return acknowledgement
