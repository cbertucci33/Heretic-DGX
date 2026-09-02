# SPDX-License-Identifier: AGPL-3.0-or-later

import multiprocessing
import socket
import struct
import threading
import time
import unittest
from multiprocessing.process import BaseProcess
from unittest.mock import patch

import heretic.distributed_transport as distributed_transport

from heretic.distributed_protocol import (
    LoadAcknowledgement,
    LoadCommand,
    ModelLoadIdentity,
    canonicalize_quantization_config,
)
from heretic.distributed_transport import (
    MAX_FRAME_BYTES,
    coordinate_load_once,
    decode_load_acknowledgement,
    decode_load_command,
    encode_load_acknowledgement,
    encode_load_command,
    receive_frame,
    run_load_worker_once,
    send_frame,
)


def _identity(*, bits: int = 8) -> ModelLoadIdentity:
    return ModelLoadIdentity(
        base_model_id="example/tiny-quantized-model",
        base_model_revision="a" * 40,
        dtype="torch.bfloat16",
        quantization_config=canonicalize_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {"weights": {"num_bits": bits}}
                },
            }
        ),
    )


def _worker_entry(
    host: str,
    port: int,
    loaded_identity: ModelLoadIdentity,
) -> None:
    def loader(_requested: ModelLoadIdentity) -> ModelLoadIdentity:
        return loaded_identity

    run_load_worker_once(host, port, loader, timeout_seconds=5.0)


class LoadWireCodecTests(unittest.TestCase):
    def test_round_trips_load_records_canonically(self) -> None:
        identity = _identity()
        command = LoadCommand(command_id=7, identity=identity)
        acknowledgement = LoadAcknowledgement(
            command_id=7,
            rank=1,
            identity=identity,
        )

        self.assertEqual(decode_load_command(encode_load_command(command)), command)
        self.assertEqual(
            decode_load_acknowledgement(
                encode_load_acknowledgement(acknowledgement)
            ),
            acknowledgement,
        )
        self.assertEqual(encode_load_command(command), encode_load_command(command))

    def test_rejects_noncanonical_or_unknown_wire_data(self) -> None:
        valid = encode_load_command(LoadCommand(command_id=7, identity=_identity()))
        malformed_payloads = (
            b"{}",
            b'{"kind":"LOAD","version":1,"command_id":true,"identity":{}}',
            valid + b" ",
            valid.replace(b'"version":1', b'"version":2'),
            valid.replace(b'"kind":"LOAD"', b'"kind":"OTHER"'),
            valid[:-1] + b',"extra":1}',
            b'{"kind":"LOAD","kind":"LOAD","version":1}',
            b'{"kind":"LOAD","version":NaN}',
        )
        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    decode_load_command(payload)

    def test_rejects_truncated_and_oversized_frames(self) -> None:
        receiver, sender = socket.socketpair()
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        receiver.settimeout(0.5)

        sender.sendall(struct.pack("!I", 5) + b"abc")
        sender.shutdown(socket.SHUT_WR)
        with self.assertRaisesRegex(ConnectionError, "truncated"):
            receive_frame(receiver)

        receiver.close()
        sender.close()
        receiver, sender = socket.socketpair()
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        sender.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "maximum"):
            receive_frame(receiver)


class TwoProcessLoadHandshakeTests(unittest.TestCase):
    def _listener(self) -> tuple[socket.socket, str, int]:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        return listener, host, port

    def _spawn_worker(
        self,
        host: str,
        port: int,
        loaded_identity: ModelLoadIdentity,
    ) -> BaseProcess:
        process = multiprocessing.get_context("spawn").Process(
            target=_worker_entry,
            args=(host, port, loaded_identity),
        )
        process.start()
        return process

    def test_completes_real_two_process_load_handshake(self) -> None:
        identity = _identity()
        command = LoadCommand(command_id=11, identity=identity)
        listener, host, port = self._listener()
        process = self._spawn_worker(host, port, identity)
        try:
            acknowledgement = coordinate_load_once(
                listener,
                command,
                timeout_seconds=5.0,
            )
        finally:
            listener.close()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(acknowledgement.identity, identity)
        self.assertEqual(acknowledgement.rank, 1)

    def test_rejects_worker_that_loaded_different_quantization(self) -> None:
        command = LoadCommand(command_id=12, identity=_identity(bits=8))
        listener, host, port = self._listener()
        process = self._spawn_worker(host, port, _identity(bits=4))
        try:
            with self.assertRaisesRegex(RuntimeError, "quantization_config"):
                coordinate_load_once(listener, command, timeout_seconds=5.0)
        finally:
            listener.close()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)

        self.assertEqual(process.exitcode, 0)

    def test_times_out_without_worker_and_closes_gate(self) -> None:
        listener, _host, _port = self._listener()
        try:
            with self.assertRaises(TimeoutError):
                coordinate_load_once(
                    listener,
                    LoadCommand(command_id=13, identity=_identity()),
                    timeout_seconds=0.05,
                )
        finally:
            listener.close()

    def test_timeout_is_one_deadline_not_reset_by_incremental_bytes(self) -> None:
        listener, host, port = self._listener()
        worker_errors: list[BaseException] = []

        def trickle_worker() -> None:
            try:
                with socket.create_connection((host, port), timeout=1.0) as connection:
                    decode_load_command(receive_frame(connection))
                    payload = encode_load_acknowledgement(
                        LoadAcknowledgement(14, 1, _identity())
                    )
                    frame = struct.pack("!I", len(payload)) + payload
                    for byte in frame:
                        connection.sendall(bytes([byte]))
                        time.sleep(0.03)
            except OSError:
                pass
            except BaseException as error:
                worker_errors.append(error)

        worker = threading.Thread(target=trickle_worker)
        worker.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                coordinate_load_once(
                    listener,
                    LoadCommand(command_id=14, identity=_identity()),
                    timeout_seconds=0.10,
                )
        finally:
            listener.close()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(worker_errors)
        self.assertLess(time.monotonic() - started, 0.50)

    def test_coordinator_rejects_ack_processing_after_deadline(self) -> None:
        original_decode = distributed_transport.decode_load_acknowledgement
        original_validate = distributed_transport.validate_load_acknowledgement

        def delayed_decode(payload: bytes) -> LoadAcknowledgement:
            result = original_decode(payload)
            time.sleep(0.10)
            return result

        def delayed_validate(
            command: LoadCommand,
            acknowledgement: LoadAcknowledgement,
        ) -> None:
            original_validate(
                command,
                acknowledgement,
            )
            time.sleep(0.10)

        delayed_functions: tuple[tuple[str, object], ...] = (
            ("decode_load_acknowledgement", delayed_decode),
            ("validate_load_acknowledgement", delayed_validate),
        )
        for function_name, delayed in delayed_functions:
            with self.subTest(function_name=function_name):
                listener, host, port = self._listener()
                worker_errors: list[BaseException] = []

                def worker() -> None:
                    try:
                        with socket.create_connection(
                            (host, port), timeout=1.0
                        ) as connection:
                            command = decode_load_command(receive_frame(connection))
                            send_frame(
                                connection,
                                encode_load_acknowledgement(
                                    LoadAcknowledgement(
                                        command.command_id,
                                        1,
                                        command.identity,
                                    )
                                ),
                            )
                    except BaseException as error:
                        worker_errors.append(error)

                worker_thread = threading.Thread(target=worker)
                worker_thread.start()
                started = time.monotonic()
                try:
                    with patch.object(
                        distributed_transport,
                        function_name,
                        side_effect=delayed,
                    ):
                        with self.assertRaises(TimeoutError):
                            coordinate_load_once(
                                listener,
                                LoadCommand(17, _identity()),
                                timeout_seconds=0.05,
                            )
                finally:
                    listener.close()
                    worker_thread.join(1.0)

                self.assertFalse(worker_thread.is_alive())
                self.assertFalse(worker_errors)
                self.assertLess(time.monotonic() - started, 0.20)

    def test_worker_deadline_bounds_slow_loader_callback(self) -> None:
        listener, host, port = self._listener()
        coordinator_errors: list[BaseException] = []

        def coordinator() -> None:
            try:
                with listener.accept()[0] as connection:
                    connection.settimeout(1.0)
                    send_frame(
                        connection,
                        encode_load_command(LoadCommand(15, _identity())),
                    )
                    with self.assertRaises(ConnectionError):
                        receive_frame(connection)
            except BaseException as error:
                coordinator_errors.append(error)

        def slow_loader(_requested: ModelLoadIdentity) -> ModelLoadIdentity:
            time.sleep(0.25)
            return _identity()

        coordinator_thread = threading.Thread(target=coordinator)
        coordinator_thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                run_load_worker_once(
                    host,
                    port,
                    slow_loader,
                    timeout_seconds=0.05,
                )
        finally:
            listener.close()
            coordinator_thread.join(2.0)

        self.assertFalse(coordinator_thread.is_alive())
        self.assertFalse(coordinator_errors)
        self.assertLess(time.monotonic() - started, 0.20)

    def test_worker_rejects_ack_encoding_after_deadline(self) -> None:
        listener, host, port = self._listener()
        coordinator_errors: list[BaseException] = []
        acknowledgements: list[bytes] = []

        def coordinator() -> None:
            try:
                with listener.accept()[0] as connection:
                    connection.settimeout(1.0)
                    send_frame(
                        connection,
                        encode_load_command(LoadCommand(18, _identity())),
                    )
                    try:
                        acknowledgements.append(receive_frame(connection))
                    except ConnectionError:
                        pass
            except BaseException as error:
                coordinator_errors.append(error)

        original_encode = distributed_transport.encode_load_acknowledgement

        def delayed_encode(acknowledgement: LoadAcknowledgement) -> bytes:
            payload = original_encode(acknowledgement)
            time.sleep(0.10)
            return payload

        coordinator_thread = threading.Thread(target=coordinator)
        coordinator_thread.start()
        started = time.monotonic()
        try:
            with patch.object(
                distributed_transport,
                "encode_load_acknowledgement",
                side_effect=delayed_encode,
            ):
                with self.assertRaises(TimeoutError):
                    run_load_worker_once(
                        host,
                        port,
                        lambda requested: requested,
                        timeout_seconds=0.05,
                    )
        finally:
            listener.close()
            coordinator_thread.join(1.0)

        self.assertFalse(coordinator_thread.is_alive())
        self.assertFalse(coordinator_errors)
        self.assertFalse(acknowledgements)
        self.assertLess(time.monotonic() - started, 0.20)

    def test_rejects_non_loopback_transport_endpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            run_load_worker_once(
                "192.0.2.1",
                12345,
                lambda requested: requested,
                timeout_seconds=0.05,
            )

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        try:
            with self.assertRaisesRegex(ValueError, "loopback"):
                coordinate_load_once(
                    listener,
                    LoadCommand(16, _identity()),
                    timeout_seconds=0.05,
                )
        finally:
            listener.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    unittest.main()
