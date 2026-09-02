# SPDX-License-Identifier: AGPL-3.0-or-later

"""Linux lease supervisor for a remote DGX rank process."""

from __future__ import annotations

import argparse
import ctypes
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

_HEARTBEAT = b"HERETIC_DGX_HEARTBEAT"
_PR_SET_PDEATHSIG = 1


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return seconds


def _child_parent_death_guard(expected_parent: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def _read_control(fd: int, pending: bytearray) -> tuple[bool, bool]:
    """Return (valid heartbeat received, control EOF received)."""

    chunk = os.read(fd, 4096)
    if not chunk:
        return False, True
    pending.extend(chunk)
    heartbeat = False
    while b"\n" in pending:
        frame, _, remainder = pending.partition(b"\n")
        pending[:] = remainder
        if frame != _HEARTBEAT:
            raise ValueError("malformed DGX rank control frame")
        heartbeat = True
    if len(pending) > len(_HEARTBEAT):
        raise ValueError("malformed DGX rank control frame")
    return heartbeat, False


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group: int, deadline: float) -> bool:
    while _process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _terminate_process_group_id(process_group: int, grace: float) -> None:
    if not _process_group_exists(process_group):
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for_process_group_exit(process_group, time.monotonic() + grace):
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_for_process_group_exit(process_group, time.monotonic() + grace)


def _terminate_process_group(process: subprocess.Popen[bytes], grace: float) -> None:
    _terminate_process_group_id(process.pid, grace)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def _rank_exit_code(return_code: int) -> int:
    return return_code if return_code >= 0 else 128 - return_code


def _guardian_supervise(
    argv: Sequence[str], lease_fd: int, term_grace_seconds: float
) -> int:
    expected_parent = os.getpid()
    process = subprocess.Popen(
        tuple(argv),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=lambda: _child_parent_death_guard(expected_parent),
    )
    selector = selectors.DefaultSelector()
    selector.register(lease_fd, selectors.EVENT_READ)
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                _terminate_process_group(process, term_grace_seconds)
                return _rank_exit_code(return_code)
            if not selector.select(0.1):
                continue
            if not os.read(lease_fd, 4096):
                _terminate_process_group(process, term_grace_seconds)
                return 125
    finally:
        selector.close()
        os.close(lease_fd)
        _terminate_process_group(process, term_grace_seconds)


def _start_guardian(
    argv: Sequence[str], term_grace_seconds: float
) -> tuple[int, int]:
    lease_read, lease_write = os.pipe()
    guardian_pid = os.fork()
    if guardian_pid == 0:
        os.close(lease_write)
        os.setsid()
        try:
            return_code = _guardian_supervise(argv, lease_read, term_grace_seconds)
        except BaseException as error:
            print(f"DGX rank guardian failed: {error}", file=sys.stderr, flush=True)
            return_code = 125
        os._exit(return_code)
    os.close(lease_read)
    return guardian_pid, lease_write


def _poll_guardian(guardian_pid: int) -> int | None:
    waited_pid, status = os.waitpid(guardian_pid, os.WNOHANG)
    if waited_pid == 0:
        return None
    return _rank_exit_code(os.waitstatus_to_exitcode(status))


def _wait_guardian(guardian_pid: int, timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while True:
        return_code = _poll_guardian(guardian_pid)
        if return_code is not None:
            return return_code
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def supervise(argv: Sequence[str], *, lease_seconds: float, term_grace_seconds: float) -> int:
    if sys.platform != "linux":
        raise RuntimeError("DGX rank supervision requires Linux")
    if not argv:
        raise ValueError("DGX rank command is required")

    control_fd = sys.stdin.buffer.fileno()
    selector = selectors.DefaultSelector()
    selector.register(control_fd, selectors.EVENT_READ)
    pending = bytearray()
    deadline = time.monotonic() + lease_seconds

    initial_heartbeat = False
    while not initial_heartbeat:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 125
        if not selector.select(remaining):
            return 125
        try:
            initial_heartbeat, eof = _read_control(control_fd, pending)
        except ValueError:
            return 125
        if eof:
            return 125

    guardian_pid, guardian_lease = _start_guardian(argv, term_grace_seconds)
    guardian_reaped = False
    lease_open = True
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }
    deadline = time.monotonic() + lease_seconds

    def revoke_guardian_lease() -> None:
        nonlocal lease_open
        if lease_open:
            os.close(guardian_lease)
            lease_open = False

    try:
        while True:
            guardian_return_code = _poll_guardian(guardian_pid)
            if guardian_return_code is not None:
                guardian_reaped = True
                return guardian_return_code
            if stop_requested:
                revoke_guardian_lease()
                _wait_guardian(guardian_pid, (2 * term_grace_seconds) + 1)
                guardian_reaped = True
                return 125
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                revoke_guardian_lease()
                _wait_guardian(guardian_pid, (2 * term_grace_seconds) + 1)
                guardian_reaped = True
                return 125
            events = selector.select(min(remaining, 0.1))
            if not events:
                continue
            try:
                heartbeat, eof = _read_control(control_fd, pending)
            except ValueError:
                revoke_guardian_lease()
                _wait_guardian(guardian_pid, (2 * term_grace_seconds) + 1)
                guardian_reaped = True
                return 125
            if eof:
                revoke_guardian_lease()
                _wait_guardian(guardian_pid, (2 * term_grace_seconds) + 1)
                guardian_reaped = True
                return 125
            if heartbeat:
                deadline = time.monotonic() + lease_seconds
    finally:
        selector.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        revoke_guardian_lease()
        if not guardian_reaped:
            _wait_guardian(guardian_pid, (2 * term_grace_seconds) + 1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lease-seconds", type=_positive_seconds, required=True)
    parser.add_argument("--term-grace-seconds", type=_positive_seconds, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    raise SystemExit(
        supervise(
            command,
            lease_seconds=arguments.lease_seconds,
            term_grace_seconds=arguments.term_grace_seconds,
        )
    )


if __name__ == "__main__":
    main()
