# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


HEARTBEAT = b"HERETIC_DGX_HEARTBEAT\n"


def _start_supervisor(*child_argv: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        (
            sys.executable,
            "-m",
            "heretic.remote_rank_supervisor",
            "--lease-seconds",
            "0.5",
            "--term-grace-seconds",
            "0.2",
            "--",
            *child_argv,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_preserves_normal_exit_status_and_output() -> None:
    process = _start_supervisor(
        sys.executable,
        "-c",
        "print('rank-output', flush=True); raise SystemExit(23)",
    )
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    process.wait(timeout=5)
    assert process.stdout is not None
    output = process.stdout.read()

    assert process.returncode == 23
    assert output == b"rank-output\n"


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_control_eof_kills_child_that_ignores_term(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    process = _start_supervisor(
        sys.executable,
        "-c",
        (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(marker)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    )
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child_pid = int(marker.read_text())

        process.stdin.close()
        process.wait(timeout=5)

        deadline = time.monotonic() + 2
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_heartbeat_expiry_kills_child_with_control_open(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "expired-child.pid"
    process = _start_supervisor(
        sys.executable,
        "-c",
        (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(marker)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    )
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child_pid = int(marker.read_text())

        process.wait(timeout=5)

        assert process.returncode == 125
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_sigkill_triggers_child_parent_death_signal(tmp_path: Path) -> None:
    marker = tmp_path / "parent-death-child.pid"
    process = _start_supervisor(
        sys.executable,
        "-c",
        (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(marker)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    )
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child_pid = int(marker.read_text())

        process.kill()
        process.wait(timeout=5)

        deadline = time.monotonic() + 2
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_control_eof_kills_descendant_after_leader_exits(tmp_path: Path) -> None:
    marker = tmp_path / "descendant.pid"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,\"-c\",{descendant_code!r}]); "
        "time.sleep(60)"
    )
    process = _start_supervisor(sys.executable, "-c", leader_code)
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    descendant_pid = 0
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        descendant_pid = int(marker.read_text())

        process.stdin.close()
        process.wait(timeout=5)

        deadline = time.monotonic() + 2
        while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{descendant_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if descendant_pid:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_sigkill_does_not_orphan_rank_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "sigkill-descendant.pid"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,\"-c\",{descendant_code!r}]); "
        "time.sleep(60)"
    )
    process = _start_supervisor(sys.executable, "-c", leader_code)
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    descendant_pid = 0
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        descendant_pid = int(marker.read_text())

        process.kill()
        process.wait(timeout=5)

        deadline = time.monotonic() + 2
        while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{descendant_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if descendant_pid:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_supervisor_process_group_hup_does_not_orphan_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "group-hup-descendant.pid"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,\"-c\",{descendant_code!r}]); "
        "time.sleep(60)"
    )
    process = _start_supervisor(sys.executable, "-c", leader_code)
    assert process.stdin is not None
    process.stdin.write(HEARTBEAT)
    process.stdin.flush()

    descendant_pid = 0
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        descendant_pid = int(marker.read_text())

        os.killpg(process.pid, signal.SIGHUP)
        process.wait(timeout=5)

        deadline = time.monotonic() + 2
        while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{descendant_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if descendant_pid:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
