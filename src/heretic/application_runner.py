# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from .launch_plan import RankLaunchPlan
from .preflight_collector import collect_rank_preflights
from .rank_preflight import RankPreflightIdentity


@dataclass(frozen=True, slots=True)
class RankApplicationResult:
    """Captured output from one successfully completed rank application."""

    rank: int
    stdout: str
    stderr: str
    stdout_log: Path
    stderr_log: Path


def _rank_log_paths(plan: RankLaunchPlan) -> tuple[Path, Path]:
    """Allocate private, durable output files before launching a rank."""
    configured_directory = os.environ.get("HERETIC_LOG_DIR")
    log_directory = (
        Path(configured_directory).expanduser()
        if configured_directory
        else Path.home() / ".local" / "state" / "heretic" / "rank-logs"
    )
    log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    prefix = f"{timestamp}-pid{os.getpid()}-rank{plan.rank}"
    return (
        log_directory / f"{prefix}.stdout.log",
        log_directory / f"{prefix}.stderr.log",
    )


def _open_private_log(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_private_log(path: Path, content: str) -> None:
    with _open_private_log(path) as log_file:
        log_file.write(content)


def _failure_detail(stdout: str, stderr: str) -> str:
    """Retain the tail of each diagnostic stream for a failed rank."""
    parts = []
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()[-8000:]}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()[-8000:]}")
    return "\n".join(parts)


def _log_reference(stdout_log: Path, stderr_log: Path) -> str:
    return f"full logs: stdout={stdout_log} stderr={stderr_log}"


def run_rank_application_plan(
    plan: RankLaunchPlan,
    *,
    timeout_seconds: int,
    cancellation: Event | None = None,
) -> RankApplicationResult:
    """Run one rank through a bounded local or noninteractive SSH command."""

    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("rank application timeout must be a positive integer")

    rank_argv = (
        "timeout",
        "--kill-after=5s",
        f"{timeout_seconds}s",
        "env",
        *(f"{name}={value}" for name, value in sorted(plan.environment)),
        *plan.argv,
    )
    command = rank_argv
    workdir: str | None = plan.workdir
    if plan.rank == 1:
        remote_command = (
            f"cd {shlex.quote(plan.workdir)} && exec {shlex.join(rank_argv)}"
        )
        command = (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            plan.host,
            remote_command,
        )
        workdir = None

    if cancellation is None:
        stdout_log, stderr_log = _rank_log_paths(plan)
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            _write_private_log(stdout_log, stdout)
            _write_private_log(stderr_log, stderr)
            raise RuntimeError(
                f"rank {plan.rank} application exceeded its cleanup deadline; "
                f"{_log_reference(stdout_log, stderr_log)}"
            ) from error
        _write_private_log(stdout_log, completed.stdout)
        _write_private_log(stderr_log, completed.stderr)
        if completed.returncode != 0:
            detail = _failure_detail(completed.stdout, completed.stderr)
            raise RuntimeError(
                f"rank {plan.rank} application failed with exit code "
                f"{completed.returncode}; "
                f"{_log_reference(stdout_log, stderr_log)}: {detail}"
            )
        return RankApplicationResult(
            rank=plan.rank,
            stdout=completed.stdout,
            stderr=completed.stderr,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    if cancellation.is_set():
        raise RuntimeError(f"rank {plan.rank} application cancelled before launch")

    stdout_log, stderr_log = _rank_log_paths(plan)
    with (
        _open_private_log(stdout_log) as stdout_file,
        _open_private_log(stderr_log) as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout_seconds + 10
    while True:
        if cancellation.is_set():
            _terminate_process_group(process)
            raise RuntimeError(
                f"rank {plan.rank} application cancelled because its peer failed; "
                f"{_log_reference(stdout_log, stderr_log)}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            raise RuntimeError(
                f"rank {plan.rank} application exceeded its cleanup deadline; "
                f"{_log_reference(stdout_log, stderr_log)}"
            )
        try:
            process.wait(timeout=min(0.1, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    stdout = _read_log(stdout_log)
    stderr = _read_log(stderr_log)
    if process.returncode != 0:
        detail = _failure_detail(stdout, stderr)
        raise RuntimeError(
            f"rank {plan.rank} application failed with exit code "
            f"{process.returncode}; {_log_reference(stdout_log, stderr_log)}: "
            f"{detail}"
        )
    return RankApplicationResult(
        rank=plan.rank,
        stdout=stdout,
        stderr=stderr,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Stop a rank wrapper and its local descendants within five seconds."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def collect_rank_applications(
    plans: tuple[RankLaunchPlan, RankLaunchPlan],
    *,
    timeout_seconds: int,
) -> tuple[RankApplicationResult, RankApplicationResult]:
    """Run exactly two rank applications concurrently and preserve rank order."""

    if tuple(plan.rank for plan in plans) != (0, 1):
        raise ValueError("rank launch plans must be ordered as rank 0 then rank 1")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("rank application timeout must be a positive integer")

    results: dict[int, RankApplicationResult] = {}
    cancellation = Event()
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                run_rank_application_plan,
                plan,
                timeout_seconds=timeout_seconds,
                cancellation=cancellation,
            ): plan.rank
            for plan in plans
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    cancellation.set()
            else:
                results[result.rank] = result
    if first_error is not None:
        raise first_error
    return results[0], results[1]


def preflight_and_collect_rank_applications(
    plans: tuple[RankLaunchPlan, RankLaunchPlan],
    checkpoint_directory: str,
    *,
    timeout_seconds: int,
) -> tuple[
    RankPreflightIdentity,
    tuple[RankApplicationResult, RankApplicationResult],
]:
    """Require matching rank identities before starting either application."""

    identity = collect_rank_preflights(
        plans,
        checkpoint_directory,
        timeout_seconds=timeout_seconds,
    )
    return identity, collect_rank_applications(
        plans,
        timeout_seconds=timeout_seconds,
    )
