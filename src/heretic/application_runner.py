# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import shlex
import signal
import subprocess
from threading import Event
import time

from .launch_plan import RankLaunchPlan
from .preflight_collector import collect_rank_preflights
from .rank_preflight import RankPreflightIdentity


@dataclass(frozen=True, slots=True)
class RankApplicationResult:
    """Captured output from one successfully completed rank application."""

    rank: int
    stdout: str
    stderr: str


def _failure_detail(stdout: str, stderr: str) -> str:
    """Retain the tail of each diagnostic stream for a failed rank."""
    parts = []
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()[-8000:]}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()[-8000:]}")
    return "\n".join(parts)


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
            raise RuntimeError(
                f"rank {plan.rank} application exceeded its cleanup deadline"
            ) from error
        if completed.returncode != 0:
            detail = _failure_detail(completed.stdout, completed.stderr)
            raise RuntimeError(
                f"rank {plan.rank} application failed with exit code "
                f"{completed.returncode}: {detail}"
            )
        return RankApplicationResult(
            rank=plan.rank,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    if cancellation.is_set():
        raise RuntimeError(f"rank {plan.rank} application cancelled before launch")

    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_seconds + 10
    stdout = ""
    stderr = ""
    while True:
        if cancellation.is_set():
            _terminate_process_group(process)
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"rank {plan.rank} application cancelled because its peer failed"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            process.communicate()
            raise RuntimeError(
                f"rank {plan.rank} application exceeded its cleanup deadline"
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    if process.returncode != 0:
        detail = _failure_detail(stdout, stderr)
        raise RuntimeError(
            f"rank {plan.rank} application failed with exit code "
            f"{process.returncode}: {detail}"
        )
    return RankApplicationResult(
        rank=plan.rank,
        stdout=stdout,
        stderr=stderr,
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
