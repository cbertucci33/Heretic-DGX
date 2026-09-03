# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import shlex
import subprocess

from .launch_plan import RankLaunchPlan
from .preflight_collector import collect_rank_preflights
from .rank_preflight import RankPreflightIdentity


@dataclass(frozen=True, slots=True)
class RankApplicationResult:
    """Captured output from one successfully completed rank application."""

    rank: int
    stdout: str
    stderr: str


def run_rank_application_plan(
    plan: RankLaunchPlan, *, timeout_seconds: int
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
        detail = (completed.stderr.strip() or completed.stdout.strip())[-2000:]
        raise RuntimeError(
            f"rank {plan.rank} application failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    return RankApplicationResult(
        rank=plan.rank,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


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
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                run_rank_application_plan,
                plan,
                timeout_seconds=timeout_seconds,
            ): plan.rank
            for plan in plans
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.rank] = result
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
