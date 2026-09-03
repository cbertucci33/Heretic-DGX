# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from .application_runner import run_rank_application_plan
from .collective_probe import (
    CollectiveProbeResult,
    parse_collective_probe_result,
)
from .launch_plan import RankLaunchPlan
from .preflight_collector import collect_rank_preflights
from .rank_preflight import RankPreflightIdentity


def _run_probe_plan(
    plan: RankLaunchPlan, *, timeout_seconds: int
) -> CollectiveProbeResult:
    environment = dict(plan.environment)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    if socket_ifname := environment.get("NCCL_SOCKET_IFNAME"):
        environment["GLOO_SOCKET_IFNAME"] = socket_ifname
    probe_plan = replace(
        plan,
        argv=(
            plan.argv[0],
            "-m",
            "heretic.collective_probe",
        ),
        environment=tuple(sorted(environment.items())),
    )
    completed = run_rank_application_plan(
        probe_plan,
        timeout_seconds=timeout_seconds,
    )
    result = parse_collective_probe_result(completed.stdout.strip())
    if result.rank != plan.rank:
        raise RuntimeError(
            f"rank {plan.rank} collective probe reported rank {result.rank}"
        )
    return result


def _collect_rank_collective_probes(
    plans: tuple[RankLaunchPlan, RankLaunchPlan],
    *,
    timeout_seconds: int,
) -> tuple[CollectiveProbeResult, CollectiveProbeResult]:
    """Run two CPU-only ranks concurrently and require a successful all-reduce."""

    if tuple(plan.rank for plan in plans) != (0, 1):
        raise ValueError("rank launch plans must be ordered as rank 0 then rank 1")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("collective probe timeout must be a positive integer")

    results: dict[int, CollectiveProbeResult] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _run_probe_plan, plan, timeout_seconds=timeout_seconds
            ): plan.rank
            for plan in plans
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.rank] = result

    ordered = results[0], results[1]
    expected = [
        CollectiveProbeResult(
            rank=rank,
            world_size=2,
            backend="gloo",
            reduced_value=3,
        )
        for rank in (0, 1)
    ]
    if list(ordered) != expected:
        raise RuntimeError("rank collective probe results do not agree")
    return ordered


def preflight_and_collect_rank_collective_probes(
    plans: tuple[RankLaunchPlan, RankLaunchPlan],
    checkpoint_directory: str,
    *,
    timeout_seconds: int,
) -> tuple[
    RankPreflightIdentity,
    tuple[CollectiveProbeResult, CollectiveProbeResult],
]:
    """Require matching rank identities before starting the collective probe."""

    identity = collect_rank_preflights(
        plans,
        checkpoint_directory,
        timeout_seconds=timeout_seconds,
    )
    results = _collect_rank_collective_probes(
        plans,
        timeout_seconds=timeout_seconds,
    )
    return identity, results
