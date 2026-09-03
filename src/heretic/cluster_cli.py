# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import random
import sys
from collections.abc import Mapping, Sequence

from .application_runner import (
    RankApplicationResult,
    preflight_and_collect_rank_applications,
)
from .cluster import load_cluster_config
from .config import Settings
from .launch_plan import build_rank_launch_plans


def has_cluster_option(argv: Sequence[str]) -> bool:
    return any(
        argument == "--cluster" or argument.startswith("--cluster=")
        for argument in argv
    )


def _without_coordinator_options(argv: Sequence[str]) -> tuple[str, ...]:
    retained: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--cluster", "--seed"):
            if index + 1 >= len(argv):
                raise ValueError(f"{argument} requires a value")
            index += 2
            continue
        if argument.startswith("--cluster=") or argument.startswith("--seed="):
            index += 1
            continue
        retained.append(argument)
        index += 1
    return tuple(retained)


def launch_cluster_application(
    settings: Settings,
    argv: Sequence[str],
    *,
    host_environment: Mapping[str, str] | None = None,
) -> tuple[RankApplicationResult, RankApplicationResult]:
    """Build, preflight, and launch the fixed two-rank application once."""

    if settings.cluster is None:
        raise ValueError("cluster application requires --cluster")
    if getattr(settings, "collect_reproducibles", None) is not None:
        raise ValueError("--collect-reproducibles is a coordinator-local operation")
    if getattr(settings, "reproduce", None) is not None:
        raise ValueError("--reproduce is not yet supported with --cluster")
    if settings.seed is None:
        settings.seed = random.randint(0, 2**32 - 1)

    config = load_cluster_config(settings.cluster)
    plans = build_rank_launch_plans(
        config,
        _without_coordinator_options(argv),
        entry_module="heretic.rank_application",
        seed=settings.seed,
        host_environment=host_environment,
    )
    _, results = preflight_and_collect_rank_applications(
        plans,
        settings.model,
        timeout_seconds=config.timeout_seconds,
    )
    return results


def run_cluster_cli(argv: Sequence[str]) -> int:
    settings = Settings()  # ty:ignore[missing-argument]
    results = launch_cluster_application(
        settings,
        argv,
        host_environment=os.environ,
    )
    sys.stdout.write(results[0].stdout)
    sys.stderr.write(results[0].stderr)
    sys.stderr.write(results[1].stderr)
    return 0
