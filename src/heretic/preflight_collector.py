# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .launch_plan import RankLaunchPlan
from .rank_preflight import (
    RankPreflightIdentity,
    parse_rank_preflight_identity,
    require_matching_rank_preflights,
)


def collect_rank_preflights(
    plans: tuple[RankLaunchPlan, RankLaunchPlan],
    checkpoint_directory: str | Path,
    *,
    timeout_seconds: int,
) -> RankPreflightIdentity:
    """Collect exactly two bounded SSH preflights and require their agreement."""

    if tuple(plan.rank for plan in plans) != (0, 1):
        raise ValueError("rank launch plans must be ordered as rank 0 then rank 1")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("preflight timeout must be a positive integer")

    identities: list[RankPreflightIdentity] = []
    for plan in plans:
        remote_argv = (
            "env",
            *(f"{name}={value}" for name, value in plan.environment),
            plan.argv[0],
            "-m",
            "heretic.rank_preflight",
            "--workdir",
            plan.workdir,
            str(checkpoint_directory),
        )
        command = remote_argv
        if plan.rank == 1:
            command = (
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                plan.host,
                shlex.join(remote_argv),
            )
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"rank {plan.rank} preflight timed out") from error
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:]
            raise RuntimeError(
                f"rank {plan.rank} preflight failed with exit code "
                f"{result.returncode}: {detail}"
            )
        identity = parse_rank_preflight_identity(result.stdout.strip())
        if identity.rank != plan.rank:
            raise RuntimeError(
                f"rank {plan.rank} preflight reported rank {identity.rank}"
            )
        identities.append(identity)

    return require_matching_rank_preflights(identities[0], identities[1])
