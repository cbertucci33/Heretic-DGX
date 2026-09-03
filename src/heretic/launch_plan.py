# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .cluster import ClusterConfig

_SAFE_FORWARDED_ENVIRONMENT = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
    }
)
_MODULE_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


@dataclass(frozen=True, slots=True)
class RankLaunchPlan:
    """Immutable process inputs for one rank; constructing a plan starts nothing."""

    rank: int
    role: Literal["coordinator", "worker"]
    host: str
    argv: tuple[str, ...]
    workdir: str
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


def _validate_application_argv(application_argv: tuple[str, ...]) -> None:
    if not application_argv:
        raise ValueError("rank application argv must not be empty")
    for argument in application_argv:
        if type(argument) is not str or "\x00" in argument:
            raise ValueError("rank application arguments must be strings without NUL")
    for index, argument in enumerate(application_argv):
        if argument == "--seed":
            if index + 1 >= len(application_argv):
                raise ValueError("--seed requires a value")
            raise ValueError("rank application argv must not provide its own seed")
        if argument.startswith("--seed="):
            raise ValueError("rank application argv must not provide its own seed")


def build_rank_launch_plans(
    config: ClusterConfig,
    application_argv: tuple[str, ...],
    *,
    entry_module: str,
    seed: int,
    host_environment: Mapping[str, str] | None = None,
) -> tuple[RankLaunchPlan, RankLaunchPlan]:
    """Build exact two-rank process inputs without executing either process."""

    if _MODULE_PATTERN.fullmatch(entry_module) is None:
        raise ValueError("rank entry module must be a dotted Python module name")
    if type(seed) is not int or seed < 0:
        raise ValueError("rank seed must be a nonnegative integer")
    _validate_application_argv(application_argv)

    python = Path(config.python)
    workdir = Path(config.workdir)
    if not python.is_absolute():
        raise ValueError("DGX cluster python path must be absolute")
    if not workdir.is_absolute():
        raise ValueError("DGX cluster workdir must be absolute")

    forwarded: dict[str, str] = {}
    for name, value in (host_environment or {}).items():
        if name in _SAFE_FORWARDED_ENVIRONMENT:
            if type(value) is not str or "\x00" in value:
                raise ValueError(f"forwarded environment value is invalid: {name}")
            forwarded[name] = value

    command = (
        str(python),
        "-m",
        entry_module,
        *application_argv,
        "--seed",
        str(seed),
    )
    plans: list[RankLaunchPlan] = []
    for rank, node in enumerate(config.nodes):
        role: Literal["coordinator", "worker"] = (
            "coordinator" if rank == 0 else "worker"
        )
        environment = {
            **forwarded,
            "HERETIC_DGX_ACTIVE": "1",
            "HERETIC_DGX_BACKEND": config.backend,
            "HERETIC_DGX_ROLE": role,
            "HERETIC_DGX_TIMEOUT_SECONDS": str(config.timeout_seconds),
            "HF_DEACTIVATE_ASYNC_LOAD": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": config.master_address,
            "MASTER_PORT": str(config.master_port),
            "PYTHONPATH": str(workdir / "src"),
            "RANK": str(rank),
            "WORLD_SIZE": str(config.world_size),
        }
        if config.nccl_socket_ifname is not None:
            environment["NCCL_SOCKET_IFNAME"] = config.nccl_socket_ifname
        plans.append(
            RankLaunchPlan(
                rank=rank,
                role=role,
                host=node.host,
                argv=command,
                workdir=str(workdir),
                environment=tuple(sorted(environment.items())),
            )
        )
    return plans[0], plans[1]
