# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import torch.distributed as dist

if TYPE_CHECKING:
    from .config import Settings


DgxRole = Literal["coordinator", "worker"]
DgxBackend = Literal["nccl", "gloo"]


@dataclass(frozen=True, slots=True)
class DgxRankEnvironment:
    role: DgxRole
    master_address: str
    master_port: int
    world_size: int
    rank: int
    local_rank: int
    backend: DgxBackend
    timeout_seconds: int


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if value is None or not value.strip():
        raise ValueError(f"missing DGX environment field: {name}")
    return value


def _integer(values: Mapping[str, str], name: str) -> int:
    raw = _required(values, name)
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"DGX environment field {name} must be an integer") from error


def read_dgx_rank_environment(values: Mapping[str, str]) -> DgxRankEnvironment:
    if values.get("HERETIC_DGX_ACTIVE") != "1":
        raise ValueError("DGX rank entry is not active; set HERETIC_DGX_ACTIVE=1")

    role = _required(values, "HERETIC_DGX_ROLE")
    if role not in {"coordinator", "worker"}:
        raise ValueError("DGX role must be coordinator or worker")

    world_size = _integer(values, "WORLD_SIZE")
    if world_size != 2:
        raise ValueError("DGX execution requires exactly two ranks")

    rank = _integer(values, "RANK")
    if rank not in {0, 1}:
        raise ValueError("DGX rank must be 0 or 1")
    expected_role = "coordinator" if rank == 0 else "worker"
    if role != expected_role:
        raise ValueError("DGX role does not match rank")

    local_rank = _integer(values, "LOCAL_RANK")
    if local_rank != 0:
        raise ValueError("DGX execution requires LOCAL_RANK=0 on each node")

    master_address = _required(values, "MASTER_ADDR")
    if "@" in master_address or "://" in master_address:
        raise ValueError("MASTER_ADDR must not contain credentials or a URL scheme")

    master_port = _integer(values, "MASTER_PORT")
    if not 1 <= master_port <= 65535:
        raise ValueError("MASTER_PORT must be between 1 and 65535")

    backend = _required(values, "HERETIC_DGX_BACKEND")
    if backend not in {"nccl", "gloo"}:
        raise ValueError("DGX backend must be nccl or gloo")

    timeout_seconds = _integer(values, "HERETIC_DGX_TIMEOUT_SECONDS")
    if timeout_seconds <= 0:
        raise ValueError("DGX timeout must be positive")

    return DgxRankEnvironment(
        role=role,  # type: ignore[arg-type]
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=backend,  # type: ignore[arg-type]
        timeout_seconds=timeout_seconds,
    )


def build_dgx_rank_identity(
    environment: DgxRankEnvironment,
    *,
    machine_identity: str,
) -> dict[str, str | int]:
    """Build privacy-safe structured evidence for one initialized DGX rank."""

    fingerprint = hashlib.sha256(
        f"heretic-dgx-host:{machine_identity}".encode()
    ).hexdigest()[:16]
    return {
        "event": "dgx_rank_initialized",
        "rank": environment.rank,
        "world_size": environment.world_size,
        "backend": environment.backend,
        "host_fingerprint": fingerprint,
    }


def synchronize_dgx_settings(
    settings: Settings | None,
    *,
    rank: int,
) -> Settings | None:
    """Broadcast rank 0's finalized settings before collective model loading."""

    if rank not in {0, 1}:
        raise ValueError("DGX settings synchronization requires rank 0 or 1")
    if rank == 1 and settings is not None:
        raise ValueError("DGX worker settings must come from rank 0")

    payload: list[object] = [
        settings.model_dump_json() if rank == 0 and settings is not None else None
    ]
    dist.broadcast_object_list(payload, src=0)
    serialized = payload[0]
    if serialized is None:
        return None
    if type(serialized) is not str:
        raise RuntimeError("DGX settings payload must be serialized JSON or null")

    from .config import Settings

    return Settings.model_validate_json(serialized)


def main() -> None:
    environment = read_dgx_rank_environment(os.environ)

    import torch
    import torch.distributed as dist

    if environment.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("DGX NCCL execution requires CUDA")
        torch.cuda.set_device(environment.local_rank)

    dist.init_process_group(
        backend=environment.backend,
        init_method="env://",
        rank=environment.rank,
        world_size=environment.world_size,
        timeout=timedelta(seconds=environment.timeout_seconds),
    )
    print(
        json.dumps(
            build_dgx_rank_identity(
                environment,
                machine_identity=socket.gethostname(),
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        from .dgx_runtime import (
            DgxCoordinatorRuntime,
            TorchDistributedCommandChannel,
            run_dgx_worker,
        )
        from .main import run
        from .runtime import LocalModelRuntime

        channel = TorchDistributedCommandChannel()
        if environment.role == "coordinator":
            coordinator_runtime: DgxCoordinatorRuntime | None = None

            def runtime_factory(model):
                nonlocal coordinator_runtime
                coordinator_runtime = DgxCoordinatorRuntime(
                    LocalModelRuntime(model), channel
                )
                return coordinator_runtime

            try:
                run(
                    runtime_factory=runtime_factory,
                    settings_synchronizer=lambda settings: synchronize_dgx_settings(
                        settings,
                        rank=environment.rank,
                    ),
                )
            finally:
                if coordinator_runtime is not None:
                    coordinator_runtime.shutdown()
        else:
            run(
                worker_runner=lambda model: run_dgx_worker(
                    LocalModelRuntime(model), channel
                ),
                settings_synchronizer=lambda settings: synchronize_dgx_settings(
                    settings,
                    rank=environment.rank,
                ),
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
