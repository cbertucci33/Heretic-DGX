# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True, slots=True)
class RankEnvironment:
    """Validated process environment for one rank in the fixed topology."""

    rank: int
    local_rank: int
    world_size: int
    role: Literal["coordinator", "worker"]
    master_address: str
    master_port: int
    backend: Literal["nccl"]
    timeout_seconds: int
    nccl_socket_ifname: str | None


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"rank environment variable {name} must be a nonempty string")
    if "\x00" in value:
        raise ValueError(f"rank environment variable {name} must not contain NUL")
    return value


def _integer(values: Mapping[str, str], name: str) -> int:
    raw = _required(values, name)
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(
            f"rank environment variable {name} must be an integer"
        ) from error


def read_rank_environment(values: Mapping[str, str]) -> RankEnvironment:
    """Validate the complete fixed-topology contract before runtime imports."""

    if _required(values, "HERETIC_DGX_ACTIVE") != "1":
        raise ValueError("HERETIC_DGX_ACTIVE must be 1")

    rank = _integer(values, "RANK")
    local_rank = _integer(values, "LOCAL_RANK")
    world_size = _integer(values, "WORLD_SIZE")
    role = _required(values, "HERETIC_DGX_ROLE")
    master_port = _integer(values, "MASTER_PORT")
    backend = _required(values, "HERETIC_DGX_BACKEND")
    timeout_seconds = _integer(values, "HERETIC_DGX_TIMEOUT_SECONDS")

    if world_size != 2:
        raise ValueError("WORLD_SIZE must be exactly 2")
    if rank not in (0, 1):
        raise ValueError("RANK must be 0 or 1")
    if local_rank != 0:
        raise ValueError("LOCAL_RANK must be 0 for one process per node")
    expected_role = "coordinator" if rank == 0 else "worker"
    if role != expected_role:
        raise ValueError(f"HERETIC_DGX_ROLE must be {expected_role!r} for rank {rank}")
    if not 1 <= master_port <= 65535:
        raise ValueError("MASTER_PORT must be between 1 and 65535")
    if backend != "nccl":
        raise ValueError("HERETIC_DGX_BACKEND must be 'nccl'")
    if timeout_seconds <= 0:
        raise ValueError("HERETIC_DGX_TIMEOUT_SECONDS must be positive")

    nccl_socket_ifname = values.get("NCCL_SOCKET_IFNAME")
    if nccl_socket_ifname is not None and (
        type(nccl_socket_ifname) is not str or not nccl_socket_ifname.strip()
    ):
        raise ValueError("NCCL_SOCKET_IFNAME must be a nonempty string when set")

    master_address = _required(values, "MASTER_ADDR")
    if "@" in master_address or "://" in master_address:
        raise ValueError("MASTER_ADDR must not contain credentials or a URL scheme")

    return RankEnvironment(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        role=expected_role,
        master_address=master_address,
        master_port=master_port,
        backend="nccl",
        timeout_seconds=timeout_seconds,
        nccl_socket_ifname=nccl_socket_ifname,
    )
