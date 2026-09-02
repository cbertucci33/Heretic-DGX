# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomli


@dataclass(frozen=True, slots=True)
class ClusterNode:
    """One physical DGX node participating in the distributed runtime."""

    host: str
    rank_address: str


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Validated configuration for Heretic's two-node DGX runtime."""

    nodes: tuple[ClusterNode, ClusterNode]
    python: str
    workdir: str
    backend: str = "nccl"
    master_port: int = 29500
    timeout_seconds: int = 900
    nccl_socket_ifname: str | None = None

    @property
    def master_address(self) -> str:
        return self.nodes[0].rank_address

    @property
    def world_size(self) -> int:
        return len(self.nodes)


def _required_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DGX cluster field {key!r} must be a non-empty string")
    return value


def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Load a fail-closed two-node DGX cluster configuration."""

    cluster_path = Path(path)
    with cluster_path.open("rb") as file:
        data = tomli.load(file)

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != 2:
        raise ValueError("DGX cluster must define exactly two nodes")

    nodes: list[ClusterNode] = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise ValueError(f"DGX cluster node {index} must be a TOML table")
        nodes.append(
            ClusterNode(
                host=_required_string(raw_node, "host"),
                rank_address=_required_string(raw_node, "rank_address"),
            )
        )

    if len({node.host for node in nodes}) != 2:
        raise ValueError("DGX cluster must use two distinct SSH hosts")
    if len({node.rank_address for node in nodes}) != 2:
        raise ValueError("DGX cluster must use two distinct rank addresses")

    backend = data.get("backend", "nccl")
    if backend != "nccl":
        raise ValueError("DGX runtime backend must be 'nccl'")

    master_port = data.get("master_port", 29500)
    if not isinstance(master_port, int) or isinstance(master_port, bool):
        raise ValueError("DGX cluster field 'master_port' must be an integer")
    if not 1 <= master_port <= 65535:
        raise ValueError("DGX cluster field 'master_port' must be between 1 and 65535")

    timeout_seconds = data.get("timeout_seconds", 900)
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise ValueError("DGX cluster field 'timeout_seconds' must be an integer")
    if timeout_seconds <= 0:
        raise ValueError("DGX cluster field 'timeout_seconds' must be positive")

    nccl_socket_ifname = data.get("nccl_socket_ifname")
    if nccl_socket_ifname is not None and (
        not isinstance(nccl_socket_ifname, str) or not nccl_socket_ifname.strip()
    ):
        raise ValueError(
            "DGX cluster field 'nccl_socket_ifname' must be a non-empty string"
        )

    return ClusterConfig(
        nodes=(nodes[0], nodes[1]),
        python=_required_string(data, "python"),
        workdir=_required_string(data, "workdir"),
        backend=backend,
        master_port=master_port,
        timeout_seconds=timeout_seconds,
        nccl_socket_ifname=nccl_socket_ifname,
    )
