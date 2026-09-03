# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from heretic.cluster import ClusterConfig, ClusterNode
from heretic.launch_plan import build_rank_launch_plans
from heretic.rank_environment import RankEnvironment, read_rank_environment


def _plans():
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="gx10-01", rank_address="10.10.10.1"),
            ClusterNode(host="gx10-02", rank_address="10.10.10.2"),
        ),
        python="/srv/heretic/.venv/bin/python",
        workdir="/srv/heretic",
        master_port=29517,
        timeout_seconds=300,
        nccl_socket_ifname="fabric0",
    )
    return build_rank_launch_plans(
        config,
        ("model",),
        entry_module="heretic.rank_entry",
        seed=7,
    )


def test_accepts_both_generated_rank_environments() -> None:
    environments = [
        read_rank_environment(plan.environment_dict()) for plan in _plans()
    ]

    assert all(isinstance(environment, RankEnvironment) for environment in environments)
    assert [(environment.rank, environment.role) for environment in environments] == [
        (0, "coordinator"),
        (1, "worker"),
    ]
    assert all(environment.world_size == 2 for environment in environments)
    assert all(environment.local_rank == 0 for environment in environments)
    assert all(environment.master_address == "10.10.10.1" for environment in environments)
    assert all(environment.master_port == 29517 for environment in environments)
    assert all(environment.backend == "nccl" for environment in environments)
    assert all(environment.timeout_seconds == 300 for environment in environments)
    assert all(environment.nccl_socket_ifname == "fabric0" for environment in environments)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HERETIC_DGX_ACTIVE", "0", "ACTIVE"),
        ("WORLD_SIZE", "1", "WORLD_SIZE"),
        ("WORLD_SIZE", "3", "WORLD_SIZE"),
        ("RANK", "2", "RANK"),
        ("LOCAL_RANK", "1", "LOCAL_RANK"),
        ("HERETIC_DGX_ROLE", "worker", "ROLE"),
        ("MASTER_PORT", "0", "MASTER_PORT"),
        ("MASTER_PORT", "65536", "MASTER_PORT"),
        ("HERETIC_DGX_BACKEND", "gloo", "BACKEND"),
        ("HERETIC_DGX_TIMEOUT_SECONDS", "0", "TIMEOUT"),
        ("NCCL_SOCKET_IFNAME", "", "NCCL_SOCKET_IFNAME"),
    ],
)
def test_rejects_invalid_fixed_topology_values(
    name: str, value: str, message: str
) -> None:
    values = _plans()[0].environment_dict()
    values[name] = value

    with pytest.raises(ValueError, match=message):
        read_rank_environment(values)


@pytest.mark.parametrize(
    "name",
    [
        "HERETIC_DGX_ACTIVE",
        "HERETIC_DGX_ROLE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "HERETIC_DGX_BACKEND",
        "HERETIC_DGX_TIMEOUT_SECONDS",
    ],
)
def test_rejects_missing_required_values(name: str) -> None:
    values = _plans()[1].environment_dict()
    del values[name]

    with pytest.raises(ValueError, match=name):
        read_rank_environment(values)


def test_rejects_rank_role_mismatch() -> None:
    values = _plans()[1].environment_dict()
    values["HERETIC_DGX_ROLE"] = "coordinator"

    with pytest.raises(ValueError, match="worker"):
        read_rank_environment(values)


@pytest.mark.parametrize("value", ["user@example.invalid", "ssh://10.10.10.1"])
def test_rejects_credentials_or_url_in_master_address(value: str) -> None:
    values = _plans()[0].environment_dict()
    values["MASTER_ADDR"] = value

    with pytest.raises(ValueError, match="MASTER_ADDR"):
        read_rank_environment(values)
