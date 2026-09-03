# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from heretic.cluster import ClusterConfig, ClusterNode
from heretic.launch_plan import RankLaunchPlan, build_rank_launch_plans


def _config(**overrides: object) -> ClusterConfig:
    values = {
        "nodes": (
            ClusterNode(host="gx10-01", rank_address="10.10.10.1"),
            ClusterNode(host="gx10-02", rank_address="10.10.10.2"),
        ),
        "python": "/srv/heretic/.venv/bin/python",
        "workdir": "/srv/heretic",
        "master_port": 29517,
        "nccl_socket_ifname": "fabric0",
    }
    values.update(overrides)
    return ClusterConfig(**values)


def test_builds_one_immutable_plan_for_each_rank() -> None:
    plans = build_rank_launch_plans(
        _config(),
        ("--cluster", "cluster.toml", "/models/checkpoint"),
        entry_module="heretic.rank_entry",
        seed=123456789,
    )

    assert all(isinstance(plan, RankLaunchPlan) for plan in plans)
    assert [(plan.rank, plan.role, plan.host) for plan in plans] == [
        (0, "coordinator", "gx10-01"),
        (1, "worker", "gx10-02"),
    ]
    assert plans[0].argv == plans[1].argv == (
        "/srv/heretic/.venv/bin/python",
        "-m",
        "heretic.rank_entry",
        "--cluster",
        "cluster.toml",
        "/models/checkpoint",
        "--seed",
        "123456789",
    )
    for rank, plan in enumerate(plans):
        environment = plan.environment_dict()
        assert environment["RANK"] == str(rank)
        assert environment["LOCAL_RANK"] == "0"
        assert environment["WORLD_SIZE"] == "2"
        assert environment["MASTER_ADDR"] == "10.10.10.1"
        assert environment["MASTER_PORT"] == "29517"
        assert environment["HERETIC_DGX_BACKEND"] == "nccl"
        assert environment["NCCL_SOCKET_IFNAME"] == "fabric0"


def test_plan_is_deterministic_and_forwards_only_cache_paths() -> None:
    arguments = {
        "entry_module": "heretic.rank_entry",
        "seed": 7,
        "host_environment": {
            "HF_HOME": "/srv/cache/huggingface",
            "TRITON_CACHE_DIR": "/srv/cache/triton",
            "HF_TOKEN": "must-not-cross-the-rank-boundary",
            "AUTHORIZATION": "must-not-cross-the-rank-boundary",
            "UNRELATED": "must-not-cross-the-rank-boundary",
        },
    }

    first = build_rank_launch_plans(_config(), ("model",), **arguments)
    second = build_rank_launch_plans(_config(), ("model",), **arguments)

    assert first == second
    for plan in first:
        environment = plan.environment_dict()
        assert environment["HF_HOME"] == "/srv/cache/huggingface"
        assert environment["TRITON_CACHE_DIR"] == "/srv/cache/triton"
        assert "HF_TOKEN" not in environment
        assert "AUTHORIZATION" not in environment
        assert "UNRELATED" not in environment


@pytest.mark.parametrize("field", ["python", "workdir"])
def test_rejects_relative_runtime_paths(field: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_rank_launch_plans(
            _config(**{field: "relative/path"}),
            ("model",),
            entry_module="heretic.rank_entry",
            seed=1,
        )


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("model", "--seed", "1"),
        ("model", "--seed=1"),
        ("model", "--seed"),
    ],
)
def test_rejects_ambiguous_rank_arguments(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        build_rank_launch_plans(
            _config(), argv, entry_module="heretic.rank_entry", seed=1
        )


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_rejects_invalid_shared_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_rank_launch_plans(
            _config(),
            ("model",),
            entry_module="heretic.rank_entry",
            seed=seed,
        )
