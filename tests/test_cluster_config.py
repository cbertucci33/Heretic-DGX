# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest

from heretic.cluster import ClusterConfig, load_cluster_config
from heretic.config import Settings


def write_cluster(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_cluster_config_accepts_exactly_two_distinct_nodes(tmp_path: Path) -> None:
    path = write_cluster(
        tmp_path / "dgx-cluster.toml",
        """
python = "/opt/heretic/venv/bin/python"
workdir = "/opt/heretic"
master_port = 29517
backend = "nccl"
nccl_socket_ifname = "rocep1s0f0,roceP2p1s0f0"

[[nodes]]
host = "dgx-01"
rank_address = "10.10.10.1"

[[nodes]]
host = "dgx-02"
rank_address = "10.10.10.2"
""",
    )

    config = load_cluster_config(path)

    assert isinstance(config, ClusterConfig)
    assert [node.host for node in config.nodes] == ["dgx-01", "dgx-02"]
    assert [node.rank_address for node in config.nodes] == ["10.10.10.1", "10.10.10.2"]
    assert config.master_address == "10.10.10.1"
    assert config.world_size == 2


def test_settings_declares_optional_dgx_cluster_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["heretic"])
    settings = Settings(model="org/model-under-test", cluster="dgx-cluster.toml")

    assert "cluster" in Settings.model_fields
    assert settings.cluster == "dgx-cluster.toml"


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            '[[nodes]]\nhost = "dgx-01"\nrank_address = "10.10.10.1"\n',
            "exactly two nodes",
        ),
        (
            '[[nodes]]\nhost = "dgx-01"\nrank_address = "10.10.10.1"\n'
            '[[nodes]]\nhost = "dgx-01"\nrank_address = "10.10.10.2"\n',
            "distinct SSH hosts",
        ),
        (
            '[[nodes]]\nhost = "dgx-01"\nrank_address = "10.10.10.1"\n'
            '[[nodes]]\nhost = "dgx-02"\nrank_address = "10.10.10.1"\n',
            "distinct rank addresses",
        ),
    ],
)
def test_load_cluster_config_rejects_ambiguous_topology(
    tmp_path: Path,
    nodes: str,
    message: str,
) -> None:
    path = write_cluster(
        tmp_path / "invalid.toml",
        'python = "/venv/bin/python"\n'
        'workdir = "/work"\n'
        'backend = "nccl"\n'
        + nodes,
    )

    with pytest.raises(ValueError, match=message):
        load_cluster_config(path)
