# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import replace
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from heretic.checkpoint_identity import build_checkpoint_payload_identity
from heretic.cluster import load_cluster_config
from heretic.launch_plan import build_rank_launch_plans
from heretic.preflight_collector import collect_rank_preflights
from heretic.rank_environment import read_rank_environment
from heretic.rank_preflight import (
    RankPreflightIdentity,
    require_matching_rank_preflights,
)
from heretic.source_identity import build_source_identity


def _source(root: Path):
    (root / "src/heretic").mkdir(parents=True, exist_ok=True)
    (root / "src/heretic/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return build_source_identity(
        root,
        python_executable="/srv/heretic/.venv/bin/python",
        python_version="3.12.12",
        package_version="2.0.0",
    )


def _checkpoint(root: Path):
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return build_checkpoint_payload_identity(root)


class TestDgxCluster(TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_two_node_config_builds_parseable_rank_plans(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            """
python = "/srv/heretic/.venv/bin/python"
workdir = "/srv/heretic"
master_port = 29517
backend = "nccl"
nccl_socket_ifname = "fabric0"

[[nodes]]
host = "gx10-01"
rank_address = "10.10.10.1"

[[nodes]]
host = "gx10-02"
rank_address = "10.10.10.2"
""",
            encoding="utf-8",
        )

        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file),
            ("/models/checkpoint",),
            entry_module="heretic.rank_entry",
            seed=7,
            host_environment={"HF_HOME": "/cache/hf", "HF_TOKEN": "do-not-forward"},
        )
        environments = [
            read_rank_environment(plan.environment_dict()) for plan in plans
        ]

        self.assertEqual(
            [(plan.host, plan.rank) for plan in plans],
            [("gx10-01", 0), ("gx10-02", 1)],
        )
        self.assertEqual(
            [(environment.rank, environment.role) for environment in environments],
            [(0, "coordinator"), (1, "worker")],
        )
        self.assertTrue(
            all("HF_TOKEN" not in plan.environment_dict() for plan in plans)
        )

    def test_identities_change_with_source_and_checkpoint_bytes(self) -> None:
        source_root = self.root / "source"
        checkpoint_root = self.root / "checkpoint"
        source_before = _source(source_root)
        checkpoint_before = _checkpoint(checkpoint_root)

        (source_root / "src/heretic/example.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        (checkpoint_root / "model.safetensors").write_bytes(b"changed")

        source_after = build_source_identity(
            source_root,
            python_executable="/srv/heretic/.venv/bin/python",
            python_version="3.12.12",
            package_version="2.0.0",
        )
        self.assertNotEqual(source_before.source_sha256, source_after.source_sha256)
        self.assertNotEqual(
            checkpoint_before.digest,
            build_checkpoint_payload_identity(checkpoint_root).digest,
        )

    def test_rank_environment_rejects_unsafe_master_address(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plan = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )[0]
        environment = plan.environment_dict()
        environment["MASTER_ADDR"] = "ssh://10.0.0.1"

        with self.assertRaisesRegex(ValueError, "MASTER_ADDR"):
            read_rank_environment(environment)

    def test_rank_preflight_rejects_checkpoint_mismatch(self) -> None:
        source = _source(self.root / "source")
        checkpoint = _checkpoint(self.root / "checkpoint")
        rank_zero = RankPreflightIdentity(rank=0, source=source, checkpoint=checkpoint)
        rank_one = RankPreflightIdentity(rank=1, source=source, checkpoint=checkpoint)

        self.assertIs(require_matching_rank_preflights(rank_zero, rank_one), rank_zero)

        mismatched = replace(
            rank_one,
            checkpoint=replace(rank_one.checkpoint, digest="f" * 64),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint-payload"):
            require_matching_rank_preflights(rank_zero, mismatched)

    def test_coordinator_collects_matching_rank_preflights(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )
        source = _source(self.root / "source")
        checkpoint = _checkpoint(self.root / "checkpoint")
        outputs = [
            RankPreflightIdentity(rank=rank, source=source, checkpoint=checkpoint)
            for rank in (0, 1)
        ]
        completed = [
            subprocess.CompletedProcess((), 0, identity.canonical_json(), "")
            for identity in outputs
        ]

        with patch("heretic.preflight_collector.subprocess.run", side_effect=completed):
            result = collect_rank_preflights(
                plans, "/models/checkpoint", timeout_seconds=30
            )

        self.assertEqual(result, outputs[0])
