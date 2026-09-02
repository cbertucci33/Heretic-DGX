# SPDX-License-Identifier: AGPL-3.0-or-later

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from heretic.distributed_partition import (
    CheckpointPartitionPlan,
    plan_checkpoint_shards,
)


class CheckpointPartitionPlanTests(unittest.TestCase):
    def _checkpoint(
        self,
        shard_sizes: dict[str, int],
        weight_map: dict[str, str],
    ) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for shard_name, size in shard_sizes.items():
            (root / shard_name).write_bytes(bytes([size % 251]) * size)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": sum(shard_sizes.values())}, "weight_map": weight_map},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return temporary, root

    def test_assigns_every_shard_and_tensor_once_with_byte_balance(self) -> None:
        temporary, root = self._checkpoint(
            {
                "model-00001-of-00004.safetensors": 9,
                "model-00002-of-00004.safetensors": 7,
                "model-00003-of-00004.safetensors": 5,
                "model-00004-of-00004.safetensors": 3,
            },
            {
                "model.layers.0.weight": "model-00001-of-00004.safetensors",
                "model.layers.1.weight": "model-00002-of-00004.safetensors",
                "model.layers.2.weight": "model-00003-of-00004.safetensors",
                "model.layers.3.weight": "model-00004-of-00004.safetensors",
            },
        )
        self.addCleanup(temporary.cleanup)

        plan = plan_checkpoint_shards(root)

        self.assertIsInstance(plan, CheckpointPartitionPlan)
        self.assertEqual(plan.rank_bytes, (12, 12))
        self.assertEqual(len(plan.shards), 4)
        self.assertEqual(len({shard.name for shard in plan.shards}), 4)
        self.assertEqual(len(plan.tensors), 4)
        self.assertEqual(len({tensor.name for tensor in plan.tensors}), 4)
        self.assertEqual({shard.rank for shard in plan.shards}, {0, 1})
        shard_ranks = {shard.name: shard.rank for shard in plan.shards}
        for tensor in plan.tensors:
            self.assertEqual(tensor.rank, shard_ranks[tensor.shard_name])
        self.assertRegex(plan.digest, r"^[0-9a-f]{64}$")

    def test_is_deterministic_and_does_not_modify_checkpoint_bytes(self) -> None:
        shard_sizes = {
            "model-00001-of-00002.safetensors": 8,
            "model-00002-of-00002.safetensors": 8,
        }
        first_map = {
            "z.weight": "model-00002-of-00002.safetensors",
            "a.weight": "model-00001-of-00002.safetensors",
        }
        temporary, root = self._checkpoint(shard_sizes, first_map)
        self.addCleanup(temporary.cleanup)
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.glob("*.safetensors")
        }

        first = plan_checkpoint_shards(root)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": dict(reversed(list(first_map.items()))), "metadata": {"total_size": 16}},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        second = plan_checkpoint_shards(root)
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.glob("*.safetensors")
        }

        self.assertEqual(first, second)
        self.assertEqual(before, after)

    def test_digest_and_assignments_bind_exact_shard_bytes(self) -> None:
        temporary, root = self._checkpoint(
            {
                "model-00001-of-00002.safetensors": 8,
                "model-00002-of-00002.safetensors": 8,
            },
            {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
            },
        )
        self.addCleanup(temporary.cleanup)

        first = plan_checkpoint_shards(root)
        changed_path = root / "model-00001-of-00002.safetensors"
        changed_path.write_bytes(b"changed!")
        second = plan_checkpoint_shards(root)

        self.assertNotEqual(first.digest, second.digest)
        first_shard = next(
            shard for shard in first.shards if shard.name == changed_path.name
        )
        second_shard = next(
            shard for shard in second.shards if shard.name == changed_path.name
        )
        self.assertNotEqual(first_shard.sha256, second_shard.sha256)
        self.assertEqual(
            second_shard.sha256,
            hashlib.sha256(changed_path.read_bytes()).hexdigest(),
        )

    def test_rejects_incomplete_or_unsafe_checkpoint_indexes(self) -> None:
        cases = {
            "missing shard": {
                "weight_map": {"layer.weight": "missing.safetensors"}
            },
            "path traversal": {
                "weight_map": {"layer.weight": "../outside.safetensors"}
            },
            "empty tensor name": {
                "weight_map": {"": "model-00001-of-00002.safetensors"}
            },
            "empty shard name": {"weight_map": {"layer.weight": ""}},
            "wrong weight map": {"weight_map": []},
        }
        for name, index in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "model-00001-of-00002.safetensors").write_bytes(b"a")
                    (root / "model.safetensors.index.json").write_text(
                        json.dumps(index), encoding="utf-8"
                    )
                    with self.assertRaises((TypeError, ValueError, FileNotFoundError)):
                        plan_checkpoint_shards(root)

    def test_rejects_duplicate_keys_empty_shards_and_single_rank_plans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = "model-00001-of-00002.safetensors"
            second = "model-00002-of-00002.safetensors"
            (root / first).write_bytes(b"a")
            (root / second).write_bytes(b"b")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer.weight":"' + first + '","layer.weight":"' + second + '"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                plan_checkpoint_shards(root)

        temporary, root = self._checkpoint(
            {
                "model-00001-of-00002.safetensors": 1,
                "model-00002-of-00002.safetensors": 0,
            },
            {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            },
        )
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "empty"):
            plan_checkpoint_shards(root)

        temporary, root = self._checkpoint(
            {"model.safetensors": 4},
            {"layer.weight": "model.safetensors"},
        )
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "two nonempty ranks"):
            plan_checkpoint_shards(root)


if __name__ == "__main__":
    unittest.main()
