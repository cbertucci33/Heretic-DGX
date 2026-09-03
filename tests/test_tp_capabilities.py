# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace
from unittest import TestCase

import torch
from torch.distributed.tensor import Replicate, Shard

from heretic.tp_capabilities import (
    directional_lora_factors,
    inspect_lora_target_topologies,
)


def _dtensor_target(*placements: object, mesh_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        weight=SimpleNamespace(
            placements=placements,
            device_mesh=SimpleNamespace(size=lambda: mesh_size),
        )
    )


class TensorParallelCapabilityTests(TestCase):
    def test_accepts_matching_rowwise_and_colwise_dtensor_layouts(self) -> None:
        topologies = inspect_lora_target_topologies(
            {
                "model.layers.0.self_attn.o_proj": _dtensor_target(Shard(1)),
                "model.layers.0.mlp.up_proj": _dtensor_target(Shard(0)),
                "model.shared": _dtensor_target(Replicate()),
            },
            model_tp_plan={
                "model.layers.*.self_attn.o_proj": "rowwise",
                "model.layers.*.mlp.up_proj": "colwise",
            },
        )
        self.assertEqual(
            topologies,
            {
                "model.layers.0.self_attn.o_proj": "rowwise",
                "model.layers.0.mlp.up_proj": "colwise",
                "model.shared": "replicated",
            },
        )

    def test_rejects_missing_or_disagreeing_tp_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing its tensor-parallel plan"):
            inspect_lora_target_topologies(
                {"model.layers.0.proj": _dtensor_target(Shard(1))}
            )
        with self.assertRaisesRegex(ValueError, "disagrees with DTensor placement"):
            inspect_lora_target_topologies(
                {"model.layers.0.proj": _dtensor_target(Shard(1))},
                model_tp_plan={"model.layers.*.proj": "colwise"},
            )

    def test_rowwise_pre_factors_match_unsharded_oracle(self) -> None:
        weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.5, 3.0], [0.5, 1.5, 2.5, 3.5]]
        )
        direction = torch.nn.functional.normalize(
            torch.tensor([0.5, -1.0, 1.5]), dim=0
        )
        local_weights = (weight[:, :2], weight[:, 2:])
        factors = []
        for rank, local_weight in enumerate(local_weights):
            peer = local_weights[1 - rank]
            factors.append(
                directional_lora_factors(
                    local_weight,
                    direction,
                    strength=0.4,
                    normalization="pre",
                    topology="rowwise",
                    sum_across_ranks=lambda local, peer=peer: local
                    + torch.sum(peer.float().square(), dim=1, keepdim=True),
                )
            )

        expected_norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
        expected_a = (
            direction @ torch.nn.functional.normalize(weight, p=2, dim=1)
        ).view(1, -1)
        expected_b = expected_norms * (-0.4 * direction).view(-1, 1)
        torch.testing.assert_close(
            torch.cat([item.a for item in factors], dim=1), expected_a
        )
        torch.testing.assert_close(factors[0].b, expected_b)
        torch.testing.assert_close(factors[1].b, expected_b)

    def test_colwise_none_factors_match_unsharded_oracle(self) -> None:
        weight = torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [3.0, 2.0]]
        )
        direction = torch.nn.functional.normalize(
            torch.tensor([0.5, -1.0, 1.5, 0.25]), dim=0
        )
        local_weights = (weight[:2], weight[2:])
        local_directions = (direction[:2], direction[2:])
        factors = []
        for rank, (local_weight, local_direction) in enumerate(
            zip(local_weights, local_directions, strict=True)
        ):
            peer_rank = 1 - rank
            peer_a = (
                local_directions[peer_rank] @ local_weights[peer_rank]
            ).view(1, -1)
            factors.append(
                directional_lora_factors(
                    local_weight,
                    local_direction,
                    strength=0.6,
                    normalization="none",
                    topology="colwise",
                    sum_across_ranks=lambda local, peer_a=peer_a: local + peer_a,
                )
            )

        expected_a = (direction @ weight).view(1, -1)
        expected_b = (-0.6 * direction).view(-1, 1)
        torch.testing.assert_close(factors[0].a, expected_a)
        torch.testing.assert_close(factors[1].a, expected_a)
        torch.testing.assert_close(
            torch.cat([item.b for item in factors], dim=0), expected_b
        )
