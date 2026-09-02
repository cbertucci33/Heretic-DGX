# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from heretic import tp_capabilities


def test_rowwise_pre_matches_unsharded_oracle() -> None:
    assert hasattr(tp_capabilities, "directional_lora_factors"), (
        "distributed topology math is not implemented"
    )
    directional_lora_factors = tp_capabilities.directional_lora_factors
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 1.0, 0.5, 3.0, 1.5, 2.5],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            [3.0, 2.0, 1.0, 0.5, 1.5, 2.5],
        ]
    )
    direction = torch.nn.functional.normalize(
        torch.tensor([0.5, -1.0, 1.5, 0.25]), dim=0
    )
    strength = 0.7

    row_norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
    normalized = torch.nn.functional.normalize(weight, p=2, dim=1)
    expected_a = (direction @ normalized).view(1, -1)
    expected_b = row_norms * (-strength * direction).view(-1, 1)

    split = weight.shape[1] // 2
    local_weights = (weight[:, :split], weight[:, split:])
    local_factors = []
    for rank, local_weight in enumerate(local_weights):
        peer = local_weights[1 - rank]

        def sum_across_ranks(local: torch.Tensor, peer: torch.Tensor = peer) -> torch.Tensor:
            return local + torch.sum(peer.float().square(), dim=1, keepdim=True)

        local_factors.append(
            directional_lora_factors(
                local_weight,
                direction,
                strength=strength,
                normalization="pre",
                topology="rowwise",
                sum_across_ranks=sum_across_ranks,
            )
        )

    actual_a = torch.cat([factors.a for factors in local_factors], dim=1)
    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(local_factors[0].b, expected_b)
    torch.testing.assert_close(local_factors[1].b, expected_b)
    torch.testing.assert_close(local_factors[0].b @ actual_a, expected_b @ expected_a)


def test_colwise_pre_matches_unsharded_oracle() -> None:
    directional_lora_factors = tp_capabilities.directional_lora_factors
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.5, 3.0],
            [0.5, 1.5, 2.5, 3.5],
            [3.0, 2.0, 1.0, 0.5],
            [1.5, 0.5, 2.0, 4.0],
            [2.5, 3.5, 1.0, 1.5],
        ]
    )
    direction = torch.nn.functional.normalize(
        torch.tensor([0.5, -1.0, 1.5, 0.25, -0.75, 1.25]), dim=0
    )
    strength = 0.6

    row_norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
    normalized = torch.nn.functional.normalize(weight, p=2, dim=1)
    expected_a = (direction @ normalized).view(1, -1)
    expected_b = row_norms * (-strength * direction).view(-1, 1)

    split = weight.shape[0] // 2
    local_weights = (weight[:split], weight[split:])
    local_directions = (direction[:split], direction[split:])
    local_factors = []
    for rank, (local_weight, local_direction) in enumerate(
        zip(local_weights, local_directions, strict=True)
    ):
        peer_weight = local_weights[1 - rank]
        peer_direction = local_directions[1 - rank]
        peer_norm = torch.nn.functional.normalize(peer_weight, p=2, dim=1)
        peer_contribution = (peer_direction @ peer_norm).view(1, -1)

        def sum_across_ranks(
            local: torch.Tensor, peer: torch.Tensor = peer_contribution
        ) -> torch.Tensor:
            return local + peer

        local_factors.append(
            directional_lora_factors(
                local_weight,
                local_direction,
                strength=strength,
                normalization="pre",
                topology="colwise",
                sum_across_ranks=sum_across_ranks,
            )
        )

    torch.testing.assert_close(local_factors[0].a, expected_a)
    torch.testing.assert_close(local_factors[1].a, expected_a)
    actual_b = torch.cat([factors.b for factors in local_factors], dim=0)
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_b @ expected_a, expected_b @ expected_a)


def test_rowwise_none_matches_unsharded_oracle_without_collective() -> None:
    directional_lora_factors = tp_capabilities.directional_lora_factors
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.5, 3.0], [0.5, 1.5, 2.5, 3.5]]
    )
    direction = torch.nn.functional.normalize(torch.tensor([0.5, -1.0, 1.5]), dim=0)
    strength = 0.4
    expected_a = (direction @ weight).view(1, -1)
    expected_b = (-strength * direction).view(-1, 1)

    factors = []
    for local_weight in (weight[:, :2], weight[:, 2:]):
        factors.append(
            directional_lora_factors(
                local_weight,
                direction,
                strength=strength,
                normalization="none",
                topology="rowwise",
                sum_across_ranks=lambda _: (_ for _ in ()).throw(
                    AssertionError("NONE rowwise must not communicate")
                ),
            )
        )

    actual_a = torch.cat([item.a for item in factors], dim=1)
    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(factors[0].b, expected_b)
    torch.testing.assert_close(factors[1].b, expected_b)
