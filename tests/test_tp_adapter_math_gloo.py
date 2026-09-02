# SPDX-License-Identifier: AGPL-3.0-or-later

import multiprocessing
import socket
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist

from heretic.tp_capabilities import directional_lora_factors


def _rowwise_pre_entry(rank: int, port: int, results: Any) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        full_weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 1.0, 0.5, 3.0],
                [0.5, 1.5, 2.5, 3.5],
            ]
        )
        direction = torch.nn.functional.normalize(
            torch.tensor([0.5, -1.0, 1.5]), dim=0
        )
        local_weight = full_weight[:, rank * 2 : (rank + 1) * 2]

        def sum_across_ranks(value: torch.Tensor) -> torch.Tensor:
            result = value.clone()
            dist.all_reduce(result)
            return result

        factors = directional_lora_factors(
            local_weight,
            direction,
            strength=0.4,
            normalization="pre",
            topology="rowwise",
            sum_across_ranks=sum_across_ranks,
        )
        results.put((rank, factors.a.tolist(), factors.b.tolist()))
    finally:
        dist.destroy_process_group()


def test_real_two_process_rowwise_pre_matches_unsharded_oracle() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(target=_rowwise_pre_entry, args=(rank, port, results))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("two-process topology-math test timed out")
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=5) for _ in range(2))
    actual_a = torch.cat([torch.tensor(item[1]) for item in observed], dim=1)
    actual_b = torch.tensor(observed[0][2])

    full_weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.5, 3.0],
            [0.5, 1.5, 2.5, 3.5],
        ]
    )
    direction = torch.nn.functional.normalize(torch.tensor([0.5, -1.0, 1.5]), dim=0)
    row_norms = torch.linalg.vector_norm(full_weight, dim=1, keepdim=True)
    normalized = torch.nn.functional.normalize(full_weight, p=2, dim=1)
    expected_a = (direction @ normalized).view(1, -1)
    expected_b = row_norms * (-0.4 * direction).view(-1, 1)
    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(torch.tensor(observed[1][2]), expected_b)
