# SPDX-License-Identifier: AGPL-3.0-or-later

import multiprocessing
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.distributed as dist

from heretic.dgx_runtime import (
    DgxCoordinatorRuntime,
    TorchDistributedCommandChannel,
    run_dgx_worker,
)
from heretic.runtime import ModelRuntime, gather_tensor_parallel_lora_shard
from heretic.utils import Prompt


@dataclass
class _RecordingRuntime(ModelRuntime):
    calls: list[str] = field(default_factory=list)

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    def reset_model(self, model: str | None = None) -> None:
        self.calls.append("reset_model")

    def abliterate(self, residual_directions, direction_index, parameters) -> None:
        self.calls.append("abliterate")

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        self.calls.append("save_adapter")

    def save_merged(self, directory: str, *, max_shard_size: int | str) -> None:
        self.calls.append("save_merged")

    def get_responses_once(
        self,
        prompts,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self.calls.append("get_responses_once")
        return ["response"]

    def get_responses(
        self,
        prompts,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self.calls.append("get_responses")
        return ["response"]

    def get_logits(self, prompts):
        self.calls.append("get_logits")
        return torch.tensor([[1.0, 2.0]])

    def get_residuals(self, prompts):
        self.calls.append("get_residuals")
        return torch.tensor([[[1.0, 2.0]]])

    def get_residuals_mean(self, prompts):
        self.calls.append("get_residuals_mean")
        return torch.tensor([[1.0, 2.0]])


@dataclass
class _CoordinatorChannel:
    peer_error: str | None = None
    sent: list[object] = field(default_factory=list)

    def send(self, command) -> None:
        self.sent.append(command)

    def receive(self):
        raise AssertionError("coordinator must not receive")

    def complete(self, local_error: str | None) -> tuple[str | None, str | None]:
        return local_error, self.peer_error


def _gloo_runtime_entry(rank: int, port: int, results: Any) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        local = _RecordingRuntime()
        channel = TorchDistributedCommandChannel()
        if rank == 0:
            runtime = DgxCoordinatorRuntime(local, channel)
            logits = runtime.get_logits([Prompt(system="system", user="user")])
            runtime.save_merged("model", max_shard_size="5GB")
            runtime.shutdown()
            results.put((rank, logits.tolist(), local.calls))
        else:
            run_dgx_worker(local, channel)
            results.put((rank, None, local.calls))
    finally:
        dist.destroy_process_group()


class DgxRuntimeTests(TestCase):
    def test_gathers_rowwise_lora_a_in_rank_order(self) -> None:
        local = torch.tensor([[1.0, 2.0]])

        def all_gather(outputs, value) -> None:
            outputs[0].copy_(value)
            outputs[1].copy_(value + 2)

        with (
            patch("heretic.runtime.dist.get_world_size", return_value=2),
            patch("heretic.runtime.dist.all_gather", side_effect=all_gather),
        ):
            gathered = gather_tensor_parallel_lora_shard(local, dimension=1)

        torch.testing.assert_close(gathered, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_worker_failure_latches_runtime_and_preserves_first_error(self) -> None:
        local = _RecordingRuntime()
        channel = _CoordinatorChannel(peer_error="rank 1: out of memory")
        runtime = DgxCoordinatorRuntime(local, channel)

        with self.assertRaisesRegex(RuntimeError, "rank 1: out of memory"):
            runtime.reset_model()
        sent_after_failure = list(channel.sent)
        channel.peer_error = None
        with self.assertRaisesRegex(RuntimeError, "has failed"):
            runtime.get_logits([])
        runtime.shutdown()

        self.assertEqual(channel.sent, sent_after_failure)
        self.assertEqual(local.calls, ["reset_model", "shutdown"])

    def test_real_two_process_gloo_command_loop_and_shutdown(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        processes = [
            context.Process(target=_gloo_runtime_entry, args=(rank, port, results))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            if process.is_alive():
                process.kill()
                process.join()
                self.fail("two-process DGX runtime test timed out")
            self.assertEqual(process.exitcode, 0)

        observed = sorted(results.get(timeout=5) for _ in range(2))
        self.assertEqual(
            observed,
            [
                (0, [[1.0, 2.0]], ["get_logits", "save_merged", "shutdown"]),
                (1, None, ["get_logits", "save_merged", "shutdown"]),
            ],
        )
