# SPDX-License-Identifier: AGPL-3.0-or-later

import multiprocessing
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist

from heretic.dgx_runtime import (
    DgxCommand,
    DgxCoordinatorRuntime,
    TorchDistributedCommandChannel,
    run_dgx_worker,
)
from heretic.runtime import ModelMetadata, ModelRuntime
from heretic.utils import Prompt


@dataclass
class RecordingRuntime(ModelRuntime):
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata("local", "org/model", "a" * 40, "torch.bfloat16", "none", 2, ("attn.o_proj",), 0)

    def shutdown(self) -> None:
        self.calls.append(("shutdown", (), {}))

    def reset_model(self, model: str | None = None) -> None:
        self.calls.append(("reset_model", (model,) if model is not None else (), {}))

    def abliterate(self, residual_directions, direction_index, parameters) -> None:
        self.calls.append(("abliterate", (residual_directions, direction_index, parameters), {}))

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        self.calls.append(("save_adapter", (directory,), {"max_shard_size": max_shard_size}))

    def get_responses(self, prompts, *, skip_special_tokens: bool = True) -> list[str]:
        self.calls.append(("get_responses", (prompts,), {"skip_special_tokens": skip_special_tokens}))
        return ["response"]

    def get_responses_once(
        self,
        prompts,
        *,
        skip_special_tokens: bool = True,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        self.calls.append(
            (
                "get_responses_once",
                (prompts,),
                {
                    "skip_special_tokens": skip_special_tokens,
                    "max_new_tokens": max_new_tokens,
                },
            )
        )
        return ["response"]

    def get_logits(self, prompts):
        self.calls.append(("get_logits", (prompts,), {}))
        return torch.tensor([[1.0, 2.0]])

    def get_residuals(self, prompts):
        self.calls.append(("get_residuals", (prompts,), {}))
        return torch.tensor([[[1.0, 2.0]]])

    def get_residuals_mean(self, prompts):
        self.calls.append(("get_residuals_mean", (prompts,), {}))
        return torch.tensor([[1.0, 2.0]])


@dataclass
class RecordingCoordinatorChannel:
    sent: list[DgxCommand] = field(default_factory=list)
    peer_error: str | None = None

    def send(self, command: DgxCommand) -> None:
        self.sent.append(command)

    def receive(self) -> DgxCommand:
        raise AssertionError("coordinator must not receive commands")

    def complete(self, local_error: str | None) -> tuple[str | None, str | None]:
        return local_error, self.peer_error


@dataclass
class ScriptedWorkerChannel:
    commands: list[DgxCommand]
    completions: list[str | None] = field(default_factory=list)

    def send(self, command: DgxCommand) -> None:
        raise AssertionError("worker must not send commands")

    def receive(self) -> DgxCommand:
        return self.commands.pop(0)

    def complete(self, local_error: str | None) -> tuple[str | None, str | None]:
        self.completions.append(local_error)
        return None, local_error


def _gloo_runtime_entry(rank: int, port: int, results: Any) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        local = RecordingRuntime()
        channel = TorchDistributedCommandChannel()
        if rank == 0:
            runtime = DgxCoordinatorRuntime(local, channel)
            logits = runtime.get_logits([Prompt(system="system", user="user")])
            runtime.shutdown()
            results.put((rank, logits.tolist(), [call[0] for call in local.calls]))
        else:
            run_dgx_worker(local, channel)
            results.put((rank, None, [call[0] for call in local.calls]))
    finally:
        dist.destroy_process_group()


def test_coordinator_broadcasts_operation_before_executing_local_runtime() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)
    prompts = [Prompt(system="system", user="user")]

    responses = runtime.get_responses(prompts, skip_special_tokens=False)

    assert responses == ["response"]
    assert channel.sent == [DgxCommand("get_responses", (prompts,), {"skip_special_tokens": False})]
    assert local.calls == [("get_responses", (prompts,), {"skip_special_tokens": False})]


def test_coordinator_broadcasts_startup_generation_bound() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)
    prompts = [Prompt(system="system", user="user")]

    responses = runtime.get_responses_once(prompts, max_new_tokens=8)

    expected = DgxCommand(
        "get_responses_once",
        (prompts,),
        {"skip_special_tokens": True, "max_new_tokens": 8},
    )
    assert responses == ["response"]
    assert channel.sent == [expected]
    assert local.calls == [
        (
            "get_responses_once",
            (prompts,),
            {"skip_special_tokens": True, "max_new_tokens": 8},
        )
    ]


def test_coordinator_fails_when_worker_reports_operation_error() -> None:
    runtime = DgxCoordinatorRuntime(
        RecordingRuntime(),
        RecordingCoordinatorChannel(peer_error="rank 1: out of memory"),
    )

    with pytest.raises(RuntimeError, match="rank 1: out of memory"):
        runtime.reset_model()


def test_operation_error_latches_runtime_and_shutdown_avoids_second_collective() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel(peer_error="rank 1: out of memory")
    runtime = DgxCoordinatorRuntime(local, channel)

    with pytest.raises(RuntimeError, match="rank 1: out of memory"):
        runtime.reset_model()
    sent_after_failure = list(channel.sent)
    channel.peer_error = None

    with pytest.raises(RuntimeError, match="failed"):
        runtime.get_logits([])
    with pytest.raises(RuntimeError, match="failed"):
        runtime.get_model_metadata()
    runtime.shutdown()
    runtime.shutdown()

    assert channel.sent == sent_after_failure
    assert [call[0] for call in local.calls] == ["reset_model", "shutdown"]


def test_coordinator_rejects_model_switch_before_broadcast() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)

    with pytest.raises(RuntimeError, match="evaluation model.*not available"):
        runtime.reset_model("org/evaluation-model")

    assert channel.sent == []
    assert local.calls == []


def test_worker_executes_commands_without_entering_interactive_main() -> None:
    prompts = [Prompt(system="system", user="user")]
    channel = ScriptedWorkerChannel(
        [
            DgxCommand("get_logits", (prompts,), {}),
            DgxCommand("reset_model", (), {}),
            DgxCommand("shutdown", (), {}),
        ]
    )
    local = RecordingRuntime()

    run_dgx_worker(local, channel)

    assert [name for name, _args, _kwargs in local.calls] == ["get_logits", "reset_model", "shutdown"]
    assert channel.completions == [None, None, None]


def test_distributed_adapter_export_is_coordinated_across_both_ranks() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)

    runtime.save_adapter("output", max_shard_size="5GB")

    assert channel.sent == [
        DgxCommand("save_adapter", ("output",), {"max_shard_size": "5GB"})
    ]
    assert local.calls == [
        ("save_adapter", ("output",), {"max_shard_size": "5GB"})
    ]


def test_distributed_adapter_export_rejects_nonempty_destination(tmp_path) -> None:
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "stale.bin").write_bytes(b"stale")
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)

    with pytest.raises(RuntimeError, match="destination.*empty"):
        runtime.save_adapter(str(destination), max_shard_size="5GB")

    assert channel.sent == []
    assert local.calls == []


def test_coordinator_shutdown_is_idempotent_and_rejects_later_operations() -> None:
    local = RecordingRuntime()
    channel = RecordingCoordinatorChannel()
    runtime = DgxCoordinatorRuntime(local, channel)

    runtime.shutdown()
    runtime.shutdown()

    assert [command.operation for command in channel.sent] == ["shutdown"]
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.get_logits([])


def test_real_two_process_gloo_command_loop() -> None:
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
            pytest.fail("two-process DGX runtime test timed out")
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=5) for _ in range(2))
    assert observed == [
        (0, [[1.0, 2.0]], ["get_logits", "shutdown"]),
        (1, None, ["get_logits", "shutdown"]),
    ]
