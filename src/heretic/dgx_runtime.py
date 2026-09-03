# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch.distributed as dist
from torch import Tensor

from .model import AbliterationParameters
from .runtime import ModelRuntime
from .utils import Prompt


DgxOperation = Literal[
    "shutdown",
    "reset_model",
    "abliterate",
    "save_adapter",
    "get_responses_once",
    "get_responses",
    "get_logits",
    "get_residuals",
    "get_residuals_mean",
]
_ALLOWED_OPERATIONS = frozenset(DgxOperation.__args__)


@dataclass(frozen=True, slots=True)
class DgxCommand:
    operation: DgxOperation
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.operation) is not str or self.operation not in _ALLOWED_OPERATIONS:
            raise ValueError("unsupported DGX runtime operation")
        if type(self.args) is not tuple:
            raise TypeError("DGX command args must be exactly tuple")
        if type(self.kwargs) is not dict or any(type(key) is not str for key in self.kwargs):
            raise TypeError("DGX command kwargs must be a string-keyed dict")


class DgxCommandChannel(Protocol):
    def send(self, command: DgxCommand) -> None: ...

    def receive(self) -> DgxCommand: ...

    def complete(self, local_error: str | None) -> tuple[str | None, str | None]: ...


class TorchDistributedCommandChannel:
    """Typed two-rank command channel over the initialized process group."""

    def __init__(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DGX command channel requires an initialized process group")
        if dist.get_world_size() != 2:
            raise RuntimeError("DGX command channel requires exactly two ranks")
        self._rank = dist.get_rank()

    def send(self, command: DgxCommand) -> None:
        if self._rank != 0:
            raise RuntimeError("only DGX rank 0 may send commands")
        payload: list[object] = [command]
        dist.broadcast_object_list(payload, src=0)

    def receive(self) -> DgxCommand:
        if self._rank != 1:
            raise RuntimeError("only DGX rank 1 may receive commands")
        payload: list[object] = [None]
        dist.broadcast_object_list(payload, src=0)
        command = payload[0]
        if type(command) is not DgxCommand:
            raise TypeError("DGX command payload has an invalid type")
        return command

    def complete(self, local_error: str | None) -> tuple[str | None, str | None]:
        errors: list[object] = [None, None]
        dist.all_gather_object(errors, local_error)
        if any(error is not None and type(error) is not str for error in errors):
            raise TypeError("DGX completion payload has an invalid type")
        return errors[0], errors[1]  # type: ignore[return-value]


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


class DgxCoordinatorRuntime(ModelRuntime):
    """Rank-0 runtime that mirrors model work to rank 1."""

    distributed = True

    def __init__(self, local: ModelRuntime, channel: DgxCommandChannel) -> None:
        self._local = local
        self._channel = channel
        self._active = True
        self._failed = False
        self._local_stopped = False

    def _invoke(self, operation: DgxOperation, *args: Any, **kwargs: Any) -> Any:
        if self._failed:
            raise RuntimeError("DGX model runtime has failed")
        if not self._active:
            if operation == "shutdown":
                return None
            raise RuntimeError("DGX model runtime has been shut down")

        self._channel.send(DgxCommand(operation, args, kwargs))
        local_error: BaseException | None = None
        result: Any = None
        try:
            result = getattr(self._local, operation)(*args, **kwargs)
        except BaseException as error:
            local_error = error

        try:
            rank_errors = self._channel.complete(
                None if local_error is None else _error_text(local_error)
            )
        except BaseException:
            self._failed = True
            self._active = False
            raise

        if local_error is not None or rank_errors[1] is not None:
            self._failed = True
            self._active = False
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"DGX worker failed: {rank_errors[1]}")
        if operation == "shutdown":
            self._active = False
            self._local_stopped = True
        return result

    def shutdown(self) -> None:
        if self._failed:
            if not self._local_stopped:
                self._local.shutdown()
                self._local_stopped = True
            return
        self._invoke("shutdown")

    def reset_model(self, model: str | None = None) -> None:
        self._invoke("reset_model", model)

    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        self._invoke("abliterate", residual_directions, direction_index, parameters)

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        self._invoke("save_adapter", directory, max_shard_size=max_shard_size)

    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._invoke(
            "get_responses_once",
            prompts,
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._invoke(
            "get_responses",
            prompts,
            skip_special_tokens=skip_special_tokens,
        )

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        return self._invoke("get_logits", prompts)

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        return self._invoke("get_residuals", prompts)

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        return self._invoke("get_residuals_mean", prompts)


def run_dgx_worker(local: ModelRuntime, channel: DgxCommandChannel) -> None:
    """Execute coordinator commands without entering Heretic's UI or storage."""

    while True:
        command = channel.receive()
        local_error: BaseException | None = None
        try:
            getattr(local, command.operation)(*command.args, **command.kwargs)
        except BaseException as error:
            local_error = error

        rank_errors = channel.complete(
            None if local_error is None else _error_text(local_error)
        )
        if local_error is not None:
            raise local_error
        if rank_errors[0] is not None:
            raise RuntimeError(f"DGX coordinator failed: {rank_errors[0]}")
        if command.operation == "shutdown":
            return
