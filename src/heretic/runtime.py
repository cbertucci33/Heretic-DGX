# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from .system import empty_cache
from .utils import Prompt

if TYPE_CHECKING:
    from .model import AbliterationParameters, Model


def gather_tensor_parallel_lora_shard(local: Tensor, *, dimension: int) -> Tensor:
    """Gather equal rank-local LoRA shards into checkpoint tensor order."""

    shards = [local.new_empty(local.shape) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, local.contiguous())
    return torch.cat(shards, dim=dimension)


class ModelRuntime(ABC):
    """Model operations that must execute in lockstep across active ranks."""

    distributed = False

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def reset_model(self, model: str | None = None) -> None: ...

    @abstractmethod
    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None: ...

    @abstractmethod
    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None: ...

    @abstractmethod
    def save_merged(self, directory: str, *, max_shard_size: int | str) -> None: ...

    @abstractmethod
    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]: ...

    @abstractmethod
    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]: ...

    @abstractmethod
    def get_logits(self, prompts: list[Prompt]) -> Tensor: ...

    @abstractmethod
    def get_residuals(self, prompts: list[Prompt]) -> Tensor: ...

    @abstractmethod
    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor: ...


class LocalModelRuntime(ModelRuntime):
    """Behavior-preserving adapter around the existing local Model."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._is_shutdown = False

    def _require_active(self) -> None:
        if self._is_shutdown:
            raise RuntimeError("model runtime has been shut down")

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        self._model.model = None  # type: ignore[assignment]
        empty_cache()

    def reset_model(self, model: str | None = None) -> None:
        self._require_active()
        if model is not None:
            self._model.settings.model = model
        self._model.reset_model()

    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        self._require_active()
        self._model.abliterate(residual_directions, direction_index, parameters)

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        self._require_active()
        if not self._model.distributed:
            self._model.model.save_pretrained(
                directory,
                max_shard_size=max_shard_size,
            )
            return

        rank = dist.get_rank()
        restored: list[tuple[Tensor, Tensor]] = []
        for module in self._model.model.modules():
            base_layer = getattr(module, "base_layer", None)
            topology = self._model._lora_target_topologies_by_base_id.get(
                id(base_layer)
            )
            if topology == "rowwise":
                parameter = module.lora_A["default"].weight
                dimension = 1
            elif topology == "colwise":
                parameter = module.lora_B["default"].weight
                dimension = 0
            else:
                continue
            full = gather_tensor_parallel_lora_shard(
                parameter.detach(),
                dimension=dimension,
            )
            if rank == 0:
                restored.append((parameter, parameter.data))
                parameter.data = full

        sink = (
            directory
            if rank == 0
            else tempfile.mkdtemp(prefix=f"heretic-adapter-rank-{rank}-")
        )
        try:
            self._model.model.save_pretrained(
                sink,
                max_shard_size=max_shard_size,
                is_main_process=rank == 0,
            )
        finally:
            for parameter, local in restored:
                parameter.data = local
            if rank != 0:
                with suppress(OSError):
                    shutil.rmtree(sink)

    def save_merged(self, directory: str, *, max_shard_size: int | str) -> None:
        self._require_active()
        merged_model = self._model.get_merged_model()
        rank = dist.get_rank() if self._model.distributed else 0
        sink = directory
        temporary_sink = False
        if rank != 0:
            sink = tempfile.mkdtemp(prefix=f"heretic-merged-rank-{rank}-")
            temporary_sink = True
        try:
            merged_model.save_pretrained(
                sink,
                max_shard_size=max_shard_size,
                is_main_process=rank == 0,
            )
        finally:
            del merged_model
            empty_cache()
            if temporary_sink:
                with suppress(OSError):
                    shutil.rmtree(sink)

    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self._require_active()
        return self._model.get_responses(
            prompts,
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self._require_active()
        return self._model.get_responses_batched(
            prompts,
            skip_special_tokens=skip_special_tokens,
        )

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        self._require_active()
        return self._model.get_logits_batched(prompts)

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        self._require_active()
        return self._model.get_residuals_batched(prompts)

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        self._require_active()
        return self._model.get_residuals_mean(prompts)
