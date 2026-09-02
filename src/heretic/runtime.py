# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
import shutil
import tempfile
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from .system import empty_cache
from .utils import Prompt

if TYPE_CHECKING:
    from .model import AbliterationParameters, Model


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Declarative model and runtime identity without implementation objects."""

    runtime_kind: str
    base_model_id: str
    base_model_revision: str | None
    dtype: str
    quantization: str
    global_layer_count: int
    abliterable_components: tuple[str, ...]
    adapter_generation: int | None


class ModelRuntime(ABC):
    """Model execution surface used by Heretic orchestration and plugins."""

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata:
        """Return declarative runtime and model identity metadata."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release runtime-owned resources and reject subsequent operations."""

    @abstractmethod
    def reset_model(self, model: str | None = None) -> None:
        """Reset mutable model state before evaluation or a trial."""

    @abstractmethod
    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        """Apply abliteration parameters to the current mutable model state."""

    @abstractmethod
    def save_adapter(
        self,
        directory: str,
        *,
        max_shard_size: int | str,
    ) -> None:
        """Save the current adapter without merging it into the base model."""

    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Return responses from one direct model call without runtime batching."""
        if max_new_tokens is not None:
            raise RuntimeError("runtime does not support a bounded direct response")
        return self.get_responses(prompts, skip_special_tokens=skip_special_tokens)

    @abstractmethod
    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Return responses for prompts in input order."""

    @abstractmethod
    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        """Return raw first-generated-token logits in prompt order."""

    @abstractmethod
    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        """Return first-generated-token residuals in global layer order."""

    @abstractmethod
    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        """Return the prompt mean of residuals in global layer order."""


def _bf16_lora_state_dict(model: object) -> dict[str, Tensor]:
    state_dict = model.state_dict()  # type: ignore[attr-defined]
    adapter_state = {
        key: tensor.detach().to(dtype=torch.bfloat16)
        for key, tensor in state_dict.items()
        if key.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
    }
    if not adapter_state:
        raise RuntimeError("no default LoRA adapter tensors are available to export")
    return adapter_state


class LocalModelRuntime(ModelRuntime):
    """Behavior-preserving runtime adapter for the existing local Model."""

    distributed = False

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

    def get_model_metadata(self) -> ModelMetadata:
        self._require_active()
        quantization_config = getattr(self._model.model.config, "quantization_config", None)
        if isinstance(quantization_config, Mapping):
            quantization = quantization_config.get("quant_method")
        else:
            quantization = getattr(quantization_config, "quant_method", None)
        if quantization is None:
            quantization = self._model.settings.quantization.value

        return ModelMetadata(
            runtime_kind="local",
            base_model_id=self._model.settings.model,
            base_model_revision=self._model.settings.model_commit,
            dtype=str(self._model.dtype),
            quantization=str(quantization),
            global_layer_count=len(self._model.get_layers()),
            abliterable_components=tuple(self._model.get_abliterable_components()),
            adapter_generation=None,
        )

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

    def save_adapter(
        self,
        directory: str,
        *,
        max_shard_size: int | str,
    ) -> None:
        self._require_active()
        adapter_state = _bf16_lora_state_dict(self._model.model)
        save_options = {
            "max_shard_size": max_shard_size,
            "state_dict": adapter_state,
            "selected_adapters": ["default"],
            "safe_serialization": True,
            "save_embedding_layers": False,
        }
        if not getattr(self._model, "distributed", False):
            self._model.model.save_pretrained(directory, **save_options)
            return

        rank = dist.get_rank()
        if rank == 0:
            self._model.model.save_pretrained(
                directory,
                **save_options,
                is_main_process=True,
            )
            return

        sink = tempfile.mkdtemp(prefix=f"heretic-adapter-rank-{rank}-")
        try:
            self._model.model.save_pretrained(
                sink,
                **save_options,
                is_main_process=False,
            )
        finally:
            with suppress(OSError):
                shutil.rmtree(sink)

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

    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        self._require_active()
        return self._model.get_responses(
            prompts,
            skip_special_tokens=skip_special_tokens,
            max_new_tokens=max_new_tokens,
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
