# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from enum import Enum
from typing import Protocol


class DistributedStrategy(str, Enum):
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class DistributedModelConfig(Protocol):
    model_type: str
    base_model_tp_plan: object
    base_model_pp_plan: object


def select_distributed_strategy(config: DistributedModelConfig) -> DistributedStrategy:
    """Select a declared distributed plan without architecture allowlists."""

    if getattr(config, "base_model_tp_plan", None):
        return DistributedStrategy.TENSOR_PARALLEL
    model_type = getattr(config, "model_type", "unknown")
    if getattr(config, "base_model_pp_plan", None):
        raise ValueError(
            f"model type {model_type!r} declares only a pipeline-parallel plan, "
            "but Heretic's pipeline-parallel runtime is not implemented"
        )

    raise ValueError(
        f"model type {model_type!r} does not declare a tensor-parallel or "
        "pipeline-parallel plan; distributed execution cannot be proven safe"
    )
