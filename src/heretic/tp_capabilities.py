# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor
from torch.distributed.tensor import Replicate, Shard

TargetTopology = Literal["replicated", "rowwise", "colwise"]
_SUPPORTED_TP_LORA_STYLES = frozenset({"rowwise", "colwise"})


class LoraTarget(Protocol):
    """Minimal Transformers TP metadata used by the PEFT compatibility gate."""


def _matching_model_plan(
    target_name: str,
    model_tp_plan: Mapping[str, object],
) -> object | None:
    target_parts = target_name.split(".")
    matches = [
        value
        for pattern, value in model_tp_plan.items()
        if len(pattern.split(".")) == len(target_parts)
        and all(
            expected == "*" or expected == actual
            for expected, actual in zip(pattern.split("."), target_parts, strict=True)
        )
    ]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous tensor-parallel plan for LoRA target {target_name!r}"
        )
    return matches[0] if matches else None


def _dtensor_topology(name: str, module: LoraTarget) -> TargetTopology | None:
    weight = getattr(module, "weight", None)
    placements = getattr(weight, "placements", None)
    mesh = getattr(weight, "device_mesh", None)
    if placements is None and mesh is None:
        return None
    if placements is None or mesh is None:
        raise ValueError(f"incomplete DTensor metadata for LoRA target {name!r}")
    if mesh.size() != 2:
        raise ValueError(
            f"LoRA target {name!r} requires a two-rank device mesh; got {mesh.size()}"
        )
    if len(placements) != 1:
        raise ValueError(
            f"unsupported DTensor placement for LoRA target {name!r}: "
            f"placements={placements!r}"
        )
    placement = placements[0]
    if isinstance(placement, Replicate):
        return "replicated"
    if isinstance(placement, Shard):
        if placement.dim == 0:
            return "colwise"
        if placement.dim == 1:
            return "rowwise"
    raise ValueError(
        f"unsupported DTensor placement for LoRA target {name!r}: "
        f"placements={placements!r}"
    )


def inspect_lora_target_topologies(
    targets: Mapping[str, LoraTarget],
    *,
    model_tp_plan: Mapping[str, object] | None = None,
) -> dict[str, TargetTopology]:
    """Fail closed unless every distributed LoRA target has a proven layout."""

    topologies: dict[str, TargetTopology] = {}
    for name, module in targets.items():
        plan = getattr(module, "_hf_tp_plan", None)
        mesh = getattr(module, "_hf_device_mesh", None)
        dtensor_topology = _dtensor_topology(name, module)

        if dtensor_topology is not None:
            model_plan = (
                _matching_model_plan(name, model_tp_plan)
                if model_tp_plan is not None
                else None
            )
            if dtensor_topology == "replicated":
                if model_plan is not None and model_plan != dtensor_topology:
                    raise ValueError(
                        "tensor-parallel plan disagrees with DTensor placement for "
                        f"LoRA target {name!r}: plan={model_plan!r}, "
                        f"placement={dtensor_topology!r}"
                    )
                topologies[name] = dtensor_topology
                continue
            if model_tp_plan is None:
                raise ValueError(
                    f"LoRA target {name!r} has a sharded DTensor but the model is "
                    "missing its tensor-parallel plan"
                )
            if model_plan != dtensor_topology:
                raise ValueError(
                    "tensor-parallel plan disagrees with DTensor placement for LoRA "
                    f"target {name!r}: plan={model_plan!r}, "
                    f"placement={dtensor_topology!r}"
                )
            topologies[name] = dtensor_topology
            continue

        if (plan is None) != (mesh is None):
            raise ValueError(
                f"incomplete tensor-parallel metadata for LoRA target {name!r}: "
                f"plan={plan!r}, device_mesh={'present' if mesh is not None else 'missing'}"
            )
        if plan is None:
            raise ValueError(
                f"LoRA target {name!r} is missing tensor-parallel metadata; "
                "refusing to assume that it is replicated"
            )
        if plan not in _SUPPORTED_TP_LORA_STYLES:
            supported = ", ".join(sorted(_SUPPORTED_TP_LORA_STYLES))
            raise ValueError(
                f"unsupported tensor-parallel LoRA target {name!r}: plan={plan!r}; "
                f"Heretic requires one of {supported} or a provably replicated layer"
            )
        topologies[name] = plan
    return topologies


@dataclass(frozen=True)
class LoraFactors:
    a: Tensor
    b: Tensor


def directional_lora_factors(
    local_weight: Tensor,
    direction: Tensor,
    *,
    strength: float,
    normalization: Literal["none", "pre"],
    topology: Literal["rowwise", "colwise"],
    sum_across_ranks: Callable[[Tensor], Tensor],
) -> LoraFactors:
    """Compute shard-local factors equivalent to the unsharded operation."""

    if normalization not in {"none", "pre"}:
        raise ValueError(
            f"unsupported distributed row normalization: {normalization!r}"
        )

    if topology == "rowwise":
        if normalization == "pre":
            local_squares = torch.sum(
                local_weight.float().square(), dim=1, keepdim=True
            )
            row_norms = torch.sqrt(sum_across_ranks(local_squares))
            effective_weight = local_weight / row_norms.clamp_min(1e-12).to(
                local_weight.dtype
            )
            b = row_norms.to(direction.dtype) * (-strength * direction).view(-1, 1)
        else:
            effective_weight = local_weight
            b = (-strength * direction).view(-1, 1)
        a = (direction.to(effective_weight.dtype) @ effective_weight).view(1, -1)
    elif topology == "colwise":
        if normalization == "pre":
            row_norms = torch.linalg.vector_norm(
                local_weight.float(), dim=1, keepdim=True
            )
            effective_weight = local_weight / row_norms.clamp_min(1e-12).to(
                local_weight.dtype
            )
            b = row_norms.to(direction.dtype) * (-strength * direction).view(-1, 1)
        else:
            effective_weight = local_weight
            b = (-strength * direction).view(-1, 1)
        local_a = (direction.to(effective_weight.dtype) @ effective_weight).view(1, -1)
        a = sum_across_ranks(local_a)
    else:
        raise ValueError(f"unsupported tensor-parallel topology: {topology!r}")

    return LoraFactors(a=a, b=b)
