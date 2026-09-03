# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_LAGUNA_DGX_TP_PLAN = {
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.gate_up_proj_scale_inv": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts.down_proj_scale_inv": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
}


def complete_laguna_dgx_tp_plan(config: object) -> object:
    """Add the TP entries omitted by Laguna S 2.1's shipped custom config."""

    current = getattr(config, "base_model_tp_plan", None)
    if not isinstance(current, dict):
        raise TypeError("Laguna config must provide a base_model_tp_plan dictionary")
    config.base_model_tp_plan = {**current, **_LAGUNA_DGX_TP_PLAN}
    return config


def build_model_load_kwargs(
    *,
    dtype: object,
    quantization_config: object | None,
    distributed: bool,
    model_commit: str | None,
    device_map: object,
    max_memory: Mapping[str, object] | None,
    trust_remote_code: bool,
    config: object | None = None,
) -> dict[str, Any]:
    """Build mutually exclusive local or fixed DGX TP loader arguments."""

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True if trust_remote_code else None,
    }
    if model_commit is not None:
        kwargs["revision"] = model_commit
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if config is not None:
        kwargs["config"] = config

    if distributed:
        kwargs["tp_plan"] = "auto"
    else:
        kwargs["device_map"] = device_map
        kwargs["max_memory"] = (
            {
                int(key) if key.isdigit() else key: value
                for key, value in max_memory.items()
            }
            if max_memory
            else None
        )
    return kwargs
