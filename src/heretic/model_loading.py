# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_model_load_kwargs(
    *,
    dtype: object,
    quantization_config: object | None,
    distributed: bool,
    model_commit: str | None,
    device_map: object,
    max_memory: Mapping[str, object] | None,
    trust_remote_code: bool,
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
