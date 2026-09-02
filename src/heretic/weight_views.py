# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Protocol, cast

import bitsandbytes as bnb
import torch
from torch import Tensor
from transformers.integrations.finegrained_fp8 import FP8Linear, Fp8Dequantize


class WeightedModule(Protocol):
    weight: Tensor


def dequantized_weight_view(module: WeightedModule) -> Tensor:
    """Return an FP32 weight view without replacing or mutating the base module."""

    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        return cast(
            Tensor,
            bnb.functional.dequantize_4bit(weight.data, quant_state),  # ty:ignore[possibly-missing-attribute]
        ).to(torch.float32)

    if isinstance(module, FP8Linear):
        scales = cast(Tensor, module.weight_scale_inv)
        return Fp8Dequantize(None)._dequantize_one(  # ty:ignore[invalid-argument-type]
            weight,
            scales,
            output_dtype=torch.float32,
        )

    if weight.is_quantized:
        return weight.dequantize().to(torch.float32)

    if weight.element_size() == 1:
        module_type = f"{type(module).__module__}.{type(module).__qualname__}"
        raise TypeError(
            f"unsupported quantized weight module {module_type}; "
            "Heretic will not interpret raw one-byte weights without a verified scale adapter"
        )

    return weight.to(torch.float32)
