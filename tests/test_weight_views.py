# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import pytest
import torch
from transformers.integrations.finegrained_fp8 import FP8Linear

from heretic.weight_views import dequantized_weight_view


def test_plain_weight_returns_fp32_view_without_mutation() -> None:
    module = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    before = module.weight.detach().clone()

    view = dequantized_weight_view(module)

    assert view.dtype == torch.float32
    torch.testing.assert_close(view, before.float())
    torch.testing.assert_close(module.weight, before)
    assert module.weight.dtype == torch.bfloat16


def test_finegrained_fp8_applies_per_block_inverse_scales_without_mutation() -> None:
    module = FP8Linear(4, 4, block_size=(2, 2))
    quantized = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    module.weight.data.copy_(quantized)
    module.weight_scale_inv.data.copy_(scales)
    weight_before = module.weight.detach().clone()
    scales_before = module.weight_scale_inv.detach().clone()

    view = dequantized_weight_view(module)

    expected = torch.tensor(
        [
            [2.0, 4.0, 9.0, 12.0],
            [4.0, 2.0, 12.0, 9.0],
            [4.0, 4.0, 10.0, 10.0],
            [12.0, 12.0, 20.0, 20.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(view, expected)
    torch.testing.assert_close(module.weight, weight_before)
    torch.testing.assert_close(module.weight_scale_inv, scales_before)


def test_unknown_byte_weight_fails_closed() -> None:
    module = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 2, dtype=torch.uint8), requires_grad=False))

    with pytest.raises(TypeError, match="unsupported quantized weight module"):
        dequantized_weight_view(module)
