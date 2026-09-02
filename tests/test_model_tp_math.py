# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import torch

from heretic.config import RowNormalization
from heretic.model import AbliterationParameters, Model


def test_abliterate_rowwise_pre_uses_global_row_norms(monkeypatch) -> None:
    full_weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.5, 3.0],
            [0.5, 1.5, 2.5, 3.5],
        ]
    )
    local_weight = full_weight[:, :2].clone()
    peer_weight = full_weight[:, 2:]
    direction = torch.nn.functional.normalize(torch.tensor([0.5, -1.0, 1.5]), dim=0)
    strength = 0.4

    base_layer = SimpleNamespace(weight=local_weight)
    module = SimpleNamespace(
        weight=local_weight,
        base_layer=base_layer,
        lora_A={"default": SimpleNamespace(weight=torch.zeros(1, 2))},
        lora_B={"default": SimpleNamespace(weight=torch.zeros(3, 1))},
    )
    model = object.__new__(Model)
    model.settings = SimpleNamespace(row_normalization=RowNormalization.PRE)
    model.distributed = True
    model.peft_config = SimpleNamespace(r=1)
    model._lora_target_topologies_by_base_id = {id(base_layer): "rowwise"}
    model.get_layers = lambda: [object()]
    model.get_layer_modules = lambda _: {"attn.o_proj": [module]}

    def all_reduce(value: torch.Tensor) -> None:
        value.add_(torch.sum(peer_weight.float().square(), dim=1, keepdim=True))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    model.abliterate(
        residual_directions=torch.stack([torch.zeros_like(direction), direction]),
        direction_index=None,
        parameters={
            "attn.o_proj": AbliterationParameters(
                max_weight=strength,
                max_weight_position=0.0,
                min_weight=strength,
                min_weight_distance=1.0,
            )
        },
    )

    row_norms = torch.linalg.vector_norm(full_weight, dim=1, keepdim=True)
    normalized = torch.nn.functional.normalize(full_weight, p=2, dim=1)
    expected_a = (direction @ normalized[:, :2]).view(1, -1)
    expected_b = row_norms * (-strength * direction).view(-1, 1)
    torch.testing.assert_close(module.lora_A["default"].weight, expected_a)
    torch.testing.assert_close(module.lora_B["default"].weight, expected_b)
    torch.testing.assert_close(base_layer.weight, local_weight)


def test_adapter_parameters_are_explicitly_cast_to_bfloat16() -> None:
    from heretic import model as model_module

    assert hasattr(model_module, "_set_lora_adapter_dtype"), (
        "adapter dtype enforcement is not implemented"
    )
    module = SimpleNamespace(
        lora_A={"default": torch.nn.Linear(4, 2, bias=False, dtype=torch.float32)},
        lora_B={"default": torch.nn.Linear(2, 4, bias=False, dtype=torch.float32)},
    )

    model_module._set_lora_adapter_dtype(module, torch.bfloat16)

    assert module.lora_A["default"].weight.dtype == torch.bfloat16
    assert module.lora_B["default"].weight.dtype == torch.bfloat16
