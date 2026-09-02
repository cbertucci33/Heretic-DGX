# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch import nn

from heretic.model import _set_lora_adapter_dtype
from heretic.runtime import LocalModelRuntime


class TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_bf16_adapter_only_export_reloads_without_mutating_base(tmp_path) -> None:
    base = TinyBase()
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["linear"],
            bias="none",
        ),
        autocast_adapter_dtype=False,
    )
    for module in peft_model.modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            _set_lora_adapter_dtype(module, torch.bfloat16)
    for name, parameter in peft_model.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(0.125)

    base_weight = peft_model.base_model.model.linear.base_layer.weight
    base_before = base_weight.detach().clone()
    inputs = torch.ones(1, 4, dtype=torch.bfloat16)
    expected = peft_model(inputs).detach()
    runtime = LocalModelRuntime(
        SimpleNamespace(model=peft_model, distributed=False)  # type: ignore[arg-type]
    )

    runtime.save_adapter(str(tmp_path), max_shard_size="5GB")

    assert torch.equal(base_before, base_weight)
    state = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert state
    assert all("lora_" in key for key in state)
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16}
    assert not any("base_layer.weight" in key for key in state)

    fresh_base = TinyBase()
    fresh_base.linear.weight.data.copy_(base_before)
    reloaded = PeftModel.from_pretrained(
        fresh_base,
        tmp_path,
        autocast_adapter_dtype=False,
    )
    actual = reloaded(inputs).detach()

    assert torch.equal(expected, actual)
