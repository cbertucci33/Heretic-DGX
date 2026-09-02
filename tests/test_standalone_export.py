# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

from safetensors.torch import save_file

from heretic.standalone_export import (
    LAGUNA_S_2_1_FP8_IDENTITY,
    LoraDelta,
    checkpoint_identity,
    export_standalone_laguna,
    load_bf16_lora_deltas,
    merge_bf16_lora_weight,
    save_runtime_as_standalone,
    validate_laguna_checkpoint,
    verify_checkpoint_identity,
    verify_sha256_manifest,
    verify_standalone_laguna,
)


def test_bf16_merge_matches_real_peft_merge() -> None:
    torch.manual_seed(7)
    base = torch.nn.Sequential()
    base.add_module(
        "linear",
        torch.nn.Linear(5, 3, bias=False, dtype=torch.bfloat16),
    )
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=2, target_modules=["linear"], bias="none"),
        autocast_adapter_dtype=False,
    )
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(
            torch.arange(10, dtype=torch.float32).reshape(2, 5).to(torch.bfloat16) / 16
        )
        layer.lora_B["default"].weight.copy_(
            torch.arange(6, dtype=torch.float32).reshape(3, 2).to(torch.bfloat16) / 8
        )
    base_before = layer.base_layer.weight.detach().clone()
    expected = (
        base_before + layer.get_delta_weight("default")
    ).to(torch.bfloat16)

    actual = merge_bf16_lora_weight(
        base_before,
        layer.lora_A["default"].weight,
        layer.lora_B["default"].weight,
        scaling=layer.scaling["default"],
    )

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


def test_merge_rejects_non_bf16_base_weight() -> None:
    with pytest.raises(ValueError, match="BF16"):
        merge_bf16_lora_weight(
            torch.zeros((3, 5), dtype=torch.float8_e4m3fn),
            torch.zeros((2, 5), dtype=torch.bfloat16),
            torch.zeros((3, 2), dtype=torch.bfloat16),
            scaling=1.0,
        )


def test_loads_only_complete_bf16_o_proj_lora_pairs(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"modules_to_save":null,'
        '"rank_pattern":{},"alpha_pattern":{}}'
    )
    prefix = "base_model.model.model.layers.3.self_attn.o_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
            f"{prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
        },
        tmp_path / "adapter_model.safetensors",
    )

    deltas = load_bf16_lora_deltas(tmp_path)

    assert set(deltas) == {"model.layers.3.self_attn.o_proj.weight"}
    delta = deltas["model.layers.3.self_attn.o_proj.weight"]
    assert delta.scaling == 1.0
    assert delta.lora_a.dtype is torch.bfloat16
    assert delta.lora_b.dtype is torch.bfloat16


def test_validates_protected_bf16_targets_in_fp8_checkpoint(tmp_path) -> None:
    base_key = "model.layers.0.self_attn.o_proj.weight"
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / "config.json").write_text(
        '{"model_type":"laguna","architectures":["LagunaForCausalLM"],'
        '"num_hidden_layers":1,"quantization_config":'
        '{"quant_method":"fp8","weight_block_size":[128,128],'
        '"activation_scheme":"dynamic","ignored_layers":'
        '["model.layers.0.self_attn.o_proj"]}}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"' + base_key + '":"' + shard + '"}}'
    )
    save_file(
        {base_key: torch.zeros((3, 5), dtype=torch.bfloat16)},
        tmp_path / shard,
    )
    deltas = {
        base_key: LoraDelta(
            torch.zeros((2, 5), dtype=torch.bfloat16),
            torch.zeros((3, 2), dtype=torch.bfloat16),
            1.0,
        )
    }

    plan = validate_laguna_checkpoint(tmp_path, deltas, expected_layer_count=1)

    assert plan == {shard: (base_key,)}


def test_exports_standalone_checkpoint_without_changing_fp8_tensors(tmp_path) -> None:
    source = tmp_path / "source"
    adapter = tmp_path / "adapter"
    output = tmp_path / "output"
    source.mkdir()
    adapter.mkdir()
    base_key = "model.layers.0.self_attn.o_proj.weight"
    fp8_key = "model.layers.0.mlp.gate_proj.weight"
    other_key = "model.embed_tokens.weight"
    changed_shard = "model-00001-of-00002.safetensors"
    untouched_shard = "model-00002-of-00002.safetensors"
    config_text = (
        '{"model_type":"laguna","architectures":["LagunaForCausalLM"],'
        '"num_hidden_layers":1,"quantization_config":'
        '{"quant_method":"fp8","weight_block_size":[128,128],'
        '"activation_scheme":"dynamic","ignored_layers":'
        '["model.layers.0.self_attn.o_proj"]}}'
    )
    index_text = (
        '{"weight_map":{"' + base_key + '":"' + changed_shard + '","'
        + fp8_key + '":"' + changed_shard + '","'
        + other_key + '":"' + untouched_shard + '"}}'
    )
    (source / "config.json").write_text(config_text)
    (source / "model.safetensors.index.json").write_text(index_text)
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / ".cache" / "huggingface" / "download.lock").write_text("local cache")
    base = torch.arange(15, dtype=torch.float32).reshape(3, 5).to(torch.bfloat16)
    fp8 = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
    save_file({base_key: base, fp8_key: fp8}, source / changed_shard)
    raw_shard = (source / changed_shard).read_bytes()
    raw_header_length = int.from_bytes(raw_shard[:8], "little")
    raw_header_end = 8 + raw_header_length
    raw_header = json.loads(raw_shard[8:raw_header_end])
    reversed_header = dict(reversed(list(raw_header.items())))
    encoded_header = json.dumps(reversed_header, separators=(",", ":")).encode()
    assert len(encoded_header) <= raw_header_length
    encoded_header += b" " * (raw_header_length - len(encoded_header))
    (source / changed_shard).write_bytes(
        raw_shard[:8] + encoded_header + raw_shard[raw_header_end:]
    )
    save_file(
        {other_key: torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)},
        source / untouched_shard,
    )
    (adapter / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"modules_to_save":null,'
        '"rank_pattern":{},"alpha_pattern":{}}'
    )
    prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    lora_a = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(torch.bfloat16) / 16
    lora_b = torch.arange(6, dtype=torch.float32).reshape(3, 2).to(torch.bfloat16) / 8
    save_file(
        {
            f"{prefix}.lora_A.weight": lora_a,
            f"{prefix}.lora_B.weight": lora_b,
        },
        adapter / "adapter_model.safetensors",
    )
    untouched_bytes = (source / untouched_shard).read_bytes()
    changed_bytes = (source / changed_shard).read_bytes()
    header_length = int.from_bytes(changed_bytes[:8], "little")
    header_end = 8 + header_length
    header = json.loads(changed_bytes[8:header_end])
    target_start, target_end = header[base_key]["data_offsets"]
    target_start += header_end
    target_end += header_end

    export_standalone_laguna(source, adapter, output, expected_layer_count=1)

    output_bytes = (output / changed_shard).read_bytes()
    assert len(output_bytes) == len(changed_bytes)
    assert output_bytes[:target_start] == changed_bytes[:target_start]
    assert output_bytes[target_end:] == changed_bytes[target_end:]
    from safetensors.torch import load_file
    state = load_file(output / changed_shard)
    assert torch.equal(
        state[base_key],
        merge_bf16_lora_weight(base, lora_a, lora_b, scaling=1.0),
    )
    assert state[fp8_key].dtype is torch.float8_e4m3fn
    assert torch.equal(state[fp8_key].float(), fp8.float())
    assert (output / untouched_shard).read_bytes() == untouched_bytes
    assert (output / "config.json").read_text() == config_text
    assert (output / "model.safetensors.index.json").read_text() == index_text
    assert not (output / "adapter_model.safetensors").exists()
    assert not (output / "adapter_config.json").exists()
    assert not (output / ".cache").exists()
    report = verify_standalone_laguna(
        source,
        adapter,
        output,
        expected_layer_count=1,
    )
    assert report.changed_targets == (base_key,)
    assert report.rewritten_shards == (changed_shard,)

    corrupted = bytearray((output / changed_shard).read_bytes())
    assert corrupted[header_end - 1] == 32
    corrupted[header_end - 1] = 10
    (output / changed_shard).write_bytes(corrupted)
    with pytest.raises(ValueError, match="outside intended target intervals"):
        verify_standalone_laguna(
            source,
            adapter,
            output,
            expected_layer_count=1,
        )


def test_runtime_adapter_is_consumed_into_standalone_checkpoint(tmp_path) -> None:
    source = tmp_path / "source-runtime"
    output = tmp_path / "output-runtime"
    source.mkdir()
    key = "model.layers.0.self_attn.o_proj.weight"
    shard = "model.safetensors"
    (source / "config.json").write_text(
        '{"model_type":"laguna","architectures":["LagunaForCausalLM"],'
        '"num_hidden_layers":1,"quantization_config":'
        '{"quant_method":"fp8","weight_block_size":[128,128],'
        '"activation_scheme":"dynamic","ignored_layers":'
        '["model.layers.0.self_attn.o_proj"]}}'
    )
    (source / "model.safetensors.index.json").write_text(
        '{"weight_map":{"' + key + '":"' + shard + '"}}'
    )
    save_file({key: torch.zeros((3, 5), dtype=torch.bfloat16)}, source / shard)

    class FileWritingRuntime:
        calls = 0
        events: list[str] = []

        def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
            self.calls += 1
            self.events.append("save")
            assert max_shard_size == "1GB"
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(
                '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
                '"fan_in_fan_out":false,"use_dora":false,"modules_to_save":null,'
                '"rank_pattern":{},"alpha_pattern":{}}'
            )
            prefix = "base_model.model.model.layers.0.self_attn.o_proj"
            save_file(
                {
                    f"{prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
                    f"{prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
                },
                adapter / "adapter_model.safetensors",
            )

        def shutdown(self) -> None:
            self.events.append("shutdown")

    runtime = FileWritingRuntime()
    report = save_runtime_as_standalone(
        runtime,
        source_directory=source,
        destination_directory=output,
        max_shard_size="1GB",
        expected_identity=checkpoint_identity(
            source,
            model_id="test/fixture",
            revision="fixture-revision",
        ),
        expected_layer_count=1,
    )

    assert report.changed_targets == (key,)
    assert report.rewritten_shards == (shard,)
    assert runtime.calls == 1
    assert runtime.events == ["save", "shutdown"]
    assert output.is_dir()
    assert (output / "SHA256SUMS").is_file()
    verify_sha256_manifest(output)
    assert not list(tmp_path.glob(".heretic-adapter-*"))


def test_export_strategy_accepts_standalone_without_changing_existing_values() -> None:
    from heretic.config import ExportStrategy

    assert ExportStrategy("standalone") is ExportStrategy.STANDALONE
    assert ExportStrategy("adapter") is ExportStrategy.ADAPTER
    assert ExportStrategy("merge") is ExportStrategy.MERGE


def test_standalone_strategy_requires_explicit_local_checkpoint(tmp_path) -> None:
    from heretic.config import ExportStrategy
    from heretic.main import require_standalone_source_directory

    source = tmp_path / "checkpoint"
    source.mkdir()
    assert require_standalone_source_directory(ExportStrategy.STANDALONE, source) == source.resolve()
    assert require_standalone_source_directory(ExportStrategy.ADAPTER, "org/model") is None
    with pytest.raises(ValueError, match="explicit local checkpoint"):
        require_standalone_source_directory(ExportStrategy.STANDALONE, "org/model")


def test_standalone_upload_cannot_fall_through_to_full_merge() -> None:
    from heretic.config import ExportStrategy
    from heretic.main import supports_direct_upload

    assert supports_direct_upload(ExportStrategy.ADAPTER)
    assert supports_direct_upload(ExportStrategy.MERGE)
    assert not supports_direct_upload(ExportStrategy.STANDALONE)


def test_ignores_non_target_adapter_only_when_b_factor_is_exactly_zero(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"use_rslora":false,'
        '"modules_to_save":null,"rank_pattern":{},"alpha_pattern":{}}'
    )
    o_prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    other_prefix = "base_model.model.model.layers.0.mlp.down_proj"
    save_file(
        {
            f"{o_prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
            f"{o_prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
            f"{other_prefix}.lora_A.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            f"{other_prefix}.lora_B.weight": torch.zeros((3, 2), dtype=torch.bfloat16),
        },
        tmp_path / "adapter_model.safetensors",
    )

    deltas = load_bf16_lora_deltas(tmp_path)

    assert tuple(deltas) == ("model.layers.0.self_attn.o_proj.weight",)


def test_rejects_rslora_scaling_variant(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"use_rslora":true,'
        '"modules_to_save":null,"rank_pattern":{},"alpha_pattern":{}}'
    )
    prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
            f"{prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
        },
        tmp_path / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="ordinary linear LoRA"):
        load_bf16_lora_deltas(tmp_path)


def test_rejects_non_finite_adapter_factors(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"use_rslora":false,'
        '"modules_to_save":null,"rank_pattern":{},"alpha_pattern":{}}'
    )
    prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    lora_b = torch.ones((3, 2), dtype=torch.bfloat16)
    lora_b[0, 0] = float("nan")
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
            f"{prefix}.lora_B.weight": lora_b,
        },
        tmp_path / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="finite"):
        load_bf16_lora_deltas(tmp_path)


def test_rejects_unpinned_laguna_checkpoint_identity(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    with pytest.raises(ValueError, match="pinned Laguna"):
        verify_checkpoint_identity(tmp_path, LAGUNA_S_2_1_FP8_IDENTITY)


def test_rejects_nonzero_non_target_adapter(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":2,"lora_alpha":2,"bias":"none",'
        '"fan_in_fan_out":false,"use_dora":false,"use_rslora":false,'
        '"modules_to_save":null,"rank_pattern":{},"alpha_pattern":{}}'
    )
    o_prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    other_prefix = "base_model.model.model.layers.0.mlp.down_proj"
    save_file(
        {
            f"{o_prefix}.lora_A.weight": torch.ones((2, 5), dtype=torch.bfloat16),
            f"{o_prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
            f"{other_prefix}.lora_A.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            f"{other_prefix}.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
        },
        tmp_path / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="not exactly zero"):
        load_bf16_lora_deltas(tmp_path)


def test_distributed_export_requires_standalone_but_local_choices_are_preserved() -> None:
    from heretic.config import ExportStrategy
    from heretic.main import require_distributed_standalone_export

    assert (
        require_distributed_standalone_export(
            ExportStrategy.STANDALONE,
            distributed=True,
        )
        is ExportStrategy.STANDALONE
    )
    assert (
        require_distributed_standalone_export(
            ExportStrategy.ADAPTER,
            distributed=False,
        )
        is ExportStrategy.ADAPTER
    )
    with pytest.raises(ValueError, match="standalone"):
        require_distributed_standalone_export(
            ExportStrategy.ADAPTER,
            distributed=True,
        )
    with pytest.raises(ValueError, match="standalone"):
        require_distributed_standalone_export(
            ExportStrategy.MERGE,
            distributed=True,
        )


def test_distributed_export_menu_only_offers_standalone() -> None:
    from heretic.config import ExportStrategy, QuantizationMethod
    from heretic.main import export_strategy_choices

    distributed = export_strategy_choices(
        distributed=True,
        quantization=QuantizationMethod.NONE,
    )
    local = export_strategy_choices(
        distributed=False,
        quantization=QuantizationMethod.NONE,
    )

    assert [choice.value for choice in distributed] == [ExportStrategy.STANDALONE]
    assert [choice.value for choice in local] == [
        ExportStrategy.MERGE,
        ExportStrategy.ADAPTER,
    ]


def test_default_laguna_contract_requires_48_layers_and_exact_architecture(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"laguna","architectures":["LagunaForCausalLM"],'
        '"num_hidden_layers":1,"quantization_config":'
        '{"quant_method":"fp8","weight_block_size":[128,128],'
        '"activation_scheme":"dynamic","ignored_layers":'
        '["model.layers.0.self_attn.o_proj"]}}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.layers.0.self_attn.o_proj.weight":"model.safetensors"}}'
    )
    save_file(
        {"model.layers.0.self_attn.o_proj.weight": torch.zeros((3, 5), dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )
    deltas = {
        "model.layers.0.self_attn.o_proj.weight": LoraDelta(
            torch.ones((2, 5), dtype=torch.bfloat16),
            torch.ones((3, 2), dtype=torch.bfloat16),
            1.0,
        )
    }

    with pytest.raises(ValueError, match="48 layers"):
        validate_laguna_checkpoint(tmp_path, deltas)
