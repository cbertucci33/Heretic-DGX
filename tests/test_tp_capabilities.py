# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import pytest
from torch.distributed.tensor import Replicate, Shard

from heretic.tp_capabilities import inspect_lora_target_topologies


def target(plan: str | None, *, has_mesh: bool) -> SimpleNamespace:
    values: dict[str, object] = {"_hf_tp_plan": plan}
    if has_mesh:
        values["_hf_device_mesh"] = object()
    return SimpleNamespace(**values)


def dtensor_target(*placements: object, mesh_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        weight=SimpleNamespace(
            placements=placements,
            device_mesh=SimpleNamespace(size=lambda: mesh_size),
        )
    )


def test_accepts_observed_rowwise_dtensor_with_matching_model_plan() -> None:
    topologies = inspect_lora_target_topologies(
        {
            "model.layers.0.self_attn.o_proj": dtensor_target(Shard(1)),
        },
        model_tp_plan={"model.layers.*.self_attn.o_proj": "rowwise"},
    )

    assert topologies == {
        "model.layers.0.self_attn.o_proj": "rowwise",
    }


def test_rejects_replicated_dtensor_when_model_plan_declares_sharding() -> None:
    with pytest.raises(ValueError, match="disagrees with DTensor placement"):
        inspect_lora_target_topologies(
            {
                "model.layers.0.self_attn.o_proj": dtensor_target(Replicate()),
            },
            model_tp_plan={"model.layers.*.self_attn.o_proj": "rowwise"},
        )


def test_accepts_colwise_dtensor_with_matching_model_plan() -> None:
    assert inspect_lora_target_topologies(
        {"model.layers.3.mlp.up_proj": dtensor_target(Shard(0))},
        model_tp_plan={"model.layers.*.mlp.up_proj": "colwise"},
    ) == {"model.layers.3.mlp.up_proj": "colwise"}


def test_accepts_provably_replicated_dtensor_without_plan_entry() -> None:
    assert inspect_lora_target_topologies(
        {"model.shared_projection": dtensor_target(Replicate())},
        model_tp_plan={},
    ) == {"model.shared_projection": "replicated"}


def test_rejects_sharded_dtensor_without_model_plan() -> None:
    with pytest.raises(ValueError, match="missing its tensor-parallel plan"):
        inspect_lora_target_topologies(
            {"model.layers.0.self_attn.o_proj": dtensor_target(Shard(1))}
        )


def test_rejects_model_plan_that_disagrees_with_shard_dimension() -> None:
    with pytest.raises(ValueError, match="disagrees with DTensor placement"):
        inspect_lora_target_topologies(
            {"model.layers.0.self_attn.o_proj": dtensor_target(Shard(1))},
            model_tp_plan={"model.layers.*.self_attn.o_proj": "colwise"},
        )


def test_rejects_dtensor_on_non_two_rank_mesh() -> None:
    with pytest.raises(ValueError, match="requires a two-rank device mesh"):
        inspect_lora_target_topologies(
            {
                "model.layers.0.self_attn.o_proj": dtensor_target(
                    Shard(1), mesh_size=4
                )
            },
            model_tp_plan={"model.layers.*.self_attn.o_proj": "rowwise"},
        )


def test_accepts_model_agnostic_supported_target_topologies() -> None:
    topologies = inspect_lora_target_topologies(
        {
            "model.layers.0.attn.output": target("rowwise", has_mesh=True),
            "model.layers.0.mlp.output": target("colwise", has_mesh=True),
        }
    )

    assert topologies == {
        "model.layers.0.attn.output": "rowwise",
        "model.layers.0.mlp.output": "colwise",
    }


def test_rejects_missing_tp_metadata_in_distributed_model() -> None:
    with pytest.raises(ValueError, match="missing tensor-parallel metadata"):
        inspect_lora_target_topologies(
            {"model.layers.0.shared.output": target(None, has_mesh=False)}
        )


@pytest.mark.parametrize("plan", ["packed_rowwise", "grouped_gemm", "local"])
def test_rejects_tp_styles_peft_cannot_safely_adapt(plan: str) -> None:
    with pytest.raises(ValueError, match="unsupported tensor-parallel LoRA target"):
        inspect_lora_target_topologies(
            {"model.layers.0.projection": target(plan, has_mesh=True)}
        )


@pytest.mark.parametrize(
    ("plan", "has_mesh"),
    [("rowwise", False), (None, True)],
)
def test_rejects_incomplete_transformers_tp_metadata(
    plan: str | None,
    has_mesh: bool,
) -> None:
    with pytest.raises(ValueError, match="incomplete tensor-parallel metadata"):
        inspect_lora_target_topologies(
            {"model.layers.0.projection": target(plan, has_mesh=has_mesh)}
        )


def test_diagnostics_use_module_identity_not_model_family() -> None:
    with pytest.raises(ValueError) as error:
        inspect_lora_target_topologies(
            {"decoder.blocks.7.custom_projection": target("custom", has_mesh=True)}
        )

    message = str(error.value)
    assert "decoder.blocks.7.custom_projection" in message
    assert "Laguna" not in message
