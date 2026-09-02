# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import pytest

from heretic.distributed_strategy import DistributedStrategy, select_distributed_strategy


@pytest.mark.parametrize(
    "model_type",
    ["dense-decoder", "mixture-of-experts", "hybrid-state-space", "custom"],
)
def test_prefers_native_tp_without_model_family_allowlist(model_type: str) -> None:
    config = SimpleNamespace(
        model_type=model_type,
        base_model_tp_plan={"layers.*.projection": "rowwise"},
        base_model_pp_plan={"layers": (["hidden_states"], ["hidden_states"])},
    )

    assert select_distributed_strategy(config) == DistributedStrategy.TENSOR_PARALLEL


def test_rejects_pipeline_only_model_until_pipeline_runtime_exists() -> None:
    config = SimpleNamespace(
        model_type="architecture-not-known-to-heretic",
        base_model_tp_plan=None,
        base_model_pp_plan={"layers": (["hidden_states"], ["hidden_states"])},
    )

    with pytest.raises(ValueError, match="pipeline-parallel.*not implemented"):
        select_distributed_strategy(config)


@pytest.mark.parametrize("missing", [None, {}])
def test_rejects_model_without_a_distributed_execution_contract(missing: object) -> None:
    config = SimpleNamespace(
        model_type="custom",
        base_model_tp_plan=missing,
        base_model_pp_plan=missing,
    )

    with pytest.raises(ValueError, match="does not declare a tensor-parallel or pipeline-parallel plan"):
        select_distributed_strategy(config)


def test_diagnostic_reports_capabilities_without_model_specific_advice() -> None:
    config = SimpleNamespace(
        model_type="custom_decoder",
        base_model_tp_plan=None,
        base_model_pp_plan=None,
    )

    with pytest.raises(ValueError) as error:
        select_distributed_strategy(config)

    message = str(error.value)
    assert "custom_decoder" in message
    assert "Laguna" not in message
    assert "Qwen" not in message
