# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from heretic.distributed_guards import (
    raise_if_distributed_failure,
    require_distributed_model_revision,
    require_local_model_operation,
)


@pytest.mark.parametrize(
    "operation",
    ["merged export", "upload", "streaming chat", "benchmark"],
)
def test_rejects_rank_local_operation_for_distributed_model(operation: str) -> None:
    with pytest.raises(RuntimeError, match=f"{operation}.*not available"):
        require_local_model_operation(distributed=True, operation=operation)


def test_allows_existing_local_model_operations() -> None:
    require_local_model_operation(distributed=False, operation="benchmark")


@pytest.mark.parametrize("revision", [None, "main", "abc123", "g" * 40])
def test_distributed_model_requires_full_hugging_face_commit(revision: str | None) -> None:
    with pytest.raises(RuntimeError, match="40-character hexadecimal"):
        require_distributed_model_revision("poolside/Laguna-S-2.1-FP8", revision)


def test_distributed_model_accepts_full_hugging_face_commit() -> None:
    require_distributed_model_revision(
        "poolside/Laguna-S-2.1-FP8",
        "06d71e91db70a11b08ee6a09c3c4818c85a61953",
    )


def test_distributed_action_failure_reraises_original_exception() -> None:
    error = ValueError("rank failed")
    with pytest.raises(ValueError) as raised:
        raise_if_distributed_failure(distributed=True, error=error)
    assert raised.value is error


def test_local_action_failure_remains_recoverable() -> None:
    raise_if_distributed_failure(distributed=False, error=ValueError("local action"))
