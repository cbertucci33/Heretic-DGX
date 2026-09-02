# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from heretic.cluster_entry import (
    build_dgx_rank_identity,
    read_dgx_rank_environment,
    synchronize_dgx_settings,
)
from heretic.config import Settings


def valid_environment() -> dict[str, str]:
    return {
        "HERETIC_DGX_ACTIVE": "1",
        "HERETIC_DGX_ROLE": "coordinator",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "29500",
        "WORLD_SIZE": "2",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "HERETIC_DGX_BACKEND": "nccl",
        "HERETIC_DGX_TIMEOUT_SECONDS": "900",
    }


def test_reads_exact_two_rank_dgx_environment() -> None:
    environment = read_dgx_rank_environment(valid_environment())

    assert environment.rank == 0
    assert environment.role == "coordinator"
    assert environment.world_size == 2
    assert environment.backend == "nccl"
    assert environment.timeout_seconds == 900


def test_builds_structured_rank_identity_without_exposing_machine_identity() -> None:
    environment = read_dgx_rank_environment(valid_environment())

    identity = build_dgx_rank_identity(
        environment,
        machine_identity="private-internal-hostname",
    )

    assert identity["event"] == "dgx_rank_initialized"
    assert identity["rank"] == 0
    assert identity["world_size"] == 2
    assert identity["backend"] == "nccl"
    fingerprint = identity["host_fingerprint"]
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16
    assert "private-internal-hostname" not in str(identity)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HERETIC_DGX_ACTIVE", "0", "active"),
        ("WORLD_SIZE", "3", "exactly two"),
        ("RANK", "2", "rank"),
        ("LOCAL_RANK", "1", "LOCAL_RANK"),
        ("MASTER_PORT", "70000", "MASTER_PORT"),
        ("HERETIC_DGX_BACKEND", "mpi", "backend"),
        ("HERETIC_DGX_TIMEOUT_SECONDS", "0", "timeout"),
    ],
)
def test_rejects_invalid_rank_environment(name: str, value: str, message: str) -> None:
    values = valid_environment()
    values[name] = value

    with pytest.raises(ValueError, match=message):
        read_dgx_rank_environment(values)


def test_rejects_role_rank_mismatch() -> None:
    values = valid_environment()
    values["HERETIC_DGX_ROLE"] = "worker"

    with pytest.raises(ValueError, match="role"):
        read_dgx_rank_environment(values)


def test_rejects_credentials_in_master_address() -> None:
    values = valid_environment()
    values["MASTER_ADDR"] = "user:secret@10.0.0.1"

    with pytest.raises(ValueError, match="MASTER_ADDR"):
        read_dgx_rank_environment(values)


def test_coordinator_broadcasts_finalized_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["heretic"])
    settings = Settings(model="org/final-model", seed=123)
    payloads: list[object] = []

    def broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        payloads.append(payload[0])

    monkeypatch.setattr("heretic.cluster_entry.dist.broadcast_object_list", broadcast)

    synchronized = synchronize_dgx_settings(settings, rank=0)

    assert synchronized is not None
    assert synchronized.model == "org/final-model"
    assert synchronized.seed == 123
    assert isinstance(payloads[0], str)


def test_worker_receives_finalized_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["heretic"])
    serialized = Settings(model="org/final-model", seed=456).model_dump_json()

    def broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        payload[0] = serialized

    monkeypatch.setattr("heretic.cluster_entry.dist.broadcast_object_list", broadcast)

    synchronized = synchronize_dgx_settings(None, rank=1)

    assert synchronized is not None
    assert synchronized.model == "org/final-model"
    assert synchronized.seed == 456


def test_worker_receives_clean_coordinator_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "heretic.cluster_entry.dist.broadcast_object_list",
        lambda payload, src: None,
    )

    assert synchronize_dgx_settings(None, rank=1) is None
