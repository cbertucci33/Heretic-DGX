# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass


def _require_nonnegative_int(field_name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative int")


def _materialize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _materialize_json_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize_json_value(nested) for nested in value]
    return value


def _has_nonempty_quantization_parameter(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and any(
            _has_nonempty_quantization_parameter(nested) for nested in value.values()
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and any(
            _has_nonempty_quantization_parameter(nested) for nested in value
        )
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def canonicalize_quantization_config(config: Mapping[str, object]) -> str:
    """Serialize the complete quantization config for exact cross-rank comparison."""
    return json.dumps(
        _materialize_json_value(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class ModelLoadIdentity:
    """Exact checkpoint representation expected on both pipeline ranks."""

    base_model_id: str
    base_model_revision: str
    dtype: str
    quantization_config: str

    def __post_init__(self) -> None:
        for field_name in ("base_model_id", "base_model_revision", "dtype"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a nonempty string")
        if re.fullmatch(
            r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", self.base_model_revision
        ) is None:
            raise ValueError("base_model_revision must be a pinned commit hash")
        if type(self.quantization_config) is not str:
            raise TypeError("quantization_config must be exactly str")

        try:
            parsed = json.loads(self.quantization_config)
        except json.JSONDecodeError as error:
            raise ValueError("quantization_config must be canonical JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("quantization_config must be a canonical JSON object")
        quant_method = parsed.get("quant_method")
        if (
            type(quant_method) is not str
            or not quant_method.strip()
            or not any(
                _has_nonempty_quantization_parameter(value)
                for key, value in parsed.items()
                if key not in {"quant_method", "version"}
            )
        ):
            raise ValueError(
                "quantization_config must be a complete checkpoint configuration"
            )
        if canonicalize_quantization_config(parsed) != self.quantization_config:
            raise ValueError("quantization_config must use canonical JSON")


@dataclass(frozen=True, slots=True)
class LoadCommand:
    """Rank-0 request to load one pinned checkpoint representation."""

    command_id: int
    identity: ModelLoadIdentity

    def __post_init__(self) -> None:
        _require_nonnegative_int("command_id", self.command_id)
        if type(self.identity) is not ModelLoadIdentity:
            raise TypeError("identity must be exactly ModelLoadIdentity")


@dataclass(frozen=True, slots=True)
class LoadAcknowledgement:
    """Rank-1 declaration of the checkpoint representation it loaded."""

    command_id: int
    rank: int
    identity: ModelLoadIdentity

    def __post_init__(self) -> None:
        _require_nonnegative_int("command_id", self.command_id)
        _require_nonnegative_int("rank", self.rank)
        if type(self.identity) is not ModelLoadIdentity:
            raise TypeError("identity must be exactly ModelLoadIdentity")


def _revalidate_identity(identity: object) -> ModelLoadIdentity:
    if type(identity) is not ModelLoadIdentity:
        raise TypeError("identity must be exactly ModelLoadIdentity")
    return ModelLoadIdentity(
        base_model_id=identity.base_model_id,
        base_model_revision=identity.base_model_revision,
        dtype=identity.dtype,
        quantization_config=identity.quantization_config,
    )


def validate_load_acknowledgement(
    command: LoadCommand,
    acknowledgement: LoadAcknowledgement,
) -> None:
    """Accept an acknowledgement for the requested two-rank load."""
    if type(command) is not LoadCommand:
        raise TypeError("command must be exactly LoadCommand")
    if type(acknowledgement) is not LoadAcknowledgement:
        raise TypeError("acknowledgement must be exactly LoadAcknowledgement")
    _require_nonnegative_int("command_id", command.command_id)
    _require_nonnegative_int("command_id", acknowledgement.command_id)
    _require_nonnegative_int("rank", acknowledgement.rank)
    requested_identity = _revalidate_identity(command.identity)
    loaded_identity = _revalidate_identity(acknowledgement.identity)

    if acknowledgement.rank != 1:
        raise RuntimeError(
            "LOAD acknowledgement rank mismatch: "
            f"expected rank 1, received rank {acknowledgement.rank}"
        )

    if acknowledgement.command_id != command.command_id:
        raise RuntimeError(
            "LOAD acknowledgement command_id mismatch: "
            f"expected {command.command_id}, received {acknowledgement.command_id}"
        )

    for field_name in (
        "base_model_id",
        "base_model_revision",
        "dtype",
        "quantization_config",
    ):
        requested = getattr(requested_identity, field_name)
        loaded = getattr(loaded_identity, field_name)
        if loaded != requested:
            raise RuntimeError(
                f"LOAD acknowledgement {field_name} mismatch: "
                f"rank 0 requested {requested!r}, rank 1 loaded {loaded!r}"
            )
