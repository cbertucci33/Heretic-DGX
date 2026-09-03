# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any


_LAGUNA_S_2_1_TRUSTED_FILES = {
    "config.json": "876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376",
    "model.safetensors.index.json": (
        "aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7"
    ),
    "configuration_laguna.py": (
        "9446b4fca6f895bd0ed79d861f33447f8c231ba42b7c89cb4b4d25af3958c1fd"
    ),
    "modeling_laguna.py": (
        "765fd328542d176ff6a62ac814327b11a824df29bdca001d341e9a7c2fe9d876"
    ),
}


def matches_pinned_local_files(
    model: str, expected_files: Mapping[str, str]
) -> bool:
    """Return whether a local model has the exact reviewed metadata and code."""
    root = Path(model)
    if not root.is_dir():
        return False
    for name, expected_digest in expected_files.items():
        path = root / name
        if Path(name).name != name or not path.is_file() or path.is_symlink():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            return False
    return True


def is_pinned_laguna_checkpoint(model: str) -> bool:
    """Trust only the exact reviewed Laguna S 2.1 FP8 local code payload."""
    return matches_pinned_local_files(model, _LAGUNA_S_2_1_TRUSTED_FILES)


def build_model_load_kwargs(
    *,
    dtype: object,
    quantization_config: object | None,
    distributed: bool,
    model_commit: str | None,
    device_map: object,
    max_memory: Mapping[str, object] | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Build mutually exclusive local or fixed DGX TP loader arguments."""

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True if trust_remote_code else None,
    }
    if model_commit is not None:
        kwargs["revision"] = model_commit
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    if distributed:
        kwargs["tp_plan"] = "auto"
    else:
        kwargs["device_map"] = device_map
        kwargs["max_memory"] = (
            {
                int(key) if key.isdigit() else key: value
                for key, value in max_memory.items()
            }
            if max_memory
            else None
        )
    return kwargs
