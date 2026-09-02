#!/usr/bin/env python3
"""Fail-closed load-only verification for the pinned Ornith FP8 fixture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText


EXPECTED_WEIGHT_SIZE = 11_910_387_016
EXPECTED_WEIGHT_SHA256 = "bc0f5fd15250c26165470e6b2853721bb1ac2ad7c58d3f54ebbdd88019e1ddbe"
EXPECTED_LOADED_IDENTITY: dict[str, object] = {
    "model_class": "Qwen3_5ForConditionalGeneration",
    "model_type": "qwen3_5",
    "quant_method": "compressed-tensors",
    "parameter_devices": {"cuda:0": 960},
    "parameter_dtypes": {
        "torch.bfloat16": 760,
        "torch.float8_e4m3fn": 200,
    },
}


def validate_weight_file(
    weight_path: Path,
    *,
    expected_size: int = EXPECTED_WEIGHT_SIZE,
    expected_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> str:
    """Require the exact pinned Ornith weight artifact and return its SHA-256."""
    actual_size = weight_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"weight size mismatch: expected {expected_size}, actual {actual_size}"
        )

    digest = hashlib.sha256()
    with weight_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "weight SHA-256 mismatch: "
            f"expected {expected_sha256}, actual {actual_sha256}"
        )
    return actual_sha256


def validate_loaded_identity(result: Mapping[str, object]) -> None:
    """Reject any loaded model that is not the pinned Ornith FP8 identity."""
    mismatches = {
        field: {"expected": expected, "actual": result.get(field)}
        for field, expected in EXPECTED_LOADED_IDENTITY.items()
        if result.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(f"Ornith FP8 identity mismatch: {mismatches}")


def get_quant_method(config: Any) -> str | None:
    """Read quantization metadata from mapping or config-object forms."""
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, Mapping):
        return quantization_config.get("quant_method")
    return getattr(quantization_config, "quant_method", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model_path = args.model.resolve()
    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    weight_sha256 = validate_weight_file(weight_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.set_device(0)
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map={"": 0},
            local_files_only=True,
        )
        device_counts = Counter(str(parameter.device) for parameter in model.parameters())
        dtype_counts = Counter(str(parameter.dtype) for parameter in model.parameters())
        if any(not device.startswith("cuda:") for device in device_counts):
            raise RuntimeError(f"non-CUDA parameter devices detected: {device_counts}")

        result = {
            "model_class": type(model).__name__,
            "model_type": model.config.model_type,
            "quant_method": get_quant_method(model.config),
            "weight_size_bytes": weight_path.stat().st_size,
            "weight_sha256": weight_sha256,
            "parameter_devices": dict(sorted(device_counts.items())),
            "parameter_dtypes": dict(sorted(dtype_counts.items())),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_compute_capability": torch.cuda.get_device_capability(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        validate_loaded_identity(result)
        result["load_only_verification"] = "passed"
        print(json.dumps(result, sort_keys=True))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
