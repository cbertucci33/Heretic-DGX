# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify exact two-process split execution of a deterministic tiny Qwen2 model."""

from __future__ import annotations

import json
import socket
import tempfile
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask

_SEED = 20260901
_WORLD_SIZE = 2
_SEQUENCE_LENGTH = 4
_SPLIT_LAYER = 2

_init_process_group = cast(
    Callable[..., None],
    getattr(dist, "init_process_group"),
)
_send = cast(Callable[..., None], getattr(dist, "send"))
_recv = cast(Callable[..., None], getattr(dist, "recv"))
_destroy_process_group = cast(
    Callable[..., None],
    getattr(dist, "destroy_process_group"),
)


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_dropout=0.0,
        use_cache=False,
    )


def _model() -> Qwen2ForCausalLM:
    torch.manual_seed(_SEED)
    return Qwen2ForCausalLM(_config()).eval()


def _forward_layers(
    model: Qwen2ForCausalLM,
    hidden_states: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    cache_position = torch.arange(
        hidden_states.shape[1],
        dtype=torch.long,
        device=hidden_states.device,
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        model.config,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    layers = cast(Sequence[Callable[..., Any]], model.model.layers)
    for layer in layers[start:stop]:
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    return hidden_states


def _run_rank(rank: int, port: int, result_path: str) -> None:
    _init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=30),
    )
    try:
        model = _model()
        input_ids = torch.tensor([[1, 7, 11, 3]], dtype=torch.long)
        with torch.no_grad():
            if rank == 0:
                baseline_logits = model(input_ids=input_ids, use_cache=False).logits
                hidden_states = model.model.embed_tokens(input_ids)
                hidden_states = _forward_layers(
                    model,
                    hidden_states,
                    0,
                    _SPLIT_LAYER,
                ).contiguous()
                _send(hidden_states, dst=1)
                distributed_logits = torch.empty_like(baseline_logits)
                _recv(distributed_logits, src=1)
                max_abs_error = (
                    distributed_logits - baseline_logits
                ).abs().max().item()
                result = {
                    "backend": "gloo",
                    "boundary_shape": list(hidden_states.shape),
                    "byte_exact": torch.equal(
                        distributed_logits.contiguous().view(torch.uint8),
                        baseline_logits.contiguous().view(torch.uint8),
                    ),
                    "exact_equal": torch.equal(
                        distributed_logits,
                        baseline_logits,
                    ),
                    "logits_shape": list(distributed_logits.shape),
                    "max_abs_error": max_abs_error,
                    "processes": _WORLD_SIZE,
                    "rank0_layers": list(range(0, _SPLIT_LAYER)),
                    "rank1_layers": list(
                        range(_SPLIT_LAYER, model.config.num_hidden_layers)
                    ),
                }
                Path(result_path).write_text(
                    json.dumps(result, sort_keys=True),
                    encoding="utf-8",
                )
            else:
                hidden_states = torch.empty(
                    (1, _SEQUENCE_LENGTH, model.config.hidden_size),
                    dtype=model.dtype,
                )
                _recv(hidden_states, src=0)
                hidden_states = _forward_layers(
                    model,
                    hidden_states,
                    _SPLIT_LAYER,
                    model.config.num_hidden_layers,
                )
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states).contiguous()
                _send(logits, dst=0)
    finally:
        _destroy_process_group()


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="heretic-distributed-proof-") as directory:
        result_path = Path(directory, "result.json")
        mp.spawn(
            _run_rank,
            args=(_reserve_loopback_port(), str(result_path)),
            nprocs=_WORLD_SIZE,
            join=True,
        )
        if not result_path.is_file():
            raise RuntimeError("rank 0 did not produce distributed proof result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("byte_exact") is not True:
        raise RuntimeError(f"distributed logits are not byte-exact: {result}")
    if result.get("exact_equal") is not True:
        raise RuntimeError(f"distributed logits differ from baseline: {result}")
    if result.get("max_abs_error") != 0.0:
        raise RuntimeError(f"distributed logits have nonzero error: {result}")
    print(json.dumps(result, sort_keys=True))
    print("distributed_tiny_model_proof_passed=true")


if __name__ == "__main__":
    main()
