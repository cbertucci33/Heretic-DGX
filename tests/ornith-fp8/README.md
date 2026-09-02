# Ornith 1.5 9B FP8 WSL qualification fixture

This fixture qualifies the current Heretic fork against the public checkpoint `Sohailhosseini/Ornith-1.5-9B-FP8` on a WSL2 development host.

## Pinned source

- Repository: `Sohailhosseini/Ornith-1.5-9B-FP8`
- Revision: `d30f70d5d62c0b9d5a1d7f015b37513fc7aa0bdc`
- Weight file: `model.safetensors`
- Expected bytes: `11910387016`
- Expected LFS SHA-256: `bc0f5fd15250c26165470e6b2853721bb1ac2ad7c58d3f54ebbdd88019e1ddbe`
- Durable native path: `$HERETIC_MODEL_ROOT/Sohailhosseini/Ornith-1.5-9B-FP8`
- WSL path: `$HERETIC_WSL_MODEL_ROOT/Sohailhosseini/Ornith-1.5-9B-FP8`

The checkpoint uses Qwen3.5 multimodal architecture and compressed-tensors FP8 format version 0.18.0. The repository therefore includes `compressed-tensors~=0.18` as a runtime dependency.

## Fail-closed stages

1. Verify the exact Hub file inventory, weight byte count, and SHA-256.
2. Verify the frozen WSL build, CUDA smoke, tests, quality checks, CLI, sdist, and wheel.
3. Load the checkpoint without CPU offload. `device_map = { "" = 0 }` intentionally requires a real single-GPU fit.
4. Require Heretic model construction, one-token generation, layer discovery, and LoRA initialization.
5. Run one real optimization trial using the local prompt files in this directory.
6. Restore the selected trial and export the LoRA adapter through `ModelRuntime.save_adapter`.
7. Verify process exit, output inventory, and that GPU memory is reclaimed after process termination.

A load or optimization OOM is a reported capacity blocker, not permission to add hidden CPU offload, alter model semantics, reduce the checkpoint, or claim success from a partial stage. The fixture does not qualify two-node DGX behavior.
