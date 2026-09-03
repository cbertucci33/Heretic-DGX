# Two-node DGX release status

Last updated: 2026-09-03

## Verdict

The two-node runtime has an end-to-end engineering proof, but it is not yet
release-ready. The current release target is deliberately narrow: exactly two
DGX Spark nodes and the pinned Laguna S 2.1 FP8 checkpoint, exported as a
standalone checkpoint that preserves the source FP8 tensors.

Do not claim general distributed model-family or quantization support from the
current evidence.

## Evidence completed

- A two-rank Qwen2.5 fixture exercised coordinator launch, NCCL startup, prompt
  ingestion, residual calculation, two optimization trials, winner restoration,
  coordinated export, clean shutdown, artifact reload, and token generation.
- The real Laguna S 2.1 FP8 checkpoint passed identity, index, FP8 dynamic
  128x128, BF16 target-inventory, four-shard, and row-wise TP preflights.
- The latest Laguna run loaded all 853 checkpoint tensors on both ranks,
  initialized 48 `attn.o_proj` targets, loaded five good and five bad prompts,
  calculated residual directions, completed one optimization trial, restored
  it, and wrote a 123 GiB standalone checkpoint.
- Standalone export completed its integrated verification before reporting
  success: non-target file bytes were unchanged, rewritten safetensors changed
  only target intervals, target tensors matched the merge oracle, FP8 tensors
  remained unchanged, and `SHA256SUMS` was written and verified.
- The selected kernel intentionally operated only within its layer window:
  layers 29-47 changed and layers 0-28 remained identical to the source.
- The current working snapshot passes 51 focused tests plus 3 subtests on each
  DGX node when run with pytest.
- All five existing single-node golden fixtures (Qwen2.5, Gemma, Qwen3.5-MoE,
  Mistral-3, and MiniCPM5) complete their two-trial workflows, but their current
  output hashes differ from the stored baselines. Treat this as unresolved
  dependency/reproducibility drift, not as a passing gate.

Preserved local evidence on `gx10-01`:

- Successful rank logs: `~/.local/state/heretic/rank-logs/20260903T174337.*`
- Standalone output: `/home/cb/artifacts/laguna-s-2.1-fp8-heretic-proof1`
- Earlier bring-up logs: `/home/cb/artifacts/laguna-s-2.1-fp8-heretic-proof1.attempt*.log`

Earlier attempts failed during tokenizer/config compatibility, custom-code
trust, module-cache permissions, custom-kernel compatibility, and incomplete
Transformers FP8 tensor-parallel planning. Those failures are evidence of the
bring-up path, not evidence that the final export is corrupt; they produced the
compatibility and diagnostic fixes now in the branch.

## Repository checkpoint

- Code checkpoint `bc943fef89bf2e4b97e58d4f966787bacdaf70f1` is pushed on
  `rebuild/two-dgx-upstream-bedb94e`.
- Both DGX nodes are clean and identical at the active branch tip; documentation-
  only descendants may follow the code checkpoint. Verify exact commit and tree
  parity immediately before every launch.
- Both nodes pass pytest (51 tests plus 3 subtests), Ruff formatting, and Ruff
  lint from that commit.
- The source distribution and wheel build successfully.
- CI now installs pytest and runs the complete pytest suite.

## Required before release

1. Resolve the type-check/Python-floor diagnostics and golden output hash drift
   across all five existing model fixtures; then make the exact CI command set
   pass from a clean checkout.
2. Reclaim retained CUDA/UVM memory and verify actual available RAM on both
   nodes before any further model workload.
3. From the clean committed checkout, load the exported Laguna checkpoint in a
   fresh process and produce a real generation.
4. Repeat one bounded Laguna run from clean identical checkouts. Use at least
   two trials so optimizer comparison and winner restoration are exercised on
   the real model, then reload and generate from the new standalone output.
5. Install the built wheel in a clean environment and prove `heretic --help`
   plus the documented coordinator command.

## Test scope after the release gate

These are expansion tests, not blockers for the narrow Laguna release:

- another dense BF16/FP16 family;
- another MoE family;
- additional direction-index and kernel-window settings;
- additional quantization formats.

Each requires its own distributed load, ablation, export, reload, and generation
proof. Other quantization formats must not reuse the Laguna FP8 standalone
exporter without a format-specific preservation oracle.

## Operating rule

During development or model work, DeepSeek and every other main model stays
off. Completing work never authorizes restoring or starting a main model; that
requires Carmen's explicit instruction.
