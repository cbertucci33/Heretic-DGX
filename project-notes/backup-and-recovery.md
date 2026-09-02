---
title: Heretic DGX - Backup and Recovery
date: 2026-09-02
status: private-backup-complete-selective-temp-cleanup
project: distributed-heretic
repository: cbertucci33/Heretic-DGX
visibility: private
---

# Heretic DGX - Backup and Recovery

## Purpose

This note is the durable recovery boundary for the Distributed Heretic v25 work while redundant older/temp artifacts are removed. **Laguna and the active resumable v25 state remain on both DGX Spark nodes because the work is not complete.**

Related notes:

- [[Distributed Heretic Fork - Current Execution Record]]
- [[Distributed Heretic Fork Plan]]
- [[Heretic Plan]]
- [[Distributed Heretic Fork - Alien and Pre-Outage Execution Plan (2026-08-31)]] — historical only

## Repository target

```text
Owner: cbertucci33
Repository: Heretic-DGX
Visibility: private
Expected URL: https://github.com/cbertucci33/Heretic-DGX
Default branch: main
```

The GitHub repository is the durable source backup. It must contain the exact sanitized v25 source, tests, runtime documentation, deterministic manifests, canonical project-note snapshots, and recovery instructions. It must not contain model weights, virtual environments, caches, credentials, tokens, private keys, or raw agent transcripts.

## Verified software breakpoint

```text
Candidate: v25
Platform: Linux only, exactly two DGX Spark nodes
Focused standalone/runtime gate: 49 passed, 8 subtests passed
Full pytest gate: 209 passed, 57 subtests passed
Ruff: passed
compileall: passed
Release archive: not yet produced
Real Laguna mutation: not run
Standalone model: not produced
```

This is a verified software breakpoint, not a completed model release.

## Final artifact contract

The temporary BF16 LoRA adapter is not the end artifact. Completion requires a standalone abliterated Laguna FP8-compatible model that:

- loads without an adapter sidecar;
- contains no LoRA parameter keys;
- changes only all 48 `model.layers.N.self_attn.o_proj.weight` BF16 protected islands;
- preserves all FP8 tensors and `.weight_scale_inv` tensors byte-for-byte;
- preserves architecture, quantization metadata, tensor inventory, shapes, and dtypes;
- passes a complete SHA-256 manifest; and
- completes deterministic functional inference on exactly two DGX Spark Linux ranks.

No full BF16 Laguna materialization and no FP8 re-encoding are permitted.

## Identity and provenance anchors

```text
Official Heretic upstream reference:
bedb94ef117a271532ac2058447fbc165d5051bd

Laguna repository:
poolside/Laguna-S-2.1-FP8

Laguna revision:
06d71e91db70a11b08ee6a09c3c4818c85a61953

v24 immutable baseline SHA-256:
df4d2a9d726e5b4668f33b6f8b40adaea21d49cc4b5dd62ad3d6a1d12fcf8a21

Laguna 62-file manifest-tree SHA-256:
044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1

config.json SHA-256:
876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376

model.safetensors.index.json SHA-256:
aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7
```

## What the private repository must preserve

1. Complete v25 `src/`, `tests/`, project metadata, lock file, scripts, and runtime documentation.
2. A sanitized snapshot of the authoritative Obsidian notes.
3. v24 baseline manifest/hash metadata; the superseded archive itself may be included only if privacy and provenance scans pass.
4. Laguna identity manifest and redownload instructions, not 131 GB of model weights.
5. Exact gate outputs and a machine-readable backup manifest.
6. A recovery runbook that rebuilds the managed environment and retrieves the pinned Laguna revision.
7. Honest unverified-state declarations.

## Deliberate exclusions

- Laguna and DeepSeek weight files
- Python virtual environments and package caches
- CUDA/Hugging Face caches
- `.pytest_cache`, `.ruff_cache`, `__pycache__`, and `.pyc`
- GitHub tokens, API keys, SSH private keys, credentials, and environment secret files
- Raw Hermes delegation transcripts and broad session caches
- Host-specific temporary paths that are not needed for reconstruction
- Any archive or generated report that fails privacy/provenance review

## Selective DGX cleanup boundary

The private GitHub push and independent read-back protect the software history, but they do **not** authorize deletion of the active model or resumable workspace.

Preserve until Laguna work is complete:

- both `poolside/Laguna-S-2.1-FP8` checkpoint copies;
- the active v25 worktree;
- the managed Heretic test environment used by v25;
- v25 breakpoint evidence/manifests needed for immediate resumption;
- DeepSeek production checkpoint and unrelated launch/runtime assets;
- general system/CUDA/container tooling, unrelated repositories, and user data;
- canonical Obsidian notes and the private GitHub repository.

Remove only:

- superseded v20–v24 temporary worktrees/configs/evidence already preserved or obsolete;
- redundant GitHub backup staging and independent recovery-clone directories after read-back verification;
- temporary backup helper scripts and repository-scoped deploy keys after use;
- project caches and other reproducible scratch data that are not part of the active v25 state.

**Do not remove Laguna, the active v25 worktree, its managed environment, or v25 evidence merely to free the nodes for short-term work.**

## Recovery procedure

1. Clone the private repository onto a DGX/Linux node.
2. Verify the expected Git commit and repository tree hash recorded in this note and the repository manifest.
3. Recreate the managed Python environment from the locked project metadata.
4. Download `poolside/Laguna-S-2.1-FP8` at revision `06d71e91db70a11b08ee6a09c3c4818c85a61953` on both nodes.
5. Recompute all 62 file hashes and require manifest-tree identity `044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1` on both nodes.
6. Run the 209-test, Ruff, and compileall software gates.
7. Re-establish two-rank source/runtime/checkpoint identity and bounded NCCL preflight.
8. Schedule an explicit DeepSeek maintenance window before any Laguna load or mutation.
9. Resume at the real two-rank Laguna/temporary-adapter/standalone-model gates; do not repeat superseded v20–v24 work.

## Key learnings

- Intended behavior is the oracle; tests must not authorize semantic changes.
- A software-green distributed path is not runtime-qualified until the exact real operation runs on both ranks.
- Source, runtime, checkpoint, tokenizer, quantization, and rank identity must fail closed before mutation.
- The adapter is temporary; a standalone abliterated model is the final product.
- Laguna’s 48 allowed `o_proj` tensors are BF16 islands, so full-BF16 materialization and FP8 re-encoding are unnecessary.
- Verification must compare raw byte ranges outside permitted target tensors, not only decoded numeric equality.
- Preserve first, delete second, and read back the remote backup before clearing hosts.
- Temporary worktrees are not durable backups.
- Exactly two DGX Spark Linux nodes are the only accepted Heretic runtime; Windows is transport/note storage only.

## Remaining work after recovery

1. Recreate and verify the v25 environment from the private repository.
2. Produce a sanitized deterministic release artifact from the repository head if still desired.
3. Run the smallest real two-rank Laguna load and generation proof.
4. Produce and verify one real temporary adapter reconstruction.
5. Run the bounded ablation/evaluation process.
6. Materialize and verify the standalone Laguna FP8-compatible model.
7. Restore and functionally verify DeepSeek after every approved maintenance window.
