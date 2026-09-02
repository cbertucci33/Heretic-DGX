# Heretic DGX

Private backup and recovery repository for the Linux-only, exactly-two-DGX Distributed Heretic fork.

## Current state

- Candidate: **v25 software breakpoint**
- Upstream behavior oracle: `bedb94ef117a271532ac2058447fbc165d5051bd`
- Target model: `poolside/Laguna-S-2.1-FP8`
- Pinned model revision: `06d71e91db70a11b08ee6a09c3c4818c85a61953`
- Software gate: **209 tests + 57 subtests**, Ruff, and compileall passed on DGX/Linux
- Real Laguna mutation: **not run**
- Standalone abliterated model: **not produced**

This repository preserves source, tests, documentation, provenance, and recovery instructions. It intentionally excludes model weights, virtual environments, caches, credentials, private keys, and raw agent/session transcripts.

## Final artifact contract

The LoRA adapter is temporary. Completion requires a standalone abliterated Laguna FP8-compatible model that loads without an adapter, changes only all 48 BF16 `self_attn.o_proj.weight` protected islands, and leaves every FP8 tensor and `.weight_scale_inv` tensor byte-identical.

No full-BF16 Laguna materialization and no FP8 re-encoding are permitted.

## Repository layout

- `src/` — Heretic v25 implementation
- `tests/` — unit, topology, runtime, identity, and standalone-export tests
- `docs/` — runtime contract and preserved upstream documentation
- `project-notes/` — authoritative Obsidian plan/execution/recovery snapshots
- `provenance/v25-breakpoint/` — baseline archive, hashes, manifests, and gate evidence
- `RECOVERY.md` — host reconstruction and work-resumption procedure
- `BACKUP_MANIFEST.json` — exact tracked-file hashes and provenance anchors

## Important boundary

This is a private recovery source, not a finished release. Unit-test success does not prove real two-rank Laguna loading, mutation, DTensor-to-PEFT reconstruction, standalone materialization, or final inference.

See `project-notes/backup-and-recovery.md` and `project-notes/current-execution-record.md` before resuming work.

The original upstream README is preserved at `docs/upstream-readme.md`.
