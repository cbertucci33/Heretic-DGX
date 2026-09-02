# Recovery procedure

## Required topology

Distributed Heretic is accepted only on exactly two DGX Spark Linux nodes. Windows may transport files or host notes but must not perform source edits, Git operations, builds, tests, model work, or acceptance qualification.

## 1. Clone and verify

Clone the private `cbertucci33/Heretic-DGX` repository on a DGX node. Verify the expected commit and tree identifiers recorded in `BACKUP_MANIFEST.json` and the canonical Obsidian recovery note.

## 2. Recreate the environment

Use the project lock metadata to create a fresh managed environment. Do not restore a copied virtual environment. Run the complete pytest, Ruff, and compileall gates before any model operation.

## 3. Restore Laguna on both nodes

Download `poolside/Laguna-S-2.1-FP8` at exact revision:

```text
06d71e91db70a11b08ee6a09c3c4818c85a61953
```

Recompute the complete 62-file manifest on each node and require:

```text
044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1
```

Additional anchors:

```text
config.json:
876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376

model.safetensors.index.json:
aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7
```

## 4. Re-establish distributed identity

Require exact source, runtime, repository, revision, tokenizer, quantization, checkpoint, rank, and topology identity before model construction or mutation. Use exactly two ranks and fail closed on any mismatch or timeout.

## 5. Resume at the real-operation boundary

Do not repeat superseded v20–v24 work. The next unverified gates are:

1. real two-rank Laguna load and bounded deterministic generation;
2. attention-only mutation on the exact 48 BF16 `o_proj` islands;
3. real DTensor-to-temporary-PEFT adapter reconstruction;
4. bounded optimization/evaluation;
5. standalone materialization with byte-preserved FP8 weights/scales;
6. final load without adapter plus deterministic two-rank inference.

## 6. Production maintenance

DeepSeek remains production. Stop it only during an explicitly approved maintenance window. After every experiment, reclaim memory, restart both DeepSeek ranks, and verify per-rank identity, health, text inference, vision inference, logs, and memory posture.

## Never restore

- copied virtual environments;
- temp worktrees or caches;
- unpinned model revisions;
- adapter-only output as the final artifact;
- full-BF16 Laguna materializations;
- Windows/WSL project execution paths.
