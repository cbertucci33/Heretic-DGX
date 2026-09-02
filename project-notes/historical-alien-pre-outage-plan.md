---
title: Distributed Heretic Fork - Alien and Pre-Outage Execution Plan
date: 2026-08-31
status: historical-superseded-by-v25-execution-record
project: distributed-heretic-laguna
scope: alien-development-and-pre-outage-staging
production_service: DeepSeek-V4-Flash-V2
production_change_authorized: false
laguna_execution_authorized: false
owner: Carmen Bertucci
agent: Hex
tags:
  - heretic
  - laguna
  - dgx-spark
  - distributed-inference
  - alien
  - maintenance-planning
---

# Distributed Heretic Fork — Alien and Pre-Outage Execution Plan

> [!important]
> **Current execution status:** [[Distributed Heretic Fork - Current Execution Record]]. This note is retained as historical execution planning; the current record supersedes stale status and runtime assumptions.
>
> **2026-09-02 correction:** the active Linux-only v25 path no longer treats adapter-only export as completion. The adapter is temporary; the final deliverable is a standalone abliterated Laguna FP8-compatible model. Alien/Windows/WSL implementation assumptions in this historical note are not part of the accepted runtime path.
>
> **Backup/resumption:** use [[Heretic DGX - Backup and Recovery]] and the private `cbertucci33/Heretic-DGX` repository. Laguna and the active v25 DGX state are intentionally preserved; only redundant older/temp artifacts are removed. Do not reconstruct active work from the obsolete Alien/WSL procedures below.

> [!summary]
> Most of the Heretic distributed-backend engineering can be completed without interrupting DeepSeek. Alien can support approximately **65–75% of implementation work**. With non-disruptive staging and CPU/control-plane checks on the GX10s, approximately **85–90% of code, tooling, staging, and test infrastructure** can be completed before the first production outage.
>
> These percentages describe engineering scope, not runtime qualification. The remaining 10–15% contains the highest-risk proofs: exact Laguna FP8 compatibility, real two-GX10 GPU execution, per-rank memory fit, adapter apply/reset/export/reload, and one complete Heretic trial.

## Relationship to the architecture plan

This is the execution-boundary companion to [[Distributed Heretic Fork Plan]].

The architecture note remains authoritative for:

- the Laguna-focused two-stage design;
- runtime interfaces;
- global layer ownership;
- scoring and adapter semantics;
- failure contracts;
- long-study gates; and
- the ultimate definition of success.

This note answers two operational questions:

1. What can be implemented and tested on Alien?
2. How much can be completed before DeepSeek must be taken down?

Related context: [[Heretic Plan]].

## 1. Executive decision

Do **not** take DeepSeek down to begin development.

The intended sequence is:

1. Complete the source refactor, distributed protocol, partition tooling, synthetic tests, small-model proofs, and maintenance tooling on Alien.
2. Stage Laguna metadata, rank-specific checkpoint shards, source, environments, and control-plane tests on the GX10s while DeepSeek remains live.
3. Review a complete first-window runbook and rollback package.
4. Take DeepSeek down only when the remaining work is target-runtime qualification rather than open-ended software development.

The first outage should start with one reviewed launch path and explicit abort gates. It should not be used to discover basic protocol, serialization, partition, or adapter-state bugs.

## 2. Verified starting state

### 2.1 Heretic source baseline

The locally inspected Heretic source corresponds to the official `p-e-w/heretic` repository.[1]

```text
Repository: https://github.com/p-e-w/heretic.git
Default branch: master
Pinned commit: bedb94ef117a271532ac2058447fbc165d5051bd
Package version: 2.0.0.dev0
Local Git state at inspection: clean
```

The pinned commit is the current inspected upstream baseline for this plan.[2]

The existing Alien checkout is:

```text
C:\Users\carme\AppData\Local\Temp\heretic-src
```

That path is acceptable for inspection but **not** for implementation. It is under a temporary directory and must not become the durable source of the fork.

Before coding, create a durable native-Windows workspace such as:

```text
C:\Projects\heretic-distributed
```

The exact final path may change, but it must remain on native `C:\`, not WSL and not another drive.

### 2.2 Alien capacity and tools

Verified on 2026-08-31:

```text
GPU: NVIDIA GeForce RTX 5080
GPU memory: 16,303 MiB
System physical RAM: 34,043,850,752 bytes
Available RAM at inspection: 11,144,278,016 bytes
Git: available
uv: available
Docker: available
```

Interpretation:

- Alien is suitable for source development, container work, protocol simulation, small-model tests, and bounded adapter experiments.
- Alien cannot hold the complete Laguna FP8 payload in VRAM or system RAM.
- A successful Alien test does not qualify the GB10 ARM64 CUDA/FP8 runtime.

### 2.3 Production DeepSeek posture

Verified immediately before this plan was written:

```text
Endpoint: http://192.168.0.32:8304
Head health: HTTP 200
gx10-01 container: running
gx10-02 container: running
Image ID on both ranks: sha256:6e3727bc68c4761dca887c338dfcbedb11952453530032ab24ce929a32724664
```

No DeepSeek restart, container replacement, GPU workload, or production configuration change was made during planning.

### 2.4 GX10 storage posture

Verified free space:

```text
gx10-01: 216,605,470,720 bytes free
gx10-02: 383,500,111,872 bytes free
```

Current Laguna discovery:

```text
gx10-01: only /home/cb/models/hf/download-logs/Laguna-S-2.1-NVFP4-refresh.log found
gx10-02: no Laguna model path found
```

The exact Laguna FP8 checkpoint is therefore not presently staged as a complete usable model on either rank.

## 3. Scope allocation

### 3.1 What Alien can complete

Alien can complete approximately **65–75% of implementation work**:

- durable fork and source provenance;
- baseline tests and compatibility contract;
- runtime abstraction;
- local runtime preservation;
- coordinator/worker command protocol;
- fake and in-memory runtimes;
- global layer catalog;
- checkpoint-key ownership;
- byte-balanced partition planning;
- stage-aware loader logic using synthetic/small fixtures;
- adapter apply/reset/version/export state machines;
- small-model two-process pipeline proof;
- numerical comparison against a local reference;
- scorer contracts and trial persistence;
- watchdogs and structured logs;
- configuration schemas;
- documentation, manifests, and maintenance scripts;
- CI/static checks; and
- privacy and artifact-boundary checks.

### 3.2 What can be completed overall before an outage

By adding non-disruptive GX10 staging, approximately **85–90% of code, tooling, staging, and test infrastructure** can be completed before DeepSeek is stopped:

- exact Laguna configuration/tokenizer/index capture;
- exact tensor and shard inventory;
- rank-specific shard plan;
- controlled checkpoint staging;
- ARM64 runtime environment/image staging;
- import and CPU-only checks;
- identical source and manifest deployment to both ranks;
- real two-host control-plane handshake on isolated ports;
- worker timeout/abort/shutdown tests without CUDA model execution;
- partition and ownership manifest generation from Laguna's real index;
- complete first-window launcher;
- complete abort and cleanup scripts;
- complete DeepSeek restoration package; and
- reviewed acceptance and rollback checklist.

### 3.3 What requires the outage

DeepSeek must be stopped before the following can be honestly qualified:

- exact Laguna FP8 module loading on GB10;
- exact target-module inspection under the loaded runtime;
- PEFT or custom adapter compatibility with Laguna's concrete FP8 modules;
- real two-node GPU stage-boundary transfer;
- rank-local Laguna layer ownership;
- no expert-tensor duplication;
- rank-local KV/cache behavior;
- per-rank CUDA and unified-memory residency;
- temporary FP32/BF16 target-view headroom;
- distributed residual ordering and values;
- first-token logits and greedy generation;
- adapter apply/reset numerical behavior;
- adapter-only export and reload against exact Laguna FP8; and
- one complete bounded Heretic trial.

## 4. Alien implementation phases

## Phase A0 — Durable fork and immutable baseline

### Actions

1. Create the durable native-Windows repository path.
2. Clone from official upstream rather than treating the temp checkout as canonical.
3. Preserve upstream as `upstream` and configure a Carmen-owned fork as the publication remote when authorized.
4. Pin the starting commit.
5. Create the distributed-runtime development branch.
6. Capture:
   - Git revision;
   - remote URLs without credentials;
   - dependency lock;
   - Python version;
   - package version;
   - license/provenance; and
   - baseline test results.
7. Run the stock test suite before source changes.
8. Preserve a machine-readable baseline manifest.

### Success gate

- Durable native-`C:\` clone exists.
- Baseline commit is exact and clean.
- Existing tests have an honest pass/fail record.
- No implementation artifact depends on the temp clone.

## Phase A1 — Runtime abstraction without behavior change

### Actions

Implement or refine:

```python
class ModelRuntime:
    def get_model_metadata(...): ...
    def get_responses(...): ...
    def get_logits(...): ...
    def get_residuals_mean(...): ...
    def reset_adapters(...): ...
    def apply_abliteration(...): ...
    def save_adapter(...): ...
    def shutdown(...): ...
```

Add:

- `LocalModelRuntime` preserving stock Heretic behavior;
- fake/in-memory runtime for deterministic tests;
- runtime injection into `Evaluator` and scorer paths;
- CLI compatibility with the existing local default; and
- regression tests for operation ordering and exception handling.

### Success gate

A small local model completes the original baseline path through `LocalModelRuntime`, with defined numerical and behavioral tolerances and no distributed runtime enabled.

## Phase A2 — Coordinator/worker protocol

### Actions

Implement:

- rank-0 coordinator;
- rank-1 command loop;
- monotonic command IDs;
- trial IDs;
- adapter generation/version IDs;
- explicit health and readiness messages;
- operation deadlines;
- structured worker failures;
- synchronized reset/apply/export;
- clean shutdown; and
- rank-0 ownership of Optuna and trial persistence.

Test locally using two processes or containers.

Required injected failures:

- worker exits;
- worker misses deadline;
- duplicate command ID;
- stale adapter generation;
- mismatched applied-module inventory;
- malformed response;
- coordinator exit; and
- shutdown during an incomplete trial.

### Success gate

Every injected worker or protocol failure stops the complete logical run and produces a deterministic, attributable error rather than hanging or continuing with divergent state.

## Phase A3 — Global catalog, partition, and shard planner

### Actions

Build a stable catalog mapping:

```text
global layer → owning rank
global layer → local stage index
checkpoint tensor key → logical module
checkpoint tensor key → owning rank
physical SafeTensors shard → required ranks
adapter target → global layer and rank
```

Partition based on serialized tensor bytes and declared runtime overhead, not a guessed 24/24 layer split.

Handle explicitly:

- embeddings;
- final normalization;
- lm_head;
- shared parameters;
- fused expert tensors;
- shards containing tensors for both ranks; and
- replicated small state.

### Success gate

For a synthetic fixture and a small real checkpoint:

- every expected tensor has an owner;
- no forbidden tensor is silently replicated;
- every stage can enumerate required physical shards;
- missing/unexpected keys fail closed; and
- repeated planning produces a stable manifest.

## Phase A4 — Stage-aware loader

### Actions

Implement:

- meta-device construction where compatible;
- rank-owned module materialization;
- rank-owned shard loading;
- missing/unexpected tensor checks;
- unowned-parameter residency checks;
- stable module names for adapters; and
- explicit dtype/quantization reporting.

Use synthetic and small architecture-compatible checkpoints on Alien.

### Success gate

A small model loads as two logical stages without either stage retaining unowned large tensors, and the split model can be compared to a complete local reference.

## Phase A5 — Adapter state machine and export

### Actions

Implement:

- global intervention specification;
- rank-local target resolution;
- apply/reset generation IDs;
- per-rank state hashes;
- inventory reconciliation;
- adapter-only export;
- combined or sharded adapter manifest;
- exact base revision binding; and
- reload validation.

### Success gate

For a small reference model:

1. Baseline output is recorded.
2. A narrow intervention changes the expected output.
3. Reset returns within defined numerical tolerance.
4. Export and reload reproduce the intervention.
5. The exported artifact contains no base weights.

## Phase A6 — Small-model distributed pipeline proof

### Actions

Run a two-process proof with:

- embeddings and lower layers in stage 0;
- upper layers, final normalization, and head in stage 1;
- prompt-end residual capture;
- residual running means;
- first-token logits;
- greedy generation;
- stage-local KV state; and
- adapter apply/reset/export.

Compare against a single-process reference for:

- layer-ordered residuals;
- residual means;
- first-token logits;
- generated token sequence;
- adapter delta behavior;
- reset-to-baseline behavior; and
- export/reload behavior.

### Success gate

Every comparison has an explicit dtype-aware tolerance. No claim of bit-for-bit equivalence is made without evidence.

## Phase A7 — Project and maintenance infrastructure

Complete on Alien:

- config schema;
- CLI commands;
- deterministic test fixtures;
- scorer contracts;
- Optuna persistence and resume tests;
- structured event log;
- watchdog;
- artifact manifest;
- checkpoint hash handling;
- secret/privacy scan;
- launch scripts;
- cleanup scripts;
- DeepSeek baseline capture script;
- DeepSeek restore script; and
- first-window runbook.

## 5. Non-disruptive GX10 preparation

All work in this section must use isolated paths, ports, and processes and must avoid meaningful GPU allocation while DeepSeek is live.

## Phase G0 — Exact Laguna metadata inventory

Fetch first:

- `config.json`;
- tokenizer files;
- generation configuration;
- custom modeling/configuration source;
- SafeTensors index;
- revision metadata; and
- license/provenance files.

Do not begin a blind full-checkpoint copy.

Generate:

- total serialized tensor bytes;
- bytes by global layer;
- embeddings/head/shared-state bytes;
- tensor-key ownership candidates;
- physical shard requirements; and
- shards required by both ranks.

### Success gate

The exact checkpoint revision and complete tensor inventory are recorded before large payload staging begins.

## Phase G1 — Rank-specific shard staging

Because `gx10-01` has less free space, avoid duplicating the full checkpoint there unless a reviewed storage calculation establishes safe headroom.

Preferred strategy:

1. Compute the final logical partition.
2. Map each rank's owned tensors to physical shards.
3. Download only the required shard set to each rank.
4. Treat a shard needed by both ranks explicitly rather than assuming one owner per file.
5. Verify size and hash for every staged file.
6. Preserve a rank-local file manifest.

### Success gate

Both ranks possess every required file and no file is treated as verified without an exact manifest check.

## Phase G2 — ARM64 environment and source staging

Stage to isolated paths:

- the exact Heretic fork revision;
- dependency lock;
- ARM64 runtime image or environment;
- launch configuration;
- tests that do not allocate the production GPUs; and
- source/build manifests.

Run only:

- imports;
- Python compilation;
- CPU-only unit/contract tests;
- image metadata inspection; and
- non-GPU CLI validation.

A cross-built ARM64 image from Alien is only **build-qualified** until imported and exercised on a GX10.

### Success gate

Identical source, config, and dependency identities are present on both ranks, and CPU/import checks pass without touching DeepSeek's runtime.

## Phase G3 — Real two-host control-plane proof

Using isolated ports and CPU/Gloo or ordinary TCP, test:

- rank handshake;
- command sequencing;
- synthetic metadata messages;
- timeout propagation;
- worker exit;
- stale adapter generation;
- clean shutdown; and
- coordinator cleanup.

Do not interpret this as CUDA/NCCL or model-pipeline proof.

### Success gate

The real host-to-host control path behaves correctly under normal and injected-failure conditions while DeepSeek remains healthy.

## Phase G4 — Complete outage and restoration package

Before requesting the first maintenance window, prepare:

- current DeepSeek head/worker identity capture;
- exact stop order;
- CUDA/unified-memory reclamation steps;
- Laguna rank launchers;
- separate per-rank logs;
- startup deadlines;
- health/readiness checks;
- first-run prompts and expected artifacts;
- abort conditions;
- Heretic process cleanup;
- worker-first DeepSeek restoration;
- head DeepSeek restoration;
- per-rank container/image verification;
- HTTP readiness;
- real text inference;
- real vision inference;
- cache allocation/capacity verification; and
- fatal-log scans on both ranks.

### Success gate

The maintenance window can be executed from a reviewed checklist without writing or debugging foundational code during the outage.

## 6. First outage: exact qualification boundary

The first outage should be intentionally bounded.

## Q0 — Baseline and stop gate

1. Capture fresh DeepSeek identities, health, endpoint behavior, cache posture, and fatal-log baseline.
2. Stop in the reviewed order.
3. Confirm all production processes are gone.
4. Reclaim CUDA/unified memory using the proven GX10 procedure.
5. Confirm both ranks have sufficient available memory before Laguna starts.

Abort if baseline capture is incomplete or the hosts do not return to the expected free-memory posture.

## Q1 — Single-stage Laguna FP8 compatibility

Before launching the complete distributed path, establish on the target runtime:

- exact FP8 module types;
- custom Laguna kernel health;
- selected attention-output target type;
- adapter wrapping or custom-wrapper compatibility;
- narrow delta changes output;
- reset restores output within tolerance; and
- export/reload works against the exact base representation.

Abort if execution silently substitutes an unapproved precision or if apply/reset cannot be proved reversible.

## Q2 — Full two-stage Laguna load

Verify on both ranks:

- expected layer ownership;
- expected tensor inventory;
- no forbidden expert-tensor duplication;
- per-rank weight residency;
- CUDA/runtime allocations;
- unified-memory residency;
- activation and communication buffers;
- adapter state;
- temporary target-matrix views; and
- remaining host memory.

A checkpoint load alone is not success.

## Q3 — Ordinary distributed generation

Before Heretic optimization:

1. Run stock distributed generation.
2. Verify no looping.
3. Verify finite logits.
4. Verify correct global layer order.
5. Verify stage-local KV behavior.
6. Verify worker failures abort the run.

Abort if ordinary generation is not stable.

## Q4 — One bounded Heretic trial

Only after Q1–Q3 pass:

1. Collect bounded good/bad residuals.
2. Reconstruct the refusal direction.
3. Apply one narrow attention-output intervention.
4. Run refusal scoring.
5. Run first-token KL scoring.
6. Run capability-preservation scoring.
7. Export the adapter.
8. Reload it.
9. Reproduce the result.
10. Shut down cleanly.

Do not start a multi-night study in this window.

## Q5 — Restore DeepSeek

1. Stop both Heretic ranks and all children.
2. Confirm no research process remains.
3. Reclaim CUDA/unified memory.
4. Restore DeepSeek worker/rank 1.
5. Restore DeepSeek head/rank 0.
6. Verify expected image and container identity on both ranks.
7. Require endpoint HTTP `200`.
8. Run real text inference.
9. Run real vision inference.
10. Verify cache capacity/allocation and memory posture.
11. Scan both ranks for fatal logs.

Production is not considered restored merely because the HTTP endpoint opened.

## 7. Pre-outage completion checklist

DeepSeek should remain live until all of the following are complete:

- [x] Durable Alien fork exists on native `C:\`.
- [x] Exact upstream and working revisions are pinned.
- [x] Stock Heretic baseline is captured.
- [x] Runtime abstraction preserves local behavior.
- [x] Fake runtime tests pass.
- [ ] Coordinator/worker protocol tests pass.
- [ ] Worker failure aborts the complete logical run.
- [ ] Global layer catalog is stable.
- [ ] Tensor and physical-shard ownership is deterministic.
- [ ] Stage-aware loader passes synthetic and small-checkpoint tests.
- [ ] Adapter apply/reset/export/reload passes on a small model.
- [ ] Split small-model residuals/logits/generation are compared to a reference.
- [ ] Laguna exact revision and metadata are pinned.
- [ ] Laguna rank-specific shards are staged and hash-verified.
- [ ] ARM64 source/runtime identities match on both GX10s.
- [ ] Real two-host CPU/control-plane handshake passes.
- [ ] DeepSeek remains healthy after all non-disruptive staging.
- [ ] First-window launch and abort scripts are reviewed.
- [ ] DeepSeek rollback package is reviewed.
- [ ] First-window runtime ceiling is explicit.
- [ ] No unresolved foundational coding task requires the outage.

## 8. Go/no-go gates

### Gate P — Start implementation on Alien

Proceed when:

- durable workspace path is chosen;
- upstream baseline is pinned;
- no work depends on the temp clone; and
- the local compatibility test contract is written.

### Gate S — Begin non-disruptive GX10 staging

Proceed when:

- the exact Laguna revision is selected;
- partition tooling can consume the real SafeTensors index;
- isolated paths and ports are defined; and
- expected disk use is approved for each rank.

### Gate O — Request the first outage

Proceed only when:

- the full pre-outage checklist passes;
- small-model split execution agrees with the reference within defined tolerances;
- worker failure reliably aborts the run;
- the exact first-window commands are reviewed;
- DeepSeek restoration is ready before Laguna starts; and
- Carmen explicitly approves the maintenance window.

### Gate M — Begin multi-night optimization

Proceed only when:

- exact FP8 Laguna completes ordinary distributed generation without looping;
- one complete Heretic trial succeeds;
- adapter-only export and reload reproduce the trial;
- DeepSeek has been restored and fully verified after the first window; and
- Carmen explicitly approves the longer study.

## 9. Stop conditions

Stop and report rather than improvising if:

- exact Laguna FP8 requires an unapproved lower-precision fallback;
- a rank owns unexpected expert tensors;
- memory pressure threatens host stability;
- the two ranks disagree about adapter generation or applied modules;
- worker failure does not abort rank 0;
- ordinary generation loops or produces invalid numerics;
- apply/reset is not reversible within tolerance;
- adapter export/reload does not reproduce the result;
- DeepSeek restoration does not recover the approved per-rank and functional posture; or
- live DeepSeek blocks a pre-outage step that was assumed to be non-disruptive.

Do not move GX10 project work to another host or alter live production services to work around such a blocker without an explicit new approval.

## 10. Deliverables before the first outage

The pre-outage artifact set should include:

```text
source-manifest.json
dependency-lock and environment freeze
baseline-test-report.json
runtime-contract.md
protocol-schema.json
partition-manifest.json
rank-0-shard-manifest.json
rank-1-shard-manifest.json
small-model-reference-report.json
adapter-roundtrip-report.json
arm64-image-or-environment-manifest.json
two-host-control-plane-report.json
deepseek-pre-window-baseline.json
launch-rank-0.sh
launch-rank-1.sh
stop-heretic.sh
restore-deepseek.sh
first-window-checklist.md
first-window-result-template.md
```

No script needed for execution should remain in a temp directory.

## 11. Honest completion semantics

The pre-outage implementation may be described as:

- source-complete;
- unit-tested;
- protocol-tested;
- small-model qualified;
- staged on both GX10s; and
- ready for a target-runtime maintenance gate.

It must **not** be described as:

- Laguna-compatible;
- GB10 FP8-qualified;
- two-GX10 GPU-verified;
- memory-fit;
- adapter-compatible with exact Laguna FP8; or
- ready for a multi-night Heretic study

until the corresponding outage-gated proofs have actually run.

## 12. Execution record — 2026-08-31

### Phase A0 complete

Durable implementation workspace:

```text
C:\Projects\heretic-distributed
```

Pinned source posture:

```text
upstream: https://github.com/p-e-w/heretic.git
baseline: bedb94ef117a271532ac2058447fbc165d5051bd
branch:   feature/distributed-runtime
tracer:   6531a96a03c065dd6997f6774ba43590841f8cf7
parity:   682ac5f3016d5bef03ff05e64be4ad47dd2c344c
```

Durable baseline artifacts:

```text
C:\Projects\heretic-distributed\artifacts\baseline\source-manifest.json
C:\Projects\heretic-distributed\artifacts\baseline\environment-freeze.txt
C:\Projects\heretic-distributed\artifacts\baseline\baseline-test-report.json
C:\Projects\heretic-distributed\artifacts\baseline\wsl-cuda-environment-report.json
C:\Projects\heretic-distributed\artifacts\verification\wsl-reproducibility-workflow-report.json
C:\Projects\heretic-distributed\docs\runtime-contract.md
```

The environment uses CPython 3.12.11 and the committed `uv.lock` under `uv --frozen`. The committed lock is stale relative to `pyproject.toml`; it was deliberately not rewritten while capturing the upstream baseline.

The upstream offline baseline passed four unit tests, Ruff, `ty`, and the installed `heretic --help` entrypoint. The five model-producing upstream fixtures remain explicitly not run because they require model/dataset downloads and inference.

### WSL-first build baseline

Linux/GPU/backend development now defaults to WSL2 on Alien, with the canonical repository and promoted evidence retained on native `C:\` storage. The execution mirror is disposable and reproducible:

```text
canonical: C:\Projects\heretic-distributed
WSL mirror: /home/cb/work/heretic-distributed
WSL: Ubuntu 24.04.4 LTS, x86_64
Python: 3.12.3
uv: 0.11.11
PyTorch: 2.13.0+cu130
CUDA: 13.0
GPU: NVIDIA GeForce RTX 5080, compute capability 12.0
Windows NVIDIA driver: 610.88
```

A real CUDA tensor operation returned the expected value (`56.0`), proving PyTorch-level GPU access rather than only `nvidia-smi` visibility. At commit `682ac5f3016d5bef03ff05e64be4ad47dd2c344c`, the Linux mirror passed 10/10 unit tests, Ruff, `ty`, `compileall`, installed CLI help, and an sdist/wheel build whose wheel contains `heretic/runtime.py`. Disposable package output was removed.

The reproducible workflow and fail-closed verifier are durable at:

```text
C:\Projects\heretic-distributed\docs\wsl-development.md
C:\Projects\heretic-distributed\scripts\verify-wsl.sh
```

The verifier requires WSL2 and CUDA-enabled PyTorch, runs a real CUDA tensor operation, the full unit/quality/CLI gates, and disposable sdist/wheel verification. Its fail-closed checks remain active with `PYTHONOPTIMIZE=1`, and its regression tests reject WSL1 kernel identity and optimization-disabled Python assertions.

Verified WSL workflow commits:

```text
7b566428925de67e6615346ef2707570dfcfb83c  [verified] add residual mean runtime boundary
5fba235708d8b7400abcb7a9c909ed04b290e7f0  [verified] add WSL CUDA verification workflow
6df56af88dde1ccc9516c5dfab3b35b0fd28a974  [verified] record exact WSL workflow qualification
```

The exact committed workflow at `5fba235708d8b7400abcb7a9c909ed04b290e7f0` was checked out in a clean disposable WSL clone and passed with `PYTHONOPTIMIZE=1`: CUDA smoke `56.0`, 15/15 tests, Ruff, `ty`, `compileall`, CLI help, sdist/wheel build, wheel-content validation, and `wsl_verification_passed=true`. The commit tree was `aa16b9c6dd10684366681de427aadef87e8cb1f0`; the durable verification record is `artifacts/verification/wsl-reproducibility-workflow-report.json`.

The stale `.profile` source line for `/tmp/uv-0.11.28/env` was removed and a normal WSL login shell was verified. WSL x86_64 plus RTX 5080 materially improves Linux/CUDA interoperability, but does not prove GX10 ARM64, GB10 unified-memory, FP8, or two-node behavior; those remain target-hardware qualification gates.

### Phase A1 in progress

The first strict-TDD tracer bullet is committed in the durable branch. It adds:

- `ModelRuntime` scorer-read operations;
- behavior-preserving `LocalModelRuntime` delegation;
- runtime injection into plugin `Context` and `Evaluator`;
- unchanged concrete `Model` ownership for trial mutation, export, chat, merge, and benchmarking; and
- regression tests for context-local response caching plus uncached logits/residuals.

Current local/WSL gates after the stock-constructor compatibility correction:

```text
unittest:  19/19 passed on native Windows and isolated WSL2
focused stock compatibility: 4/4 passed
Ruff:      passed
ty:        passed
compileall: passed
CLI help:  passed
sdist/wheel build and wheel runtime-content check: passed
PYTHONOPTIMIZE=1 WSL verifier and CUDA smoke: passed
security scan of added lines: no sensitive-pattern findings
independent ad-hoc identity/exception/dispatch probe: passed; temporary verifier removed
independent fail-closed review: PASS
latest commit: 84c598e181f4b9bc53967f81bc13995e7c75473e
commit subject: [verified] restore stock runtime constructor compatibility
working tree immediately after commit: clean
```

#### Stock API drift discovered and corrected

An independent audit after `44da31cc12d902cb1c097e194eb59e860fdcb2c1` found that the runtime refactor was not yet strictly stock-compatible. The in-tree CLI still worked, but the public construction forms `Evaluator(settings, model)` and `Context(settings, model)` had effectively become runtime-only. Existing callers using `model=` would fail with `TypeError`; positional callers could pass a concrete `Model` that was later treated as a runtime; `Evaluator.model` and `Context._model` identity compatibility had also been lost. This was treated as a production compatibility blocker rather than documented away or hidden behind a fallback.

The correction at `84c598e181f4b9bc53967f81bc13995e7c75473e` is additive:

- stock positional and `model=` constructors accept the same concrete `Model` object;
- `Evaluator.model` and `Context._model` preserve identity with that object;
- the local model is wrapped exactly once by `LocalModelRuntime` for scorer reads;
- explicit `runtime=` injection remains available for future distributed runtimes;
- passing both `model` and `runtime`, or neither, fails explicitly rather than dispatching ambiguously;
- scorer contexts continue to receive the same runtime;
- mutation, randomness, trial sequencing, merge, export, chat, benchmark, and CLI paths remain on the existing concrete `Model` implementation.

Strict TDD evidence included RED failures for the stock constructors before the production correction, followed by focused and full GREEN gates. No test-, CI-, environment-, mock-, or caller-conditioned production branch was added.

Exact tree identities used by the gates:

```text
source-only native/WSL/Qwen tested tree: 17100b29480f3da9dc04a9ebe80e757a31549198
final five-file staged/review tree:       7ad40fd4a6a547550e0587d68f6c2b7ec16715ff
committed correction:                    84c598e181f4b9bc53967f81bc13995e7c75473e
base before correction:                  44da31cc12d902cb1c097e194eb59e860fdcb2c1
pinned stock behavior oracle:            bedb94ef117a271532ac2058447fbc165d5051bd
```

The staged and source-only trees differ only because the final staged tree also contains the completed contract document and preservation report. The independent reviewer verified that exact distinction and left the repository unchanged.

Verification artifacts:

```text
C:\Projects\heretic-distributed\artifacts\verification\runtime-tracer-report.json
C:\Projects\heretic-distributed\artifacts\verification\qwen25-local-runtime-parity.json
C:\Projects\heretic-distributed\artifacts\verification\residual-mean-tracer-report.json
C:\Projects\heretic-distributed\artifacts\verification\evaluator-constructor-runtime-report.json
C:\Projects\heretic-distributed\artifacts\verification\local-behavior-preservation-report.json
C:\Projects\heretic-distributed\docs\runtime-contract.md
```

Selected real-model parity gate after the correction:

```text
fixture: tiny-random/qwen2.5 @ 7a6a3128ee4137a248d6d1582824592b87a81647
host/runtime: Alien, Windows 11, PyTorch 2.13.0+cpu
workload: 2 trials; 5 good + 5 bad prompts; KeywordRate + KL divergence
trial KL: 0.0005 and 0.0006
export: merged model
expected files: 6
actual files: 6
hash result: 6/6 files match accepted upstream Windows/CI hashes
missing/extra files: none
original parity commit: 682ac5f3016d5bef03ff05e64be4ad47dd2c344c
compatibility-corrected verification report commit: 84c598e181f4b9bc53967f81bc13995e7c75473e
```

The non-PTY attempt completed model loading and both trials but hit `NoConsoleScreenBufferError` when Heretic eagerly constructed a `questionary` object despite configured noninteractive values. The PTY retry completed trial restoration, merge, and export. The underlying Heretic PID disappeared after export, while Hermes retained a stale PTY tracker; the tracker was terminated after confirming the real process was absent. Artifact parity is verified, but a clean natural PTY-tracker closure is explicitly not claimed. Generated model output, temporary patch files, and ad-hoc verifier scripts were removed after evidence capture.

The independent final review reran native gates and an exact-tree WSL2 verification with `PYTHONOPTIMIZE=1`, including 19/19 tests, CUDA smoke, Ruff, `ty`, compileall, CLI help, sdist/wheel build, and wheel-content checks. It also checked all six export hashes against the named accepted hash sources, the contract text, qualification boundaries, and honest PTY disclosure. Verdict: PASS with no hidden fallback, import cycle, typing/runtime defect, security issue, or acceptance-gaming test found.

No GX10 service, DeepSeek process, remote model checkpoint, or production runtime was changed. The selected local scorer-read runtime path is now stock-API compatible and exact-hash verified.

#### Adapter state runtime boundary completed

The second strict-TDD runtime tracer bullet is committed at:

```text
commit: 0fd9b383523fe524c4975004d8798f7c026d184e
subject: [verified] add adapter state runtime boundary
base: 84c598e181f4b9bc53967f81bc13995e7c75473e
pinned stock behavior oracle: bedb94ef117a271532ac2058447fbc165d5051bd
source-only native/WSL/Qwen tested tree: 779ba392fb37cc593f4581deed2a150eec8a5fe8
final five-file staged/review tree:       7c2f8a0646075c8148a1bd32f429edde5d1345aa
```

The slice adds these abstract operations and direct local mappings:

```text
ModelRuntime.reset_model()       -> Model.reset_model()
ModelRuntime.abliterate(...)     -> Model.abliterate(...)
```

`LocalModelRuntime` keeps the identical concrete `Model`. It delegates each operation exactly once, preserving operation order, the original residual-direction tensor, direction index, parameter mapping, and exception object. No copy, transform, retry, fallback, exception translation, or test/environment-conditioned branch was added.

Exactly five existing post-composition orchestration calls changed only their receiver from `model` to `runtime`:

1. evaluate-model reset;
2. per-trial reset;
3. per-trial abliteration;
4. selected-trial/post-merge restoration reset;
5. selected-trial/post-merge restoration abliteration.

Model loading, layer/component discovery, randomness, adapter math, scorer sequencing, merge/export, chat, and benchmark behavior remain on their existing paths. Export remains concrete-model-owned until its own future tracer bullet.

Strict TDD and focused evidence:

```text
RED: LocalModelRuntime lacked reset_model and raised AttributeError
GREEN unit tests:
  - exact reset then abliteration delegation and argument identity
  - unchanged exception-object propagation
focused ad-hoc verifier: passed
AST orchestration counts:
  runtime.reset_model = 3
  runtime.abliterate  = 2
  model.reset_model   = 0
  model.abliterate    = 0
temporary hermes-verify scripts: removed
```

Native and exact-tree WSL2 gates:

```text
native unittest: 21/21 passed
focused runtime tests: passed
Ruff / ty / compileall / CLI help: passed
sdist + wheel build and wheel runtime-content check: passed
WSL2 PYTHONOPTIMIZE=1 unittest: 21/21 passed
WSL2 RTX 5080 CUDA smoke: passed; sum 56.0
WSL2 package and quality gates: passed
terminal marker: wsl_verification_passed=true
```

The real `tiny-random/qwen2.5` fixture was rerun through the actual PTY application path. It completed two trials, reset → abliterate → evaluate sequencing, selected-trial restoration, merge, post-merge reset/restoration, and merged export. Exactly six expected files were emitted; all six matched the checked-in accepted Windows hash set, applicable CI hashes overlapped, and there were no missing or extra files. Final observed metric remained Keywords `0/5`, KL divergence `0.0006`. The real Heretic process was absent after export; only the stale Hermes PTY tracker required termination, so natural tracker closure is not claimed. Generated model output and the temporary transfer patch were removed.

Durable evidence and implementation references:

```text
C:\Projects\heretic-distributed\artifacts\verification\adapter-state-runtime-report.json
C:\Projects\heretic-distributed\docs\runtime-contract.md
C:\Projects\heretic-distributed\src\heretic\runtime.py
C:\Projects\heretic-distributed\src\heretic\main.py
C:\Projects\heretic-distributed\tests\test_runtime.py
C:\Projects\heretic-distributed\scripts\verify-wsl.sh
```

Independent fail-closed review of exact staged tree `7c2f8a0646075c8148a1bd32f429edde5d1345aa` reran 13 focused tests, 21/21 full native tests, all native package/quality gates, the exact retained WSL verifier, and the real tiny-Qwen artifact checks. It compared all five mutation calls by AST and found only receiver changes with identical argument expressions. Verdict: **PASS**, with no behavior-preservation, evidence-integrity, security, hidden-fallback, or acceptance-bypass blocker.

One non-blocking forward-compatibility caveat is recorded rather than hidden: `typing.get_type_hints(ModelRuntime.abliterate)` currently raises `NameError` because `AbliterationParameters` is imported only under `TYPE_CHECKING`. Static `ty` passes and no current production path introspects this method. Resolve this in a separate test-first slice if runtime annotation introspection becomes part of the interface; do not silently broaden the reviewed adapter-state commit.

The qualification boundary remains explicit: Alien local/WSL2 x86_64 behavior is verified for the exact tested tree, but the distributed protocol, GX10 ARM64/GB10, Laguna FP8, and two-node behavior are not implemented or qualified. DeepSeek and both GX10s remained untouched.

## 13. Recommended next action

1. Decide the next isolated runtime tracer bullet: model metadata/discovery or adapter export/save, with a RED behavior oracle before production edits.
2. Add distributed protocol work only after local ownership boundaries are complete and small-model parity continues to pass.
3. Keep DeepSeek live; do not begin Gate S or request an outage.

## Sources

[1] [p-e-w/heretic repository](https://github.com/p-e-w/heretic)

[2] [Heretic pinned upstream commit bedb94e](https://github.com/p-e-w/heretic/commit/bedb94ef117a271532ac2058447fbc165d5051bd)
