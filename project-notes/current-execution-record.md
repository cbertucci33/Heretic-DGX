---
title: Distributed Heretic Fork - Current Execution Record
date: 2026-09-01
updated: 2026-09-02
status: v25-private-backup-complete-selective-temp-cleanup
project: distributed-heretic
owner: Carmen Bertucci
agent: Hex
privacy: private-local
canonical_runtime_target: exactly-two-dgx-spark-linux-nodes
tags:
  - heretic
  - dgx-spark
  - distributed-inference
  - laguna
  - fp8
  - execution-record
---

# Distributed Heretic Fork — Current Execution Record

> [!important]
> This is the maintained execution record for the active fork. It supersedes stale implementation assumptions in [[Distributed Heretic Fork Plan]] and [[Distributed Heretic Fork - Alien and Pre-Outage Execution Plan (2026-08-31)]] while preserving those notes as historical planning provenance.
>
> **Backup/resumption:** the v25 software candidate is preserved in the private `cbertucci33/Heretic-DGX` repository. Cleanup is limited to redundant older/temp work. **Laguna, the active v25 worktree, its managed environment, and v25 evidence remain on the DGX nodes because work is not complete.** Resume from [[Heretic DGX - Backup and Recovery]]. DeepSeek remains production.

## Objective and literal scope

Make Heretic operate naturally across **exactly two DGX Spark Linux nodes** so larger compatible models can be processed collectively while preserving the input checkpoint's quantization:

```text
input quantization → Heretic processing → output with the exact same quantization
```

Required natural UX:

```text
heretic --config config.toml --cluster dgx-cluster.toml
```

Scope boundaries:

- **Linux on exactly two DGX Spark nodes is the only supported and accepted platform.**
- Windows is only the unavoidable Hermes SSH transport. Source edits, Git, archives, tests, review, model work, and acceptance probes run on DGX/Linux; Windows/WSL results never qualify or block the fork.
- All runtime, functional, dependency, CUDA, NCCL, model-loading, mutation, and export acceptance evidence must come from the exact candidate on DGX/Linux.
- The implementation must be model-agnostic rather than Laguna-specific.
- Cluster support launches and synchronizes ranks; it does not replace Heretic's normal model source/config path.
- No internal machine, user, address, interface, or private path identity may enter code, docs, tests, generated public configuration, or public artifacts.
- The pinned source checkpoint remains immutable; a derivative is written to a separate directory.
- BF16 LoRA adapter state is a temporary delta carrier and verification checkpoint, never the final deliverable.
- The end artifact is a standalone abliterated model that loads without the original adapter and preserves the source quantization exactly.
- Unsupported distributed mutation, evaluation-model switching, adapter-only completion, legacy full merge, and unsafe export paths fail closed before state mutation, broadcast, or writes.

## Current exact candidate

```text
Candidate: v25 software candidate
DGX/Linux worktree: /tmp/heretic-v25-work
Immutable baseline archive: heretic-dgx-exact-tree-v24.tar.gz
Baseline archive SHA-256: df4d2a9d726e5b4668f33b6f8b40adaea21d49cc4b5dd62ad3d6a1d12fcf8a21
Official upstream reference: bedb94ef117a271532ac2058447fbc165d5051bd
Runtime nodes: exactly 2 DGX Spark Linux nodes
Source state: extracted non-Git worktree; modified; not packaged
Qualification: software gates green after independent-review corrections
Release status: not packaged; no Laguna mutation or final-model runtime proof
```

v25 corrects the rejected v24 architecture rather than extending the v23 → v24 launcher-only delta. It adds topology-aware distributed LoRA math, coordinated adapter gathering, rank/source/checkpoint identity guards, fatal distributed lifecycle handling, and a bounded standalone Laguna exporter. Because the worktree is not a Git repository, exact review and packaging use deterministic source manifests and archive hashes rather than `git diff` claims.

## Independent v20 review

**Verdict: PASS — exactly-two-DGX runtime foundation.**

Independent exact-archive evidence:

- SHA-256 matched the expected v20 digest.
- Archive extraction was isolated and safe.
- All 102 extracted files remained byte-identical after review/testing.
- Required natural command reached deliberate cluster-file lookup without corrupting option values or constructing a model.
- Config behavior passed for defaults, spaced/equals forms, duplicates, precedence, malformed values, and synchronization exclusions.
- Prior fail-closed DGX guarantees remained intact.

Executed reviewer gates:

```text
Pytest: 157 passed
Subtests: 57 passed
Ruff: passed
Scoped Ty: exit 0 with 9 known torch.distributed stub warnings
compileall: passed
CLI help: passed
Ad-hoc argv/config probe: passed
```

Qualification boundary: this is a **software foundation PASS**, not Laguna runtime qualification and not adapter-export proof.

## Candidate lineage and key learnings

| Candidate | Verdict | Key result |
|---|---:|---|
| v14 | FAIL | Revision forwarding, startup bound, and artifact provenance blockers |
| v15 | FAIL | Distributed evaluation guard timing and privacy blockers |
| v16 | FAIL | Guard timing, revision, privacy, and unqualified export blockers |
| v17 | PASS foundation | Exact foundation accepted; live natural UX then exposed missing native `--config` |
| v18 | FAIL | Duplicate config inconsistency and natural-command argv corruption |
| v19 | REJECTED | Config semantics fixed, but natural command still corrupted `--cluster` |
| v20 | PASS foundation | Option-aware legacy positional-model rewrite preserves option values and legacy model UX |
| v21 | FAIL review | Sequential TP loading added, but descendant cleanup remained incomplete |
| v22 | PASS review; FAIL runtime | Guardian/lease cleanup passed independent review; natural worker launch could not import the source-tree supervisor |
| v23 | FAIL review | Absolute staging worked, but relative workdirs produced a duplicated effective import path after `chdir` |
| v24 | REJECTED | Full-codebase audit found distributed math, identity, lifecycle, export, privacy, and evidence blockers |
| v25 | software breakpoint GREEN | Review blockers corrected; 209 tests + 57 subtests, Ruff, and compileall pass; packaging and real Laguna proof remain pending |

Important lessons:

1. Unit/config tests can bypass the real console boundary; natural CLI behavior must be exercised through the actual entrypoint.
2. A repeated config option cannot select one file for `settings.config` and another for TOML values. v20 uses consistent final-value-wins semantics.
3. Legacy positional-model compatibility must understand which options consume scalar values; otherwise config/cluster paths can be mistaken for model identifiers.
4. Passing tests does not qualify a model runtime. Exact artifact review and real two-rank functional inference are separate gates.
5. Copied editable virtual environments do not prove provenance. Each candidate uses a fresh environment bound to the exact source tree/prefix/shebang.
6. Old candidate trees and archives should be deleted only after the replacement is integrity-verified and no old executable remains.

## Laguna FP8 baseline

Public model identity:

```text
Repository: poolside/Laguna-S-2.1-FP8
Revision: 06d71e91db70a11b08ee6a09c3c4818c85a61953
Files: 62
Bytes: 131287549859
FP8 shards: 49
Cross-node 62-file manifest-tree SHA-256: 044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1
Architecture: LagunaForCausalLM
Quantization: native FP8
Config TP-plan metadata: absent; topology support must be explicit in Heretic
```

The matching checkpoint exists on both nodes. The base manifest is the pre-run immutability anchor.

Narrow eventual intervention:

- Base Laguna checkpoint only
- No speculative decoding, DFlash, MTP, or vision wrapper
- MLP targets disabled
- Target BF16 `self_attn.o_proj` only
- Keep every original FP8 tensor untouched
- Temporary BF16 LoRA adapter parameters only during optimization/export handoff
- Final standalone derivative rewrites only the 48 protected BF16 `self_attn.o_proj.weight` tensors
- Every FP8 tensor and `.weight_scale_inv` tensor remains byte-identical
- No full-BF16 Laguna materialization and no FP8 re-encoding
- Text-only evaluation
- Exactly two NCCL ranks

## Runtime readiness before first v20 load

Privacy-safe readiness evidence:

```text
Available unified memory: approximately 117.5 GiB per node
Fresh swap-in/out activity: 0 KiB/s on both nodes
Active v20 processes: 0
Identified DeepSeek containers: 0
Disk available: approximately 193 GiB and 215 GiB
```

Fabric discovery found two shared RDMA subnets. Only one passed all launch-path checks:

- Explicit bound route and ping in both directions
- Coordinator-to-worker SSH
- Link-speed parity
- Interface-name parity

No attempt will be made to modify or "fix" the second rail because that is outside project scope.

## First v20 Laguna launch preparation

Cost/decision gate:

- **Decision:** whether v20 functions as an exactly-two-DGX Heretic runtime.
- **Question:** whether the unchanged Laguna FP8 checkpoint can load over two NCCL ranks and complete one real bounded text forward.
- **Minimum method:** Heretic's built-in startup check.
- **Bound:** one fresh process, eight new tokens, 30-minute ceiling, no retry inside a failed/poisoned process.

Fresh intended settings:

```text
Pinned model revision
Native checkpoint quantization path (`quantization = none` in Heretic, allowing the checkpoint's own FP8 recipe)
Dtype auto
Batch size 1
Max batch size 1
Response cap 8
Startup check enabled
```

### Corrected schema and live run

The initial invalid temporary cluster TOML was overwritten with the exact v20 schema:

```text
Top-level shared python/workdir/backend/port/timeout/interface
Exactly two [[nodes]] tables
Node 0 host matches the coordinator hostname
Node 1 host is the qualified worker SSH target
Distinct rank addresses
```

Validation passed through exact v20 on both nodes:

```text
NODE_1_FRESH_CONFIG_PARSE=passed
NODE_2_FRESH_CONFIG_PARSE=passed
VALID_FABRIC_COUNT=1
SHARED_EXECUTABLE_AND_WORKDIR=true
FRESH_PORT_SELECTED=true
```

Final preflight passed on both nodes:

```text
Exact v20 archive SHA: passed
Available-memory floor (110 GiB): passed
Active v20 ranks: none
Fresh swap activity: zero
Coordinator port availability: passed
```

A single bounded natural-command run started at **2026-09-01 21:56:27 CDT**:

```text
heretic --config <fresh Laguna TOML> --cluster <fresh DGX cluster TOML>
Tracker: proc_75931bab6ff9
Hard ceiling: 1800 seconds
Termination grace: 30 seconds
Automatic retry: disabled
```

Current state: **second Linux runtime attempt failed; no further retry authorized until both blockers are fixed and reviewed.**

### Second Linux runtime attempt — OOM and orphaned worker

The unchanged exact v20 candidate started both ranks and reached real distributed checkpoint loading, but did not complete the startup forward:

```text
Rank 0 NCCL initialization: passed
Rank 1 NCCL initialization: passed
Rank 0 checkpoint progress: 823/853 (96%)
Rank 1 checkpoint progress: 853/853 (100%)
Bounded startup text forward: not reached / unverified
Rank 0 exit: SIGKILL (-9)
```

Current kernel evidence proves the first causal failure was a global Linux OOM kill on the coordinator:

```text
Killed process: exact v20 Python rank
Anon RSS at kill: ~52.8 GiB
Constraint: global OOM, not a cgroup memory limit
Worker NVRM log: GPU allocation returned NV_ERR_NO_MEMORY
```

The worker's subsequent TCPStore/NCCL broken-pipe messages are consequences of coordinator death, not the initiating failure.

A second substantive runtime blocker was exposed: after the launcher killed its local SSH process, the remote `heretic.cluster_entry` remained alive. It was explicitly terminated and fresh scans verified zero v20 ranks on both nodes.

No automatic or manual model retry will occur from v20. After process cleanup, a synchronized host cache drop restored `118.3 GiB` MemAvailable on each node. The temporary secret-handling helper was removed.

Required before another model load:

1. ~~Deterministically fix and test remote-rank teardown so failure of either rank leaves zero remote processes.~~ Completed in v21 with a heartbeat lease supervisor and real two-node SSH lifecycle proof.
2. ~~Identify and address exact Transformers/DTensor load-time peak-memory behavior without changing Laguna's FP8 checkpoint identity or quantization.~~ Completed in v21 by setting `HF_DEACTIVATE_ASYNC_LOAD=1` only in DGX rank environments.
3. ~~Build a new exact candidate and run Linux gates on both DGX nodes.~~ Completed for exact v21.
4. Obtain independent exact-tree review of v21. **In progress.**
5. Only after review PASS, stage exact v21 as the runtime candidate and run one bounded two-node load/forward attempt.

## v21 runtime-blocker correction

Read-only investigation confirmed Transformers' default four-thread asynchronous loader remains active for pre-quantized FP8 weight-converter loads. It has no byte-based in-flight bound, allowing DTensor-local source slices, converted tensors, allocator reservations, and mmap-backed pages to overlap. Upstream explicitly recommends:

```text
HF_DEACTIVATE_ASYNC_LOAD=1
```

v21 injects this variable only into the exactly-two-DGX rank environment. It does not add `device_map`, `max_memory`, offloading, dequantization, or any model-path/revision/TP change.

The earlier PID-marker/second-SSH cleanup experiment was rejected after independent investigation showed its `finally` path could not guarantee cleanup after abrupt launcher death or a network partition. v21 instead adds a Linux remote rank supervisor with this contract:

1. Receive a fixed initial heartbeat before starting the rank.
2. Keep SSH stdin as a non-secret control lease.
3. Kill the rank process group after EOF, malformed control input, heartbeat expiry, or supervisor signal.
4. Escalate `TERM` to `KILL` after a bounded grace period.
5. Set Linux `PR_SET_PDEATHSIG=SIGKILL` before child exec and close the parent-death race.
6. Propagate the child exit status.

Verified lifecycle evidence:

```text
Focused launcher/supervisor tests on DGX/Linux: 12 passed
Real exactly-two-node SSH cleanup probe: passed
Remote stubborn child remaining after coordinator failure: false
Heartbeat-expiry regression: passed
Supervisor-SIGKILL parent-death regression: passed
No model, CUDA, NCCL, or checkpoint used by the lifecycle probe
```

Exact v21 artifact evidence:

```text
Files: 104
SHA-256: 73c0073bc762fec8f590a7514935b5075101560ea3de436937a38b7085e4ba14
Exact extraction/hash parity: passed on both DGX nodes
Pytest on node 1: 162 passed, 57 subtests passed
Pytest on node 2: 162 passed, 57 subtests passed
Ruff: passed on both nodes
Scoped Ty: exit 0 on both nodes with 9 known torch.distributed stub warnings
compileall: passed on both nodes
CLI help: passed on both nodes
Independent exact-archive review: pending
Laguna load/forward: not attempted
```

## Next gated plan

The first natural-command run exited with status 1 before its 30-minute ceiling:

```text
Rank 0 NCCL initialization: passed
Rank 1 NCCL initialization: passed
World size: 2
Distinct privacy-safe host fingerprints: passed
Rank 1 checkpoint load: all 853 weight entries read
Rank 1 bounded generation: failed before completion
Rank 1 error class: PermissionError
Both ranks stopped: passed
Stale v20 rank count on both nodes: zero
```

The first causal error was not a checkpoint permission or distributed-launch failure. Both nodes had the same stale root-owned, unwritable `~/.triton/cache`; the denied object was a not-yet-created child triggered during the first post-load generation/compilation.

Narrow repair, performed worker first and then coordinator:

```text
Only ~/.triton/cache ownership corrected to the runtime user
Recursive ownership/writability verification: passed on both nodes
No model/checkpoint/config identity changed
Temporary secret-handling helper: removed
```

After failed CUDA load, no process owned the missing memory (`0.4–0.5 GiB` total process RSS; `0.2 GiB` AnonPages; no CUDA process). A synchronized host cache drop recovered the known retained-memory condition:

```text
Node 1 MemAvailable: 118.3 GiB
Node 2 MemAvailable: 118.2 GiB
Remaining Cached: 0.2 / 0.1 GiB
Temporary secret-handling helper: removed
```

## Next gated plan

1. Read the exact v20 `cluster.py` schema and its tests.
2. Generate a fresh cluster TOML with the required explicit two-node collection.
3. Parse both the Laguna and cluster settings through exact v20 on both nodes.
4. Recheck archive SHA, process quiescence, memory/swap, and selected-port availability.
5. Launch exactly one bounded startup-check process with a 30-minute ceiling.
6. Preserve coordinator and worker logs separately.
7. Validate two structured rank-initialization records:
   - ranks 0 and 1
   - world size 2
   - NCCL backend
   - distinct privacy-safe host fingerprints
8. Validate one finite, non-empty text response plus model metadata.
9. Verify clean shutdown and absence of stale rank processes.
10. Produce a redacted durable evidence record and remove raw logs/configs containing private paths or topology.
11. Recompute the complete Laguna manifest on both nodes and require equality with the baseline.
12. Only after those gates pass, implement and independently review the attention-only component filter.
13. Only after that, run the smallest exactly-two-rank Heretic ablation and implement/prove adapter-only export.

### Bounded Linux retry

After the root-cause repair and retained-memory recovery, both DGX/Linux nodes passed the retry preflight:

```text
Exact v20 archive identity: passed
Fresh cluster parse: passed on both nodes
Fresh coordinator port: selected
Triton cache ownership/writability: passed on both nodes
MemAvailable: 118.2 GiB on both nodes
Active v20 ranks: zero
```

One retry began at **2026-09-01 22:10:37 CDT**:

```text
Tracker: proc_3ed204d3f7a7
Invocation: native heretic --config … --cluster …
Candidate: unchanged exact v20
Hard ceiling: 1800 seconds
Termination grace: 30 seconds
Automatic additional retry: disabled
Prior failed-run evidence: preserved
```

Current state: **started**, not yet functionally verified.

## v22 review PASS, runtime rejection, and v23 correction

Independent exact-v22 review returned **PASS** after reproducing all prior lifecycle failures against a fresh 104-file extraction. It verified control-EOF descendant cleanup, supervisor-SIGKILL private-lease cleanup, whole-session `SIGHUP` cleanup, bounded reaping, 40 immediate-signal race iterations per signal class, literal argv handling, normal output/exit propagation, and no persistent guardian/FD/zombie/descendant leaks.

Exact v22 identity and gates:

```text
SHA-256: 98bf15bb94ba4d5b1ce91a101f31be01abdf622a7b692a09153666aae04acf92
Files: 104
Pytest on each node: 165 passed, 57 subtests passed
Ruff: clean on both nodes
Ty: exit 0 with 3 known torch.distributed stub warnings
compileall and real CLI help: passed on both nodes
Real coordinator-to-worker SSH-death descendant cleanup: passed
```

The single authorized v22 natural startup attempt failed before model loading:

```text
DgxLaunchError: rank 1 exited with status 1
First causal error: No module named heretic.remote_rank_supervisor
Model checkpoint loading: not started
NCCL initialization: not reached
Automatic retry: disabled
Stale ranks after failure: 0 on both nodes
Runtime-port listeners after failure: 0 on both nodes
MemAvailable after failure: approximately 118 GiB on both nodes
```

Root cause: the cluster file selected the reviewed source worktree, but `build_rank_launches` did not set `PYTHONPATH=<workdir>/src`. The existing venv therefore searched its installed package rather than the reviewed tree and could not import the new supervisor module. This was an undeclared installed-package workaround in the natural source-tree UX, not a model or memory failure.

v23 adds exactly one production contract plus one regression expectation:

```text
Each rank environment sets PYTHONPATH=<configured workdir>/src
No host PYTHONPATH or secret is forwarded
Coordinator and worker bind to the same reviewed source tree
Normal installed-package fallback remains possible if workdir/src is absent
```

v23 evidence:

```text
Archive SHA-256: d18a9a03d5d5151b2a0f3116b1c7e2d16f547aea7049d2ca831ceed84809cf78
Archive files: 104
TDD regression: observed RED for missing PYTHONPATH, then GREEN after the one-line source fix
Pytest on each exact node tree: 165 passed, 57 subtests passed
Ruff: clean on both nodes
Ty: exit 0 with 3 known torch.distributed stub warnings
compileall and real CLI help: passed on both nodes
Independent exact-v23 review: in progress
Laguna v23 load/forward: not started
```

The byte-level pre-run Laguna payload anchor was recomputed on both nodes over the same 62 canonical files excluding `.cache`:

```text
Payload content manifest SHA-256: 2027323dd78008015dedff9a10a270bed3ce0963d3224bb1c970e23d5dee5fda
Cross-node equality: passed
```

## Pending acceptance gates

- [x] v20 exact artifact independently reviewed and accepted
- [ ] Correct fresh two-node cluster TOML parses on both nodes
- [ ] Real two-rank NCCL Laguna FP8 load
- [ ] Real bounded text forward
- [ ] Distinct rank-local structured evidence
- [ ] Clean shutdown without stale ranks
- [ ] Post-run checkpoint manifests equal the pre-run baseline
- [ ] Attention-only component filtering implemented and reviewed
- [ ] Smallest real Heretic ablation completed
- [x] Coordinated temporary adapter export and standalone materialization logic implemented and covered by Linux tests
- [ ] Real two-rank DTensor → PEFT adapter reconstruction proven on Laguna
- [ ] Standalone final model loads without adapter and runs deterministic two-rank inference
- [ ] Exact quantization-preservation contract proven end to end
- [ ] Final privacy/provenance review
- [ ] Commit only after required gates pass

## Durable evidence locations

Keep raw private runtime logs outside the vault. Store only redacted summaries here.

- v20 independent review summary: Hermes delegation cache, `subagent-summary-0-20260901_214610_770741.txt`
- v20 independent review transcript: Hermes delegation cache, delegation `deleg_35fc3d9d`
- Canonical reviewed source: exact DGX/Linux v24 archive identified below; Windows is SSH transport and note storage only.
- v25 Laguna materialization audit: Hermes delegation `deleg_26b2960f`
- v25 independent candidate review: Hermes delegation `deleg_b0da2584`

## Checkpoint cadence

Update this note whenever any of the following occurs:

- A candidate is accepted or rejected
- A substantive blocker is found or fixed
- A model load/forward starts, fails, or passes
- A quantization or immutability proof completes
- The active plan or scope boundary changes
- Before a long-running operation
- Before ending a work session

## 2026-09-02 file-complete v24 audit — rejection

**Decision: v24 is rejected. No further model load, ablation, export, or candidate patch is permitted until the architecture-level blockers below are corrected and independently reviewed.**

Review scope and identity:

```text
Archive: heretic-dgx-exact-tree-v24.tar.gz
SHA-256: df4d2a9d726e5b4668f33b6f8b40adaea21d49cc4b5dd62ad3d6a1d12fcf8a21
Members: 104
Production modules reviewed: 28 / 28 (7,695 lines)
Production source drift on both staged DGX trees: 0 files
Official upstream comparison base: bedb94ef117a271532ac2058447fbc165d5051bd
Delta: 53 files added, 8 modified, 0 upstream files removed
```

Executed gates:

- Pytest: 166 passed plus 57 subtests.
- Default Ruff: clean.
- Full-package Ty: exit 0 with nine PyTorch distributed stub warnings.
- Compile/import and packaged-wheel CLI help: passed.
- Offline wheel and sdist build: passed; archive membership/path checks passed.
- `ruff format --check`: failed for 24 files.
- `uv pip check`: flags NVIDIA's SBSA wheel tag as a platform mismatch, although the installed native library was independently verified as AArch64 ELF on both nodes.
- Invalid typed config through the real CLI: reproduced exit status 0, confirming fail-open automation behavior.
- No model was loaded and no GPU inference was run during this audit.

Release/runtime blockers:

1. **TP abliteration math is wrong for admitted layouts.** The live code discards topology after admission and applies one shard-local formula everywhere. Independent deterministic probes reproduced rowwise PRE normalization error `0.257786244`; colwise one-rank adapter error `1.279124022`; summing rank contributions reduced the latter to approximately `6e-8`. FULL normalization is also not globally equivalent because it performs local norms/SVDs.
2. **Exact quantization preservation is false for merged output.** BNB4 explicitly reloads and merges a full-precision base. Implicit FP8 falls through a branch with no scale regeneration or quantization inventory check. Adapter-only output is the only currently defensible contract.
3. **Distributed output is unusable.** Local adapter save is rejected; rank-0-only adapter upload calls PEFT 0.19.1 state gathering while rank 1 remains in the command loop, creating an unmatched collective risk.
4. **Ranks do not prove code/checkpoint identity.** No active pre-load comparison binds source tree, package stack, model revision/content, tokenizer/config, quantization metadata, TP plan, or adapter topology across ranks. The stronger identity/partition protocol is historical and unreachable.
5. **Default process/liveness contracts are incomplete.** Rank-0 descendants are not process-group contained; there is no end-to-end progress deadline; SSH lacks an explicit noninteractive bounded contract; operation failure can be masked by a second shutdown collective.
6. **Fatal errors can exit zero.** Invalid settings were reproduced through the real CLI with status 0. Action, environment, and reproduction-integrity failures are also suppressed in multiple paths.
7. **Reproduction can disable cluster mode.** Loading recorded settings replaces current operational settings; the excluded cluster field disappears, so a requested two-DGX reproduction can silently become local.
8. **Remote reproduction crosses into code execution.** Unvalidated HTTP(S) JSON can choose scorer plugin imports/files, whose module bodies execute before subclass validation. HTTP, redirects, response size, timeout, SSRF, schema, and signature controls are absent.
9. **Core model/scoring defects exist outside launch.** Evaluation-model switching keeps the old tokenizer/processor/revision; BF16 KL can become negative (independent probe: FP32 positive, BF16 `-2.51531600952e-05`); generic model support is contradicted by hard-coded layer/projection layouts; scorer/result pairing truncates silently.
10. **Tests overstate production proof.** Four heavily tested distributed modules are unreachable. The advertised tiny distributed verifier is same-host CPU/Gloo pipeline code that imports none of the production DGX path. `unittest` discovers 63 tests versus pytest's 166. Golden hashes are unioned across platforms and can pass with zero eligible fixtures.
11. **The exact source archive is not privacy-clean or release-ready.** Historical reports contain private paths/internal machine labels; WSL/Windows material remains in the DGX-only source archive; README/examples/CI remain upstream-local rather than documenting the fork. Built wheel/sdist scans were clean of those private reports, but the source archive is not publishable.
12. **Dependency metadata is not a complete DGX contract.** Torch/torchvision are unbounded in package metadata; pip does not consume `uv.lock`; no immutable DGX runtime/container, CUDA/NCCL compatibility matrix, machine-readable cluster manifest, or in-product environment preflight is supplied.

Required correction order:

1. Choose one active distributed architecture; remove/quarantine the unreachable protocol/partition/strategy/transport design.
2. Define the output contract as immutable quantized base plus coordinated BF16 adapter-only export; keep merged export fail-closed.
3. Implement topology-specific distributed math and prove it against the unsharded upstream oracle before any large model run.
4. Add a single in-product preflight/identity manifest covering two distinct physical nodes, source/runtime identity, exact checkpoint/config/tokenizer/quantization identity, SSH/NCCL/network readiness, and bounded startup phases.
5. Make rank lifecycle terminal on first failure, preserve the primary error, contain both process trees, and enforce explicit progress/deadline phases.
6. Repair CLI nonzero exit semantics and validate unknown keys/actions/config precedence.
7. Implement coordinated two-rank adapter gather with rank-0-only write followed by reload/manifest verification.
8. Replace synthetic/irrelevant gates with production-path two-node tests and an unsharded mathematical oracle.
9. Remove or isolate WSL/private historical material and regenerate a sanitized, deterministic release archive and provenance manifest.

That decision was correct at the v24 rejection boundary. It is superseded by the v25 software candidate documented below; no v25 release archive exists yet.

## 2026-09-02 v25 software breakpoint — GREEN and documentation pause

**Decision: pause implementation at this verified software breakpoint. Do not package, stop DeepSeek, start Laguna, or claim release readiness until documentation and the remaining release gates are complete.**

### Corrected final deliverable

The LoRA adapter is only an intermediate delta carrier. Completion means a **standalone abliterated Laguna FP8-compatible model** that:

- loads directly without the original adapter;
- contains no LoRA parameters or adapter sidecars;
- preserves the same tensor keys, shapes, dtypes, architecture, and quantization metadata;
- changes only `model.layers.N.self_attn.o_proj.weight` for all 48 Laguna layers;
- preserves all FP8 weights and all `.weight_scale_inv` tensors byte-for-byte;
- includes and passes a complete SHA-256 manifest; and
- passes deterministic functional inference across exactly two DGX Spark Linux nodes.

### Laguna materialization finding

The independent checkpoint audit established a safer path than the earlier FP8 re-encoding concept:

```text
Quantization: FP8, dynamic activations, 128×128 weight blocks
FP8 weight dtype: F8_E4M3 / torch.float8_e4m3fn
Scale dtype/suffix: F32 / .weight_scale_inv
FP8 weight/scale pairs: 33,024
Permitted targets: 48 / 48 self_attn.o_proj.weight tensors
Permitted target dtype: BF16 protected islands
Affected shards: 1, 2, 7, and 8 of 49
Largest rewritten shard: approximately 4.91 GB
```

Therefore standalone materialization applies the PEFT-compatible LoRA delta directly to the existing BF16 target tensors using bounded CPU arithmetic. It does not dequantize the model, materialize a full BF16 base, modify FP8 tensors/scales, or re-encode FP8.

Pinned public identity:

```text
Repository: poolside/Laguna-S-2.1-FP8
Revision: 06d71e91db70a11b08ee6a09c3c4818c85a61953
config.json SHA-256: 876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376
model.safetensors.index.json SHA-256: aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7
Cross-node 62-file manifest-tree SHA-256: 044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1
```

### Implemented v25 contracts

- Topology-specific distributed LoRA factor math is compared against an unsharded oracle.
- Source identity and full 40-hex model revision are required before distributed launch.
- Local and DGX runtimes declare distributed identity explicitly; action failures preserve the primary error.
- Distributed export resolves only to `standalone`; configured `adapter` and legacy `merge` completion fail closed.
- The temporary adapter rejects unsupported `fan_in_fan_out`, DoRA, and rsLoRA variants, non-finite factors, and active non-target deltas.
- Standalone source copying rejects symlinks and excludes Hugging Face `.cache/` bookkeeping.
- Rewritten safetensors shards preserve the original file and change only the byte ranges belonging to permitted BF16 targets.
- Verification checks all bytes outside target intervals, not merely numerical tensor equality.
- Temporary adapter material is consumed and removed after standalone export.
- SHA-256 manifest generation and verification are implemented.

### Independent review and closure

Independent review `deleg_b0da2584` initially returned **do not package** with four blockers:

1. no exact 48-layer/Laguna architecture requirement;
2. distributed adapter-only completion remained possible;
3. action failures could be masked by a missing runtime attribute; and
4. Hugging Face cache bookkeeping entered the release tree.

Each blocker now has a focused regression and implementation correction. Secondary review gaps were also hardened with byte-range preservation and rsLoRA rejection. The remaining real-model collective/export proof is not represented by a fake or unit test and remains an explicit runtime gate.

### Exact breakpoint gates

Executed on `gx10-01` through the managed DGX/Linux environment after the final semantic-contract changes:

```text
Focused standalone/runtime gate: 49 passed, 8 subtests passed
Full pytest gate: 209 passed, 57 subtests passed
Known warnings: 14 upstream torch.jit deprecation warnings
Ruff: all checks passed
compileall: passed
```

### Resource and packaging boundary

- The standalone derivative is approximately 122.27 GiB.
- Export fits current storage, but only approximately 70 GiB would remain on the measured coordinator posture.
- Do not create a second full local archive beside source and derivative; stream packaging elsewhere or free space first.
- The independent mixed-shard round-trip measured approximately 10.80 GiB peak RSS, so bounded CPU materialization is feasible.
- The v25 worktree is not yet sanitized or packaged. A credential/local-path scan and deterministic archive manifest remain required.

### Remaining gates before touching Laguna

1. Recheck review findings against the final staged tree and produce a sanitized deterministic v25 archive and hash.
2. Verify the exact archive on both DGX nodes; preserve source/checkpoint/runtime identity evidence.
3. Obtain explicit approval for the maintenance window before stopping DeepSeek.
4. Run the smallest bounded real two-rank Laguna load/generation/mutation trial with per-rank evidence and a 30-minute ceiling.
5. Prove real DTensor → temporary PEFT adapter reconstruction; do not infer this from recording fakes.
6. Materialize the standalone derivative without retaining a second full archive locally.
7. Verify exact tensor inventory, dtypes, permitted changed-target boundary, byte-preserved FP8/scale pairs, and `SHA256SUMS`.
8. Load the final directory without the adapter and run deterministic functional inference on both ranks.
9. Restore DeepSeek and verify per-rank identity, live health, text inference, vision inference, logs, and memory posture.

### Explicitly unverified

- No real Laguna ablation delta has been produced.
- No standalone Laguna derivative exists yet.
- No real two-rank collective adapter reconstruction has been exercised.
- No final standalone directory has loaded or generated.
- No sanitized v25 archive has been produced.
- DeepSeek has not been stopped or modified.

## 2026-09-02 preservation-first temporary cleanup

Stale v20–v24 temporary trees, worktrees, audit copies, configs, evidence folders, old archives, and generated caches were removed after preserving the minimum reconstruction package.

Durable evidence on both DGX nodes:

```text
/home/cb/heretic-artifacts/v25-breakpoint/heretic-dgx-exact-tree-v24.tar.gz
SHA-256: df4d2a9d726e5b4668f33b6f8b40adaea21d49cc4b5dd62ad3d6a1d12fcf8a21

/home/cb/heretic-artifacts/v25-breakpoint/laguna-base-sha256.json
SHA-256 / manifest-tree identity: 044b918f232f4efcd8acd8de167dd6b54461019737da08735408ea4387625fd1

/home/cb/heretic-artifacts/v25-breakpoint/PRESERVED.json
```

`gx10-01` additionally retains the v25 focused/full gate, Ruff, compileall, and clean-manifest records in the same durable directory.

Cleanup verification:

- `gx10-01`: `/tmp/heretic-v25-work` is the only remaining Heretic/Laguna project match; it is the active, unarchived v25 source and must remain until a sanitized v25 artifact exists.
- `gx10-02`: no matching Heretic/Laguna project temp paths remain.
- Disposable `__pycache__`, `.pytest_cache`, `.ruff_cache`, and `.pyc` files were removed from the active v25 tree.
- Windows local temp contains no matching Heretic, Laguna, or Qwen-Heretic scratch paths.
- `/home/cb/heretic-dgx-v20` remains because its managed Python environment is the current v25 test runtime.
- The Laguna source checkpoint, DeepSeek production checkpoint, canonical Obsidian notes, and cited Hermes review summaries/transcripts remain protected.
