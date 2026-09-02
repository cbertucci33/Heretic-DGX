# Distributed Heretic Fork Plan

> [!important]
> **Current execution status:** [[Distributed Heretic Fork - Current Execution Record]]. This note is retained as historical architecture/planning provenance; the current record supersedes stale runtime assumptions and status.

> **Status:** v25 software candidate implemented, green on DGX/Linux, and preserved in the private GitHub backup. Only redundant older/temp artifacts are being removed; Laguna and the active resumable v25 state remain on the DGX nodes. Packaging and real Laguna runtime qualification remain pending; no production outage has begun. Recovery source: [[Heretic DGX - Backup and Recovery]].
>
> **Primary target:** Run one exact `poolside/Laguna-S-2.1-FP8` model cooperatively across both networked GX10s for a narrow, reversible Heretic study.
>
> **Related plan:** [[Heretic Plan]]
>
> **Execution boundary:** [[Distributed Heretic Fork - Alien and Pre-Outage Execution Plan (2026-08-31)]] — records what can be completed on Alien, what can be staged while DeepSeek stays live, and the explicit gate for the first production maintenance window.

## 1. Decision and objective

Fork Heretic to add a two-node model runtime capable of:

1. Loading one Laguna FP8 checkpoint across both GX10s without materializing the full model on either node.
2. Collecting prompt-end residual means from every Laguna layer.
3. Computing refusal directions on rank 0.
4. Applying, resetting, and evaluating rank-local reversible LoRA interventions for each Heretic trial.
5. Running first-token KL, refusal, false-positive, and capability-preservation scorers.
6. Using a coordinated BF16 LoRA adapter only as a temporary delta carrier, then producing a standalone abliterated Laguna derivative tied to the exact FP8 base revision.

This is a distributed Heretic backend, not a generic distributed training framework. The first deliverable is intentionally Laguna-focused.

## 2. Locked constraints

- Use **FP8 or better** throughout residual extraction, intervention construction, optimization, and acceptance testing.
- Do not use BNB4 Laguna as a behavioral proxy. Laguna loops badly at 4-bit.
- Do not use Huihui's heavily quantized Laguna GGUF as a source checkpoint.
- Keep the exact FP8 base immutable.
- Export reversible adapter state during research, but never treat it as the final artifact.
- The final deliverable is a standalone abliterated Laguna directory that loads without an adapter; only the 48 protected BF16 `self_attn.o_proj.weight` tensors may change.
- No full-BF16 Laguna materialization, FP8 tensor/scale mutation, or FP8 re-encoding is permitted.
- Both GX10s cooperate on one Laguna instance. They are not independent search lanes.
- DeepSeek remains the production baseline and must be restored and functionally verified after every maintenance window.
- Do not begin a multi-night study until one complete distributed trial and adapter reload have passed.

## 3. Current source assessment

Inspected Heretic source commit:

```text
bedb94ef117a271532ac2058447fbc165d5051bd
```

Current Heretic assumptions:

- `main.run()` creates one in-process `Model`.
- Model loading uses one Transformers `from_pretrained()` call with a local `device_map` and `max_memory`.
- Inputs are moved to `self.model.device` and executed through the ordinary local `generate()` path.
- The complete layer stack is expected to be locally traversable as one `ModuleList`.
- PEFT discovers targets through the complete model's `named_modules()` and wraps one complete model.
- Reset, abliteration, generation, scoring, and export mutate or inspect local Python objects.
- No process-group, rank, RPC, or multi-node execution code exists in stock Heretic.

Useful existing seam:

- Scorer plugins already interact through a small context surface: responses, logits, and residuals.
- Optuna's trial controller is centralized and synchronous.
- Heretic constructs LoRA deltas directly and does not require distributed backpropagation or model optimizer-state synchronization.

Therefore, most scoring and study logic can remain on rank 0 while the model implementation moves behind a distributed runtime interface.

## 4. Recommended architecture

Use two-stage pipeline parallelism with one process per GX10.

```text
GX10-01 / rank 0 / coordinator
  tokenizer and prompt assembly
  datasets and scorer orchestration
  Optuna study and persistent trial journal
  embeddings
  first contiguous range of Laguna layers
  global residual-direction computation
  final adapter-manifest assembly

                 stage-boundary activations
                            ↓

GX10-02 / rank 1 / worker
  remaining contiguous Laguna layers
  final normalization
  lm_head
  final logits and token selection
```

The layer boundary must be selected from actual checkpoint tensor bytes and measured runtime headroom, not assumed to be exactly 24/24.

### Why pipeline parallelism

- Laguna's 256 routed experts are stored as large fused parameter tensors.
- A partially defined tensor-parallel plan could replicate expert weights and defeat the memory objective.
- Whole-layer ownership naturally partitions fused expert tensors.
- Heretic already reasons in global layer indices.
- Each rank can modify and reset only the adapters associated with its local layers.
- Residual running sums and adapter state are small enough to gather economically.

## 5. Runtime interface

Introduce a `ModelRuntime` abstraction with at least:

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

Implementations:

- `LocalModelRuntime`: preserves existing Heretic behavior.
- `DistributedPipelineRuntime`: rank-0 proxy controlling the two-node Laguna execution path.
- `DistributedPipelineWorker`: rank-1 command loop and local model stage.

Scorers and `Evaluator` should continue consuming the existing context API wherever possible.

## 6. Global layer catalog

Build a catalog that maps:

```text
global layer index
component name
full module name
owning rank
rank-local module reference
checkpoint tensor keys
serialized byte count
```

This replaces the assumption that every layer and module exists in one process.

Required invariants:

- Every owned base tensor loads exactly once.
- Every required checkpoint tensor has one declared owner, except deliberately replicated small state.
- No fused expert tensor exists on both ranks unless explicitly justified.
- Global layer ordering is stable across residual extraction, trial application, logs, and export.
- Module names in the final adapter match the exact FP8 base model.

## 7. Coordinator/worker protocol

Rank 0 is the sole authority for trials and sends explicit commands:

```text
LOAD
GENERATE
FIRST_TOKEN_LOGITS
CAPTURE_RESIDUAL_SUMS
RESET_ADAPTERS
APPLY_ABLITERATION
EXPORT_ADAPTER_STATE
HEALTH
SHUTDOWN
```

Each command carries:

- Monotonic command ID
- Study and trial ID where applicable
- Expected adapter generation/version
- Payload metadata and shape contract
- Timeout

Both ranks acknowledge the same command ID before the trial advances. A worker timeout, crash, or mismatched adapter generation aborts the current trial and the runtime rather than continuing with divergent state.

## 8. Stage-aware FP8 loading

The complete 122.258 GiB model must never materialize on either GX10.

Loader requirements:

1. Resolve and pin the exact Laguna FP8 revision.
2. Read the safetensors index without loading tensor payloads.
3. Calculate rank ownership and serialized bytes.
4. Construct stage modules on `meta` where supported.
5. Materialize only the embeddings/layers/norm/head owned by that rank.
6. Load only owned checkpoint tensors.
7. Verify missing and unexpected keys against the stage manifest.
8. Record post-load memory and confirm that no unowned expert tensors are resident.
9. Execute a local stage-level forward test before connecting the complete pipeline.

Laguna's Transformers configuration contains tensor- and pipeline-plan metadata. Treat that as a useful starting point, not proof that multi-node generation works without a custom executor.

## 9. FP8 and adapter compatibility gate

This is the first technical gate, before distributed generation.

On one rank with a small owned Laguna stage:

1. Load the exact FP8 checkpoint representation for that stage.
2. Execute one stage forward.
3. Discover one attention output projection (`o_proj`).
4. Inspect its concrete loaded module and weight types.
5. Attach either a standard PEFT rank-1 adapter or a purpose-built compatible wrapper.
6. Calculate Heretic's `lora_A = vᵀW` and `lora_B = -λv` construction in controlled precision.
7. Confirm that the adapter changes the stage output.
8. Reset the adapter to identity and confirm the original output returns within the agreed tolerance.
9. Export and reload the adapter state.

The Laguna FP8 configuration excludes attention projections such as `o_proj` from blockwise FP8 quantization, which is encouraging for the narrow initial target. The actual loaded runtime types still require verification.

## 10. Distributed execution modes

### 10.1 First-token forward

Used for residual collection and KL:

1. Rank 0 tokenizes and embeds the prompt batch.
2. Rank 0 executes its layers and retains local prompt-end residuals.
3. Rank 0 transfers the boundary activation plus mask and position metadata.
4. Rank 1 executes its layers, retains local prompt-end residuals, and computes final logits.
5. Rank 1 returns first-token logits or the required reduction to rank 0.

### 10.2 Residual means

- Accumulate per-layer sums locally in FP64.
- Track exact prompt counts.
- Gather only final sums/counts or means to rank 0.
- Preserve the embedding residual plus all layer residuals in global order.
- Convert final directions to FP32 for Heretic's calculations.
- Avoid gathering full prompt-by-layer residual tensors unless explicitly required for a bounded research diagnostic.

### 10.3 Greedy generation

- Each stage retains its own layer-local KV cache.
- Rank 0 owns prompt/input assembly and final output decoding.
- Rank 1 returns final logits or selected tokens for each step.
- Generation remains greedy and deterministic by contract.
- Stage-local cache lengths, positions, masks, and selected token IDs are checked at each step during the initial proof.

## 11. Distributed trial lifecycle

For each Optuna trial:

1. Rank 0 selects the global direction scope, direction index, layer weighting, and component parameters.
2. Rank 0 assigns a trial ID and adapter generation number.
3. Rank 0 broadcasts `RESET_ADAPTERS`.
4. Both ranks confirm all local LoRA-B tensors are identity/zero state.
5. Rank 0 broadcasts `APPLY_ABLITERATION` with residual directions and parameters.
6. Each rank updates only its owned modules using global layer indices.
7. Both ranks report applied module names and adapter-state hashes.
8. Rank 0 verifies the expected global module set.
9. Heretic runs the configured scorers through the distributed runtime.
10. Rank 0 records metrics, timings, memory, and worker health in the persistent study.
11. A failed rank or incomplete module set prunes/aborts the trial and stops the distributed runtime if state cannot be proven synchronized.

Only rank 0 runs Optuna and writes the primary study database.

## 12. Temporary adapter and standalone export

The search-stage export gathers only the temporary BF16 adapter and must not merge or dequantize Laguna in memory.

Each rank emits:

- Rank-local LoRA state keyed by the original full module names
- Global layer indices
- Base model repository and revision
- Runtime and package versions
- Rank-local state hashes

Rank 0 assembles:

- One standard adapter state file or sharded adapter set
- `adapter_config.json`
- Base-model revision manifest
- Dataset hashes and private-corpus provenance without private prompt contents
- Trial parameters and scorer results
- Global module inventory
- Per-rank source hashes

Reload verification must apply the combined adapter against the same FP8 base and reproduce the selected trial within tolerance. This proves the delta carrier, not the final deliverable.

After a winner is selected, standalone materialization must:

1. require exact `LagunaForCausalLM`, 48 layers, the pinned revision, and the exact FP8 metadata;
2. require all 48 intended `self_attn.o_proj.weight` targets to be BF16 protected/ignored layers;
3. apply PEFT-compatible CPU FP32 delta arithmetic with BF16 cast/add only to those BF16 targets;
4. copy every FP8 tensor and `.weight_scale_inv` tensor byte-for-byte;
5. preserve every byte outside the permitted target intervals in rewritten safetensors shards;
6. exclude caches, symlinks, adapter sidecars, and non-model leftovers;
7. generate and verify `SHA256SUMS`; and
8. load and generate from the final directory without the adapter.

## 13. Initial feature boundary

### Required for version one

- Two-node launch and health handshake
- Stage-aware partial FP8 loading
- First-token distributed forward
- Residual means
- First-token logits and KL
- Greedy response generation
- Built-in Heretic scorers needed by the Laguna study
- Synchronized reset and intervention application
- Persistent Optuna study
- Temporary adapter export/reload and standalone materialization verification
- Immediate failure reporting and bounded timeouts

### Deferred

- Generic full-precision merge/requantization paths
- LM Evaluation Harness integration
- Interactive streaming/chat UI
- Multimodal models
- Arbitrary model hot swapping
- Full residual plotting and geometric analysis
- Generic support for every Heretic architecture
- Vision grafting

## 14. Implementation phases and acceptance gates

### Phase 0 — Creator alignment and source baseline

Status: creator contacted on Reddit; awaiting response.

Tasks:

- Preserve the exact upstream source revision.
- Record creator feedback and architectural preferences.
- Confirm whether distributed work is planned upstream.
- Confirm whether the creator prefers a generic runtime abstraction or external Laguna proof first.
- Confirm contribution, test, and AGPL expectations.

Gate:

- A fork strategy is selected without conflicting with known upstream work.

Creator silence does not permanently block a private proof, but a reasonable response window should be allowed before committing to an upstream-facing architecture.

### Phase 1 — Local runtime abstraction

Tasks:

- Introduce `ModelRuntime` without changing stock single-node outcomes.
- Route scorer context through the runtime surface.
- Add fake/in-memory runtime tests for reset, apply, score, and export sequencing.
- Preserve existing CLI behavior by default.

Gate:

- Existing Heretic tests pass.
- A small local model completes the same baseline and one-trial flow before and after the refactor.

### Phase 2 — Laguna stage and FP8 adapter spike

Tasks:

- Parse Laguna's safetensors index and build stage manifests.
- Load a bounded stage without unowned tensors.
- Exercise one FP8 Laguna stage forward.
- Attach, change, reset, export, and reload one narrow adapter.

Gate:

- FP8 stage execution and reversible adapter behavior are proven.

Failure here stops the distributed fork until Laguna runtime/module compatibility is solved.

### Phase 3 — Small-model two-node pipeline proof

Tasks:

- Launch one process per GX10.
- Load a small architecture-compatible model split across nodes.
- Verify one first-token forward.
- Verify residual ordering and means against a single-node reference.
- Verify first-token logits within tolerance.
- Verify greedy generation.
- Verify synchronized adapter reset/application and temporary adapter export.
- Verify standalone materialization on a bounded quantization-preserving fixture.

Gate:

- The distributed result agrees with the single-node reference within explicit tolerances and survives worker shutdown/error tests.

### Phase 4 — Laguna FP8 first maintenance night

Prerequisite: stop DeepSeek using the approved maintenance procedure and reclaim memory on both nodes.

Tasks:

1. Load exact Laguna FP8 across both GX10s.
2. Verify per-rank tensor ownership and memory.
3. Run ordinary distributed generation and confirm no looping.
4. Collect bounded good/bad residual means.
5. Apply one narrow attention-output intervention.
6. Complete refusal, KL, and preservation scoring for one trial.
7. Export and reload the temporary adapter.
8. Materialize and verify the standalone derivative for the selected candidate.
9. Stop the experiment and restore DeepSeek.

Gate:

- One complete trial succeeds end to end, adapter reload reproduces the result, and production DeepSeek is restored and verified.

Any failure before the full trial causes an early abort and production restoration. Do not leave an unproductive run active overnight.

### Phase 5 — Bounded multi-night study

Tasks:

- Run a narrow attention/output-projection search.
- Persist every completed trial.
- Save Pareto candidates atomically.
- Use finite per-trial and outer wall-time limits.
- Alert immediately on OOM, process exit, synchronization mismatch, or stalled progress.
- Restore DeepSeek after each scheduled window until the distributed runtime has earned unattended confidence.
- Perform sequential independent-seed replication using the same two-node backend.

Gate:

- A stable candidate passes private false-positive, blind holdout, KL, capability-preservation, and real-workflow evaluation.

## 15. Verification strategy

### Unit and contract tests

- Global-to-local layer mapping
- Checkpoint-key ownership
- Byte-balanced partitioning
- Command serialization and monotonic IDs
- Adapter generation/version checks
- Rank-local reset and apply behavior
- Adapter state combination
- Worker-error propagation
- Resume after completed trials

### Numerical tests

On a small single-node reference model versus the distributed split:

- Prompt-end residuals by global layer
- Residual means
- First-token logits
- Greedy token sequence
- Adapter delta application
- Reset-to-baseline output
- Export/reload output

Each comparison must define dtype-aware tolerances. Do not claim bit-for-bit reproducibility across distributed devices without evidence.

### Laguna-specific tests

- Exact checkpoint revision and tensor inventory
- No unowned expert tensor residency
- FP8 kernel/runtime health
- No looping in stock distributed generation
- Correct global layer ordering
- Attention `o_proj` adapter compatibility
- Per-rank peak memory
- Cross-node transfer time and stall detection
- One complete Heretic trial

## 16. Runtime and resource boundaries

Planning payload:

```text
Laguna FP8 tensors:             122.258 GiB aggregate
Ideal two-way weight average:    61.129 GiB per rank
```

Actual partitioning must include:

- Embeddings and lm_head imbalance
- Norms and shared parameters
- Activations
- Stage-local KV caches
- Temporary FP32 target-matrix views
- Optional row-normalization and SVD intermediates
- PEFT/custom adapter bookkeeping
- CUDA/runtime allocations
- Unified-memory and OS headroom

A serialized-size fit is not a runtime-fit proof. Record actual per-rank residency and available memory at every gate.

## 17. Failure and recovery contract

Stop the distributed run when:

- Either worker exits or misses its command deadline.
- Rank adapter generations differ.
- Expected applied-module inventories do not match.
- An unowned expert tensor is found resident.
- FP8 execution falls back to an unapproved lower precision.
- Ordinary stock Laguna generation loops or produces invalid numerics.
- A trial cannot be exported and reloaded.
- Memory pressure threatens host stability.

Recovery:

1. Stop rank 0 and rank 1 Heretic processes.
2. Confirm no worker or child process remains.
3. Reclaim CUDA/unified memory using the proven GX10 procedure.
4. Restart DeepSeek worker/rank 1 and head/rank 0 in the approved order.
5. Verify both ranks and the live endpoint.
6. Require HTTP `200`.
7. Run real text inference.
8. Run real vision inference.
9. Confirm the expected cache capacity and memory profile.
10. Declare production restored only after all checks pass.

## 18. Creator outreach

The creator was contacted on Reddit on 2026-08-26. Record the response here when received.

Questions awaiting guidance:

1. Is distributed or backend-abstracted execution already planned?
2. Would upstream prefer a generic `torchrun` pipeline backend or a Laguna-specific external proof first?
3. Beyond responses, logits, residuals, reset, abliteration, and adapter export, what model-object invariants should scorers/plugins retain?
4. Is whole-model `PeftModel` wrapping essential, or is rank-local LoRA with standards-compatible combined export acceptable?
5. Have native or blockwise FP8 checkpoints been tested with Heretic?
6. What contribution structure, tests, and review sequence would make a future pull request acceptable?

### Response log

- **2026-08-26:** Initial Reddit outreach sent by Carmen. Awaiting response.

## 19. Go/no-go criteria

Proceed to implementation when:

- The source fork and exact Laguna revision are pinned.
- The creator's response is incorporated, or a reasonable response window has passed.
- The local runtime abstraction has a testable compatibility contract.
- The stage-aware loading approach has a bounded proof path.
- DeepSeek maintenance and restoration procedures are ready before any Laguna test.

Proceed to a Laguna maintenance window only when:

- Small-model two-node residuals, logits, generation, adapter reset, and export agree with a single-node reference.
- One Laguna FP8 stage can load and accept a reversible narrow adapter.
- Worker failure reliably aborts the complete run.

Proceed to multi-night optimization only when:

- Exact FP8 Laguna completes one distributed ordinary generation without looping.
- One complete Heretic trial succeeds.
- Temporary adapter export/reload reproduce the trial and the final standalone derivative passes its independent load/manifest boundary.
- DeepSeek restoration has been verified after the first maintenance gate.

## 20. Definition of success

The fork is successful when:

1. One immutable Laguna FP8 base is split across both GX10s without lower-precision substitution.
2. Distributed residual means and first-token logits are numerically validated.
3. Heretic can run resumable synchronized trials through the distributed runtime.
4. Trial adapters are reversible and exported without reconstructing the full base on one host.
5. A selected adapter improves the approved refusal/false-positive contract while remaining within KL and capability-preservation limits.
6. The selected delta is materialized as a standalone quantization-preserving model that loads and generates without an adapter.
7. Production DeepSeek can be restored and functionally verified after every research window.

The project is not considered successful merely because the checkpoint loads or a trial starts.
