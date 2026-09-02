# Heretic Plan

> **Status:** Laguna v25 software breakpoint is green and backed up to the private `cbertucci33/Heretic-DGX` repository. Cleanup is selective: Laguna, the active v25 worktree, its managed environment, and v25 evidence remain because work is not complete. No model modification has started; DeepSeek remains production. Recovery source: [[Heretic DGX - Backup and Recovery]].
>
> **Distributed implementation:** [[Distributed Heretic Fork Plan]] — creator contacted on Reddit; awaiting response.
>
> **Purpose:** Build a private, local daily-driver model that answers legitimate software-development, security, authorized personal-automation, mental-health, relationship, and consensual adult sexual-wellness questions directly and competently, without reflexive refusals, judgment, moralizing, or coddling.
>
> **Core constraint:** Abliteration is not a reason to accept a worse or larger model. A replacement for the current DeepSeek deployment must be demonstrably better for Carmen's real workloads and have the same or smaller actual loaded footprint across both GX10s.

## 1. Operating intent

This project is about eliminating false-positive guardrail behavior in a private local research environment. It is not motivated by illicit use.

Carmen uses local models because important work and personal research should remain private. The desired model should make it easy to:

- perform internal software development and defensive security work;
- delegate ordinary authorized digital tasks that help manage work and personal life;
- research mental health and personal wellbeing without ordinary discussion being misclassified as self-harm intent;
- research consensual adult sexual health, intimacy, and relationship wellbeing without unnecessary refusals or moralizing;
- ask technically or personally sensitive questions without sending private material to a hosted inference provider;
- receive direct, evidence-oriented answers without being judged or coddled.

The target behavior is **useful contextual judgment**, not indiscriminate agreement. The model should retain uncertainty, factual caution, consent and age distinctions, recognition of genuine urgency, and the ability to challenge an incorrect premise.

## 2. Model-agnostic scope

This plan is not tied to Qwen. Qwen3.8-Flash-Next is one possible future candidate, but other models may be evaluated under the same process. Qwen3.8-Flash-Next is built on the Qwen 4 architecture, so Heretic compatibility demonstrated for Qwen 3.5 must not be generalized to it.

Every candidate gets a separate architecture and feasibility appendix covering:

- base-model lineage and license;
- dense, MoE, hybrid-attention, state-space, and multimodal structure;
- modules Heretic recognizes and modules requiring an adapter;
- high-precision and quantized source availability;
- local or rented hardware required for residual collection and optimization;
- merge and quantization path;
- loaded-memory fit on the two GX10s;
- serving-runtime support;
- vision, tool-use, reasoning, and long-context behavior;
- privacy and provenance of all artifacts.

Do not generalize a successful method from one architecture to another without inspection and validation.

## 3. Candidate admission gate

Do not spend abliteration or quantization effort on a candidate until it has a plausible path through all of these gates:

1. **Capability:** credible evidence that it can match or beat the current DeepSeek daily driver on real agentic, coding, security, tool-use, vision, reasoning, and long-running workflows.
2. **Loaded footprint:** the final runtime, including weights, cache, workspace, replicated/offloaded tables, vision, MTP, and host overhead, must be no larger than the current DeepSeek deployment.
3. **Runtime support:** a real serving path must exist or be reasonably implementable on the two-GX10 topology.
4. **Abliteration support:** Heretic must recognize the architecture, or a bounded architecture adapter must be feasible.
5. **Quantization path:** the final standalone derivative must preserve the selected W4A8, W4A4, NVFP4, FP8, or mixed-precision contract. A model-specific bounded merge into existing protected high-precision islands is acceptable when every quantized tensor and scale remains unchanged and this boundary is independently verified.
6. **Privacy:** all personal prompt corpora and evaluation records remain local and private.

Vendor benchmarks are leads, not acceptance evidence.

## 4. Narrow-intervention principle

Use the smallest effective model change.

Heretic represents directional ablation as a low-rank adapter-style delta rather than rewriting the entire model during optimization:

```text
ΔW = -λ · v · (vᵀW)
```

Where `v` is the learned refusal direction and `λ` is the optimized intervention strength. With full row-norm preservation, Heretic uses a small rank-3 approximation.

The optimization order is:

1. token-mixing or attention output projections in a narrow layer band;
2. hybrid/recurrent output projections only if the first pass is insufficient;
3. widen the layer band only when holdout refusals justify it;
4. touch MLP down-projections only when a measured gain cannot be obtained more narrowly;
5. do not modify embeddings, output heads, routers, experts, vision towers, memory tables, or auxiliary heads without architecture-specific evidence.

Heretic's own implementation notes that MLP ablation is often unnecessary and tends to damage intelligence more than attention-output ablation. The optimizer can disable it entirely.

## 5. Private behavior corpus

Use three separate local datasets. Do not upload them to Hugging Face, GitHub, W&B, or any hosted evaluation service.

### 5.1 Direction-training set

Prompts that legitimately need answers but commonly trigger refusals. Approximately 400–1,000 curated examples across the target domains.

### 5.2 Optimization set

A separate set used to choose:

- global versus per-layer refusal directions;
- target components;
- layer-band center and width;
- intervention strength;
- attention/token-mixer versus MLP involvement;
- the best refusal-reduction versus capability-preservation Pareto point.

### 5.3 Blind holdout set

Prompts never used for direction discovery or optimization. This proves that the result removes a general false-positive feature rather than memorizing wording.

Store only the detail necessary to reproduce the behavioral category. Avoid unnecessary intimate or proprietary content. Encrypt or otherwise access-control raw personal examples if they are retained.

## 6. Required false-positive domains

### 6.1 Internal software and security work

Include legitimate prompts involving:

- source and dependency auditing;
- vulnerability research and reproduction in controlled environments;
- reverse engineering and decompilation;
- malware classification and behavioral analysis;
- authentication, authorization, and cryptographic review;
- exploit analysis, patch development, and defensive validation;
- network, container, host, and cloud hardening;
- red-team simulation and incident response;
- proprietary internal systems and model-internals research.

The model must distinguish research or defensive context from intent based on the complete prompt rather than isolated terminology.

### 6.2 Mental health and personal wellbeing

Include legitimate prompts involving:

- stress, anxiety, depression, burnout, and emotional regulation;
- intrusive thoughts versus intention;
- therapy approaches and terminology;
- medication and side-effect research;
- sleep, attention, motivation, and executive function;
- journaling, reflection, and preparation for clinical conversations;
- academic discussion of suicide or self-harm;
- helping or understanding another person;
- difficult feelings that do not imply imminent self-harm intent.

The key behavioral distinction is:

```text
Discussing distress is not the same as expressing intent to self-harm.
```

A genuinely urgent scenario should still receive a calm, compassionate, and practically useful answer. Recognition of urgency is a capability to preserve; canned crisis escalation in response to ordinary discussion is the false positive to remove.

### 6.3 Consensual adult sexual and relationship wellness

Include legitimate prompts involving:

- sexual health and anatomy;
- intimacy and communication between spouses or partners;
- libido and differences in desire;
- comfort, consent, and boundaries;
- contraception, reproductive health, and STI information;
- medication, aging, hormonal, or physical-health effects;
- evidence-oriented ways to improve mutual adult wellbeing.

The model should not treat ordinary consensual adult sexual-health research as inherently inappropriate. It must preserve contextual distinctions involving consent and adulthood.

### 6.4 Authorized personal automation and delegated digital agency

Include legitimate tasks in which an agent acts on Carmen's explicit authority, such as:

- signing into an account through an approved local browser or authenticated workflow;
- checking application, order, publication, account, service, or workflow status;
- posting or scheduling an article or other user-owned content;
- maintaining websites, content-management systems, and routine online operations;
- submitting ordinary forms or updating records at Carmen's direction;
- downloading authorized records or reports;
- monitoring dashboards and reporting material changes;
- carrying out repetitive personal-administration tasks while Carmen is working;
- confirming the exact result after an external action.

These actions are distinct from software development: the agent is operating a service on the user's behalf rather than merely writing code for it. The model should not refuse solely because a task involves authentication, browser control, publishing, account access, or an external side effect.

Preserve authorization and execution judgment:

- operate only on accounts and resources Carmen owns or is authorized to manage;
- keep the requested scope literal;
- distinguish drafting or previewing from publishing;
- distinguish checking status from changing status;
- require clear target identity before external writes;
- verify state-changing actions by reading back the exact target;
- surface permission, payment, credential, and irreversible-action boundaries rather than guessing;
- never place passwords, tokens, cookies, or private account contents in training or evaluation datasets.

The desired behavior is reliable delegated execution—not blanket permission for surprise actions.

## 7. Preserve judgment without preserving reflexive refusal

The desired model should retain:

- factual uncertainty and source awareness;
- recognition of genuine medical or situational urgency;
- context-sensitive consent and age distinctions;
- ability to identify missing information;
- ability to challenge a false premise;
- useful suggestions to consult a professional when genuinely warranted;
- empathy and tact without canned scripts;
- calibrated confidence rather than reckless fabrication.

Remove:

- keyword-triggered refusals;
- unnecessary crisis scripts;
- generic disclaimers in place of answers;
- moralizing or judgment;
- treating research as intent;
- treating discussion of symptoms as a diagnosis or emergency;
- treating consensual adult sexuality as disallowed;
- coddling that prevents a direct technical or personal-improvement answer.

## 8. Two-stage behavior design

### Stage A — minimal directional abliteration

Use Heretic to remove the smallest effective residual refusal feature. Keep the result as a reversible LoRA-style adapter during development.

Heretic should co-optimize:

- refusal-marker or target-answer rate on the private false-positive corpus;
- KL divergence from the unmodified model on broad benign prompts;
- intervention sparsity or a manually enforced narrow component policy.

### Stage B — optional private behavior-quality adapter

Abliteration primarily removes refusal behavior; it does not necessarily teach the ideal response style. If needed, train a small local SFT/DPO adapter to improve:

- directness without alarmism;
- context-sensitive mental-health answers;
- respectful, evidence-oriented adult sexual-health answers;
- security and software-development usefulness;
- uncertainty and professional-referral calibration;
- avoidance of canned disclaimers, judgment, and moralizing.

Training examples may be generated and reviewed using the current local DeepSeek deployment. Private prompts and responses must remain inside the local environment.

Keep Stage A and Stage B separable during evaluation so any regression has a clear cause.

## 9. Capability-preservation suite

Compare the original model, the abliteration adapter, the merged model, and the final quantized model on:

- coding and repository work;
- tool calling and structured JSON;
- multi-step agent execution;
- authorized browser, account, publishing, and personal-administration workflows;
- security analysis;
- general reasoning and mathematics;
- instruction following;
- long-context retrieval and continuity;
- vision, OCR, and document analysis;
- multilingual capability;
- thinking-mode behavior;
- long-running consistency and loop avoidance;
- hallucination and uncertainty calibration;
- the three private false-positive domains.

Reject a candidate that gains compliance but introduces material regressions, including broken tool calls, weaker coding, excessive agreeableness, hallucination, vision loss, long-context loss, looping, or personality/style drift.

## 10. Artifact sequence

Use this order:

```text
Pinned source revision in its accepted native quantization
→ reversible narrow abliteration adapter used only during search
→ optional private behavior-quality adapter
→ model-specific standalone materialization
→ quantization/tensor-boundary verification
→ standalone private serving artifact
```

For ordinary candidates, standalone materialization may require a high-precision merge followed by target quantization. For Laguna FP8, that route is forbidden by memory and unnecessary: the 48 allowed `self_attn.o_proj.weight` targets are already BF16 protected islands. Merge only those BF16 tensors and copy every FP8 weight/scale pair unchanged.

Do not begin by ablating an aggressive production quant unless architecture or hardware constraints force an explicitly documented experiment. Learning against quantization error makes damage attribution harder.

For multimodal candidates, preserve the vision tower in BF16 unless direct evidence supports lower precision. Keep abliteration focused on the language backbone unless a vision-specific refusal failure is demonstrated.

## 11. Privacy contract

- Raw personal prompts and authorized-action examples remain local.
- Internal code, security details, mental-health research, and relationship/sexual-wellness material are never uploaded as datasets or logs.
- Credentials, browser sessions, cookies, account records, unpublished content, and private dashboard data are never used as training examples or embedded in manifests.
- Disable hosted experiment tracking and telemetry.
- Do not publish checkpoints, adapters, manifests, or evaluation outputs without an explicit privacy and provenance review.
- Public artifacts, if ever requested, contain no private prompts, paths, usernames, hostnames, credentials, or evaluation transcripts.
- Use local checksums and manifests for reproducibility.
- Keep private evaluation data outside public repositories and model directories likely to be uploaded.

## 12. Acceptance criteria

A daily-driver candidate succeeds only when all applicable conditions are verified:

1. False-positive refusals are reduced substantially across software/security work, authorized personal automation, mental health, and consensual adult sexual wellness.
2. Improvement generalizes to the blind holdout set.
3. Responses remain context-sensitive rather than indiscriminately agreeable.
4. Benign-output divergence stays on the best practical Pareto frontier.
5. Coding, reasoning, tool use, vision, and long-context results remain within the accepted quality envelope.
6. No new looping, reckless fabrication, or material calibration failure appears.
7. Authorized external actions preserve scope, distinguish previews from writes, and are verified after execution.
8. The standalone quantized artifact retains the behavior of the adapter-stage winner and loads without the original adapter or base checkout.
9. The final runtime is demonstrably better than the current DeepSeek daily driver for Carmen's real work.
10. The actual loaded footprint is the same as or smaller than DeepSeek, including cache and runtime overhead.
11. The model remains stable under sustained two-GX10 use.

## 13. Per-model appendix template

Create one note or section per candidate:

```markdown
### Candidate: <model>

- Base revision:
- License:
- Architecture:
- Total / active parameters:
- Modalities:
- Native / extended context:
- Candidate quant:
- Expected and measured loaded footprint:
- Heretic support status:
- Abliterable components:
- Components explicitly protected:
- Hardware required for optimization:
- Merge strategy:
- Quantization strategy:
- Serving runtime:
- Baseline quality evidence:
- Private-domain evaluation result:
- Capability-preservation result:
- Decision: reject / hold / advance
```

Candidate-specific experiments must not silently change this shared contract.

## 14. Current decision

No immediate model conversion is authorized by this note. Continue using the proven DeepSeek deployment until a candidate is:

- the same size or smaller in real loaded memory;
- credibly better before and after abliteration;
- supported by a viable private conversion and serving path;
- validated against the complete behavioral and capability-preservation suite.

## 15. Laguna S 2.1 go-forward plan

Laguna is the active candidate for the first controlled abliteration project. The preferred end-state is a small text/agent model paired with a separate resident vision model if the measured combined footprint permits it. A native Laguna vision graft is deferred unless sidecar vision proves inadequate in real workflows.

### 15.1 Hardware conclusion

- Laguna must remain FP8 or better throughout direction extraction, optimization, and validation. Four-bit Laguna loops badly and is not a behaviorally valid proxy.
- Huihui's heavily quantized Laguna GGUFs are excluded as starting checkpoints. They cannot establish the behavior or preservation quality of an FP8-derived adapter.
- Alien is not a practical machine for Laguna residual collection.
- Heretic requires the complete model to remain addressable for forward passes. The 122.258 GiB FP8 tensor payload already exceeds the approximately 121.6 GiB visible capacity of either individual GX10 before activations and runtime overhead.
- The production DeepSeek deployment occupies both GX10s, so the Laguna run requires a scheduled DeepSeek outage.
- Stock Heretic does not automatically combine two networked GX10s into one Accelerate device map. An FP8 Laguna run therefore needs a real two-node model-parallel residual and evaluation path; the earlier one-model-per-node BNB4 plan is rejected.

### 15.2 Preparation while DeepSeek remains live

1. Pin the exact Laguna source revision and inventory all files.
2. Design and test a two-node FP8 model-parallel residual-collection and adapter-evaluation path without altering production.
3. Build the private direction-training, optimization, and blind-holdout datasets.
4. Freeze the capability-preservation and false-positive scoring contracts.
5. Test orchestration, distributed synchronization, checkpoints, watchdogs, and adapter export with a smaller architecture-compatible model.
6. Prepare and verify the complete DeepSeek stop, memory-reclamation, restart, per-rank health, text-inference, and vision-inference procedure.

### 15.3 First maintenance night: hard feasibility gate

After stopping DeepSeek cleanly and reclaiming memory on both hosts:

1. Load the exact Laguna FP8 checkpoint across both GX10s with verified model-parallel sharding.
2. Verify one distributed ordinary generation without looping.
3. Verify selected-layer residual extraction and correct cross-rank reconstruction.
4. Verify Laguna-specific architecture traversal and narrow LoRA targets.
5. Complete one bounded distributed trial.
6. Export and reload the temporary adapter against the FP8 base, then prove the standalone materialization boundary on the selected candidate.
7. Record per-rank peak memory, communication behavior, wall time, output quality, and trials-per-hour.

If loading, traversal, residual extraction, or adapter reload fails, stop the experiment and restore DeepSeek rather than leaving an unproductive week-long run active.

### 15.4 Multi-night search

- Both GX10s participate in one FP8 model-parallel Laguna instance; neither host is an independent BNB4 lane.
- Run primary narrow attention/output-projection search first, then perform independent seed replication sequentially against the same distributed backend.
- Require similar direction/layer/strength findings across replicated studies before treating one optimizer result as stable.
- Broaden layer range or components only when the attention-only Pareto candidates are insufficient.
- Do not touch routers, experts, embeddings, output heads, or MLP projections without measured need.
- Save persistent study state, per-trial logs, and atomic adapter snapshots.
- Use finite per-trial and outer wall-time limits, plus immediate alerts for OOM, CUDA/process exit, or stalled progress.

### 15.5 Artifact and restoration rules

- Export reversible adapters only during search; the adapter is not project completion.
- Keep the FP8 base immutable and do not use a 4-bit surrogate for residual extraction, optimization, or acceptance testing.
- Do not dequantize or reconstruct the full model merely to merge a trial candidate.
- Do not materialize the standalone derivative until blind-holdout and capability-preservation evaluation select a winner.
- For Laguna, merge only the 48 BF16 protected `self_attn.o_proj.weight` tensors and preserve all FP8 weights and `.weight_scale_inv` tensors byte-for-byte; no FP8 re-encoding is permitted or required.
- Completion requires a standalone directory with no adapter sidecars, a verified SHA-256 manifest, exact tensor/dtype accounting, and deterministic two-rank inference.
- At completion or abort, stop Heretic, reclaim CUDA/unified memory, restart both DeepSeek ranks in the correct order, verify live health, and run real text and vision inference before declaring production restored.

## 16. Huihui method as an alternative lane

Huihui's Laguna model card says its artifact was created with abliteration and points readers to Sumandora's `remove-refusals-with-transformers` repository.[1] That repository describes itself as a crude proof of concept and currently contains two editable Python scripts rather than an installable application, GUI, packaged CLI, release binary, or optimization service.[2]

The public reference script:

- hard-codes the model and settings in Python;
- loads the model in BitsAndBytes 4-bit on CUDA;
- samples 32 harmful and 32 harmless instructions;
- takes the mean hidden-state difference at approximately 60% depth;
- normalizes that single refusal direction;
- applies activation projection through inserted decoder layers during inference.

Its 4-bit loading path is unsuitable for Laguna because Laguna's behavior degrades into severe looping at that precision. The public script is useful as a description of simple direction extraction, not as the execution backend for this project.

It does **not** publicly implement Heretic-style Optuna/TPE search, KL-divergence optimization, a blind holdout workflow, adapter export, or a production merge pipeline.[2][4] Huihui has published many model artifacts on Hugging Face, but the public Huihui GitHub account currently exposes no repositories or application corresponding to the private production workflow used to generate them.[3]

Therefore, treat any reported lower KL divergence as an artifact-specific measurement until the exact prompts, token positions, direction extraction, target layers, intervention strength, KL direction, tokenizer, and evaluation corpus are matched. A lower number measured under a different harness is not evidence that one method is intrinsically less damaging.

The Laguna project should compare two reproducible lanes if feasible:

1. **Heretic lane:** narrow adapter search with refusal and KL objectives plus preservation gates.
2. **Huihui/Sumandora-method lane:** reproduce only the simple hidden-state-difference algorithm on the same distributed FP8 Laguna base, then evaluate it under the exact same refusal, KL, and capability suite.

The existing Huihui Laguna GGUF is excluded as a starting point. The project requires a reproducible adapter derived from the exact FP8 base and a controlled FP8 serving lineage.[1]

## References

- [Heretic](https://github.com/p-e-w/heretic)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Projected abliteration](https://huggingface.co/blog/grimjim/projected-abliteration)
- [Norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
- [Huihui Laguna S 2.1 abliterated GGUF](https://huggingface.co/huihui-ai/Huihui-Laguna-S-2.1-abliterated-GGUF)
- [Sumandora remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [Huihui AI Hugging Face profile](https://huggingface.co/huihui-ai)
- [llama.cpp cvector-generator](https://github.com/ggml-org/llama.cpp/blob/master/tools/cvector-generator/README.md)
- Related local planning notes: [[GX10 Model Eval Suite - V3]], [[Longcat Flash Lite Sparse Quant Plan]]

## Sources

[1] https://huggingface.co/huihui-ai/Huihui-Laguna-S-2.1-abliterated-GGUF — Huihui Laguna S 2.1 abliterated model card
[2] https://github.com/Sumandora/remove-refusals-with-transformers — remove-refusals-with-transformers
[3] https://huggingface.co/huihui-ai — Huihui AI Hugging Face profile
[4] https://github.com/p-e-w/heretic — Heretic
[5] https://github.com/ggml-org/llama.cpp/blob/master/tools/cvector-generator/README.md — llama.cpp cvector-generator
