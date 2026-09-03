# Heretic DGX 0.1 validation status

Release 0.1 is validated for a deliberately narrow deployment: exactly two
NVIDIA DGX Spark systems running one NCCL rank per node.

## Completed validation

- A two-rank small-model fixture exercised coordinator launch, collective
  startup, prompt ingestion, residual calculation, two optimization trials,
  winner restoration, coordinated export, clean shutdown, artifact reload, and
  token generation.
- The full `poolside/Laguna-S-2.1-FP8` checkpoint loaded on both ranks, completed
  scoring and optimization, restored the selected trial, and produced a
  standalone checkpoint.
- Export verification confirmed that only intended target intervals changed,
  reconstructed targets matched the merge oracle, source FP8 tensors remained
  unchanged, and the artifact checksum manifest passed.
- The exported Laguna checkpoint passed a clean distributed runtime load and
  generated a completion.
- The selected Laguna trial measured KL divergence `0.0156` against the
  untouched model's first-token distributions on five
  `mlabonne/harmless_alpaca` prompts.

## Supported boundary

The evidence above does not establish general support for more than two nodes,
multiple ranks per node, other hardware, every model family, or every
quantization format. Each new combination requires its own load, optimization,
export, reload, and generation proof.

The validated model artifact and its detailed limitations are documented at
<https://huggingface.co/cbert33/Laguna-S-2.1-Heretic-FP8>.
