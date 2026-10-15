### Your current environment

- GPU: one NVIDIA L4 on Modal; Python 3.12.3; PyTorch 2.13.0+cu130.
- Python source: `c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a`, including the forward benchmark introduced in #61339.
- Compiled runtime: official vLLM image reporting `0.32.0rc1.dev100+g77154ca61`.
- Model architecture/config: Qwen3-0.6B; random dummy weights, TP=1.

**Build provenance:** this is a composite setup: the pinned Python source is overlaid on the official image's compiled extensions and generated third-party wrappers. The installed wheel by itself did not contain the new benchmark. The archive records the image digest, source hash, runtime versions, commands, and the differing public-registry/Modal image config hashes; it does not claim a bit-identical source build. These checks are benchmark correctness smoke tests, not model-quality evaluation.

### Describe the bug

`vllm bench forward` accepts a verify case requiring draft tokens even when the runner has no draft capacity. On the setup above, `verify=2*4@64:3` without `--speculative-config` reaches CUDA execution and fails with an illegal memory access. Ordinary decode passes, and the same verify case passes when capacity 3 is explicitly configured. The failure reproduced in both an uninstrumented native run and a preservation rerun using an exported checkpoint.

Expected: reject a case that exceeds the runner's resolved draft capacity with an actionable configuration error, before constructing/preparing that case's inputs.

### Minimal reproduction

On the source revision above with its CUDA dependencies available:

```bash
vllm bench forward \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --load-format dummy --seed 0 \
  --max-model-len 512 --max-num-batched-tokens 128 \
  --max-num-seqs 4 --gpu-memory-utilization 0.4 \
  --routing real --kv-len-jitter 0 --iters 2 --samples 2 \
  --case 'verify=2*4@64:3'
```

Control: replace the case with `decode=2*1@64`, or keep verify and add:

```bash
--speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3}'
```

The linked Modal scripts automate the complete source overlay and reproduction. The public checkpoint allows replay with the exact exported random weights using `--load-format sharded_state`; the initial dummy run's discarded weights are not claimed to have been recovered byte for byte.

### Actual result and suspected mechanism

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The full original traceback and stdout/stderr are preserved in the raw-log links below. Modal also reported:

```text
Xid 31 ... faulted @ 0x0_00000000 ... ACCESS_TYPE_VIRT_READ
```

The preservation run recorded `num_speculative_steps=0`, `draft_tokens_shape=[4,0]`, and `draft_tokens_data_ptr=0`, while the batch schedules three draft tokens per verify request. This suggests missing case/capacity validation before the draft-reading input preparation path. CUDA errors are asynchronous; the traceback alone does not identify the faulting kernel, and no CUDA core dump was collected. This report concerns the benchmark, not a public inference request.

### Proposed fix and validation

The prepared fix checks the maximum case draft count against `runner.num_speculative_steps` at the start of `run_case`. It uses resolved runner capacity rather than assuming an explicit CLI configuration, and reports the needed capacity plus a `--speculative-config` hint. Engine initialization still precedes this check.

- Regression subset: before patch **2 failed, 2 passed**; after patch **4 passed, 1 deselected**. The existing model-dependent end-to-end test was deselected; three explicit L4 CLI runs below provide GPU coverage.
- Patched verify without config: clean `ValueError` (`needs 3 draft tokens per request, but the runner supports 0`), with no illegal memory access.
- Patched decode and configured verify: both exit 0, finite/deterministic outputs, graph-versus-eager relative RMS 0.
- Full repository pre-commit checks on the three fix files passed.
- GPU controls use the same public exported checkpoint revision; no accuracy or serving-throughput claim is made.

### Reproduction artifacts

- [Reproduction archive and runnable scripts](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15)
- [Fix branch](https://github.com/znzryb/vllm/tree/fix-forward-verify-capacity), commit `80b4c3c2fd04e92d5b1d9f979202445a98ff41aa`.
- [Original uninstrumented raw log](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/raw/initial-modal.log), [preservation-run full traceback](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/data/verify/stderr.log), [post-fix validation logs](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/data/validation).
- [Public model weights, fixed revision](https://huggingface.co/znzryb/qwen3-0.6b-dummy-forward-repro/tree/4bae0ac45199180f8d59d05bb3b6cc1b19c58357).
- [Public raw data and validation, fixed revision](https://huggingface.co/datasets/znzryb/vllm-forward-verify-repro/tree/239159c4fdaaf568fe9c00153cbfbf57ab1bf8bc).
- [Validation receipt](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/results/validation.json) and [pre-commit log](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/raw/fix-precommit.log).

The model repository contains **random dummy weights**, not a pretrained language model. Synthetic token IDs, block assignments, runner metadata, raw process logs, and output JSON are public in the dataset and mirrored in the fork. No credentials or private prompts are included.

### Duplicate checks and assistance

Searched existing issues and open PRs for `"bench forward"` and draft/capacity terms. #61692 adds profiling and does not address this validation failure; no overlapping fix was found at submission time.

AI assistance was used to investigate, prepare the fix/scripts, and execute checks. No human code review is claimed. No PR has been opened.

- [x] Searched relevant issues and PRs.
- [ ] Consulted the documentation chatbot: the documentation page was opened, but no chatbot control was visible, so consultation is not claimed.
