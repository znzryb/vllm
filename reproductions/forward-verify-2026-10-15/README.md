# Forward verify draft-capacity investigation

This directory is a research archive on a personal vLLM fork, not an upstream PR.
It preserves a native GPU reproduction and a proposed validation fix for
`vllm bench forward`.

## Observed failure

On one NVIDIA L4, Python source at
`c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a`, a three-token verify case
(`2*4@64:3`) with no speculative configuration exits with CUDA illegal memory
access. The Modal GPU health log reports Xid 31, a virtual read at address zero.
The same setup passes ordinary decode and passes verify with
`{"method":"ngram_gpu","num_speculative_tokens":3}`.

| Initial native run | Exit code | Output |
| --- | --- | --- |
| `decode=2*1@64` | 0 | Finite, deterministic; graph/eager relative RMS 0 |
| `verify=2*4@64:3`, no spec config | 1 | CUDA illegal memory access; Xid 31 |
| Same verify, configured capacity 3 | 0 | Finite, deterministic; graph/eager relative RMS 0 |

These are correctness smoke checks with four timed samples, not a serving
performance or model-quality evaluation. Random dummy weights are used.

The source schedules three draft tokens per request but does not validate the
runner's draft capacity. A runner without speculation allocates a zero-width
draft buffer; the input-preparation path still receives scheduled draft tokens.
This is the suspected mechanism. The initial asynchronous Python traceback
does not identify the faulting CUDA kernel; no CUDA core dump was collected.

## Proposed fix

`fix.patch` checks each case against `runner.num_speculative_steps` before batch
construction and input preparation. It rejects missing or insufficient capacity
with a case-specific `ValueError` and a `--speculative-config` hint. It does not
enable speculative decoding automatically or change target-model computation.

The fix lives on branch `fix-forward-verify-capacity`, commit `80b4c3c2f`.
The full regression suite is preserved as `test_forward_cli.py`; the source
checkout's existing benchmark suite is extended, rather than a new upstream
test file being introduced. The repository pre-commit checks passed, including
Ruff, mypy, test tethering, SPDX, and skill validation; the full log is retained.

## Reproduce

Install `uv`, authenticate the Modal CLI, and have `gh` available. Run:

```bash
bash reproductions/forward-verify-2026-10-15/run.sh
```

`prepare_source.py` downloads the pinned public source commit and checks the
forward entrypoint's SHA-256. Runtime snapshots and new outputs go under
`.local/bug-audits/forward-verify/`, which is ignored by Git. This directory
contains the reviewed copies intended for public preservation.

The Python source snapshot is overlaid on an official CUDA image; its compiled
extensions and build-generated third-party wrappers come from that image.
The initial image reports `0.32.0rc1.dev100+g77154ca61`, Torch `2.13.0+cu130`,
and Python 3.12.3. This is a composite source/runtime validation, not a claim
that the installed wheel alone contains `bench forward`. That wheel did not
contain the new benchmark. The replay scripts pin a public image digest and
record source and runtime versions separately.

`results/image-resolution.json` retains both the public registry resolution
and the initial Modal image config hash. These differ; it is not evidence
that the two complete container images are bit-identical.

## Public model and data

- Model: <https://huggingface.co/znzryb/qwen3-0.6b-dummy-forward-repro>
- Data: <https://huggingface.co/datasets/znzryb/vllm-forward-verify-repro>
- Original config/tokenizer: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0).

The public checkpoint is random vLLM dummy state, not the pretrained Qwen
weights. It is exported during the preservation rerun and requires vLLM
`--load-format sharded_state`, TP=1. It must not be presented as a useful
language model or as a byte-for-byte recovery of the initial run's discarded
weights. Exact input IDs and runner capacity are recorded in the preservation
rerun, while the initial uninstrumented raw logs remain separate.

`publish_modal.py::publish` exports that checkpoint, captures synthetic input
IDs and raw process logs, and uploads them to the public model/dataset repos.
It requires the owner's HF write authentication. `validate_modal.py::check`
runs the regression tests before/after `fix.patch`, then tests the patched
benchmark against the exported model revision. Publication receipts and pinned
Hub revisions are retained with the results.

## Scope and provenance

This report concerns benchmark input validation. It does not demonstrate a
serving-engine failure from a public inference request. No upstream PR has
been opened. AI assistance was used for investigation, scripts, the proposed
patch, and executing checks. No human code review is claimed.

## Completed patch validation

- Regression subset: before patch **2 failed, 2 passed**; after patch **4 passed, 1 deselected**. The existing model-dependent end-to-end test was deselected; three explicit L4 CLI runs below provide GPU coverage.
- Patched verify without config: clean `ValueError` (`needs 3 draft tokens per request, but the runner supports 0`), with no illegal memory access.
- Patched decode and configured verify: both exit 0, finite/deterministic outputs, graph-versus-eager relative RMS 0.
- Full repository pre-commit checks on the three fix files passed.
- GPU controls use the same public exported checkpoint revision; no accuracy or serving-throughput claim is made.

- [Reproduction archive and runnable scripts](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15)
- [Fix branch](https://github.com/znzryb/vllm/tree/fix-forward-verify-capacity), commit `80b4c3c2fd04e92d5b1d9f979202445a98ff41aa`.
- [Original uninstrumented raw log](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/raw/initial-modal.log), [preservation-run full traceback](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/data/verify/stderr.log), [post-fix validation logs](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/data/validation).
- [Public model weights, fixed revision](https://huggingface.co/znzryb/qwen3-0.6b-dummy-forward-repro/tree/4bae0ac45199180f8d59d05bb3b6cc1b19c58357).
- [Public raw data and validation, fixed revision](https://huggingface.co/datasets/znzryb/vllm-forward-verify-repro/tree/239159c4fdaaf568fe9c00153cbfbf57ab1bf8bc).
- [Validation receipt](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/results/validation.json) and [pre-commit log](https://github.com/znzryb/vllm/tree/contribution-rules-and-bug-audit/reproductions/forward-verify-2026-10-15/raw/fix-precommit.log).
