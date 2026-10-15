# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe native benchmark inputs and export its random checkpoint."""

import hashlib
import json
import os
from pathlib import Path

import vllm.benchmarks.forward.batch as batch_module
import vllm.benchmarks.forward.main as bench_module

capture_dir = Path(os.environ["VLLM_AUDIT_CAPTURE_DIR"])
capture_dir.mkdir(parents=True, exist_ok=True)
build_batch = batch_module.build_batch
run_case = bench_module.run_case


def record_batch(*args, **kwargs):
    batch = build_batch(*args, **kwargs)
    data = {
        "expected": batch.expected,
        "query_token_ids": {k: v.tolist() for k, v in batch.query_token_ids.items()},
        "group_block_ranges": batch.group_block_ranges,
        "requests": [
            {
                "req_id": req.req_id,
                "prompt_token_ids": req.prompt_token_ids,
                "block_ids": req.block_ids,
                "num_computed_tokens": req.num_computed_tokens,
            }
            for req in batch.scheduler_output.scheduled_new_reqs
        ],
        "scheduled_spec_decode_tokens": (
            batch.scheduler_output.scheduled_spec_decode_tokens
        ),
    }
    (capture_dir / "batch.json").write_text(json.dumps(data, indent=2))
    return batch


def record_runner(runner, case, args):
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    state = runner.req_states
    metadata = {
        "case": case.name,
        "num_speculative_steps": runner.num_speculative_steps,
        "decode_query_len": runner.decode_query_len,
        "draft_tokens_shape": list(state.draft_tokens.shape),
        "draft_tokens_data_ptr": state.draft_tokens.data_ptr(),
        "model_config_revision": getattr(
            runner.model_config.hf_config, "_commit_hash", None
        ),
    }
    (capture_dir / "runner.json").write_text(json.dumps(metadata, indent=2))
    if os.environ.get("VLLM_AUDIT_EXPORT_MODEL"):
        model_dir = Path(os.environ["VLLM_AUDIT_EXPORT_MODEL"])
        model_dir.mkdir(parents=True, exist_ok=True)
        ShardedStateLoader.save_model(runner.model, str(model_dir))
        runner.model_config.hf_config.save_pretrained(model_dir)
        weights = {}
        for file in model_dir.glob("*.safetensors"):
            digest = hashlib.file_digest(file.open("rb"), "sha256").hexdigest()
            weights[file.name] = {"sha256": digest, "bytes": file.stat().st_size}
        (capture_dir / "weights.json").write_text(json.dumps(weights, indent=2))
    return run_case(runner, case, args)


def main():
    batch_module.build_batch = record_batch
    bench_module.run_case = record_runner

    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
