# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the real source patch and preserve unit and GPU evidence."""

import json
from datetime import UTC, datetime

import modal
from huggingface_hub import get_token
from modal_forward import (
    AUDIT_DIR,
    RUNTIME_IMAGE,
    SCRIPT_DIR,
    app,
    runtime_image,
    source_with_extensions,
)

MODEL_REPO = "znzryb/qwen3-0.6b-dummy-forward-repro"
DATA_REPO = "znzryb/vllm-forward-verify-repro"
validation_image = (
    runtime_image.uv_pip_install("pytest")
    .add_local_dir(AUDIT_DIR / "upstream/vllm", "/audit-src/vllm")
    .add_local_python_source("modal_forward")
    .add_local_file(
        SCRIPT_DIR / "test_forward_cli.py",
        "/audit-tests/benchmarks/test_forward_cli.py",
    )
    .add_local_file(SCRIPT_DIR / "fix.patch", "/audit-fix.patch")
    .add_local_file(SCRIPT_DIR / "capture_driver.py", "/audit-driver.py")
)


@app.function(
    image=validation_image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=650,
    max_containers=1,
    retries=0,
    scaledown_window=2,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": get_token()})],
)
def validate(model_revision: str):
    import hashlib
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import torch
    from huggingface_hub import HfApi, snapshot_download

    import vllm

    source_with_extensions()
    root = Path("/validation")
    root.mkdir()
    env = {**os.environ, "PYTHONPATH": "/audit-src", "VLLM_USE_RUST_FRONTEND": "0"}
    result = {
        "source_commit": "c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a",
        "runtime_image": RUNTIME_IMAGE,
        "compiled_runtime_version": vllm.__version__,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "model_repo": MODEL_REPO,
        "model_revision": model_revision,
        "runs": [],
    }

    def run(name, command, timeout=210, extra_env=None):
        folder = root / name
        folder.mkdir()
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**env, **(extra_env or {})},
        )
        item = {
            "name": name,
            "command": command,
            "returncode": completed.returncode,
            "elapsed_s": time.monotonic() - started,
        }
        (folder / "stdout.log").write_text(completed.stdout)
        (folder / "stderr.log").write_text(completed.stderr)
        result["runs"].append(item)
        print(json.dumps(item), flush=True)
        return completed

    test = [
        sys.executable,
        "-m",
        "pytest",
        "--confcutdir=/audit-tests/benchmarks",
        "/audit-tests/benchmarks/test_forward_cli.py",
        "-q",
        "-k",
        "not test_bench_forward or rejects_oversized_verify",
    ]
    before = run("unit_before", test)
    patch = run(
        "apply_patch",
        ["patch", "--batch", "-p1", "-d", "/audit-src", "-i", "/audit-fix.patch"],
    )
    if patch.returncode:
        raise RuntimeError(patch.stdout + patch.stderr)
    after = run("unit_after", test)
    result["unit_red_green"] = before.returncode != 0 and after.returncode == 0
    result["patched_source_sha256"] = hashlib.sha256(
        Path("/audit-src/vllm/benchmarks/forward/main.py").read_bytes()
    ).hexdigest()
    model = snapshot_download(MODEL_REPO, revision=model_revision, token=False)
    base = [
        sys.executable,
        "/audit-driver.py",
        "bench",
        "forward",
        "--model",
        model,
        "--tokenizer",
        model,
        "--load-format",
        "sharded_state",
        "--seed",
        "0",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.4",
        "--kv-len-jitter",
        "0",
        "--routing",
        "real",
        "--iters",
        "2",
        "--samples",
        "2",
    ]
    for name, shape, config in (
        ("patched_verify_rejected", "2*4@64:3", None),
        ("patched_decode", "2*1@64", None),
        (
            "patched_verify_configured",
            "2*4@64:3",
            {"method": "ngram_gpu", "num_speculative_tokens": 3},
        ),
    ):
        folder = root / name
        cmd = base + [
            "--case",
            f"{name}={shape}",
            "--output-json",
            str(folder / "output.json"),
        ]
        if config is not None:
            cmd += ["--speculative-config", json.dumps(config)]
        completed = run(name, cmd, extra_env={"VLLM_AUDIT_CAPTURE_DIR": str(folder)})
        if name == "patched_verify_rejected":
            result["clean_rejection"] = (
                completed.returncode != 0
                and "needs 3 draft tokens per request, but the runner supports 0"
                in completed.stderr
                and "illegal memory access" not in completed.stderr
            )
        else:
            result[name + "_passed"] = completed.returncode == 0
    (root / "results.json").write_text(json.dumps(result, indent=2))
    info = HfApi().upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=root,
        path_in_repo="validation",
        commit_message="Preserve source-patch unit and GPU regression validation",
    )
    result["dataset_commit"] = info.oid
    return result


@app.local_entrypoint()
def check(model_revision: str):
    result = validate.remote(model_revision)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = AUDIT_DIR / f"validation-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved", path)
