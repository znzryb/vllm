# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture exact tensors and a random checkpoint, then publish public evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal
from huggingface_hub import get_token
from modal_forward import (
    AUDIT_DIR,
    RUNTIME_IMAGE,
    SCRIPT_DIR,
    app,
    image,
    source_with_extensions,
)

MODEL_REPO = "znzryb/qwen3-0.6b-dummy-forward-repro"
DATA_REPO = "znzryb/vllm-forward-verify-repro"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

capture_image = image.add_local_python_source("modal_forward").add_local_file(
    SCRIPT_DIR / "capture_driver.py", "/audit-driver.py"
)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": get_token()})


@app.function(
    image=capture_image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    secrets=[hf_secret],
)
def preserve():
    import hashlib
    import os
    import shutil
    import subprocess
    import sys
    import time

    import torch
    from huggingface_hub import HfApi, snapshot_download

    import vllm

    source_with_extensions()
    root = Path("/capture")
    model_dir = root / "model"
    data_dir = root / "dataset"
    data_dir.mkdir(parents=True)
    result = {
        "source_commit": "c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a",
        "compiled_runtime_version": vllm.__version__,
        "runtime_image": RUNTIME_IMAGE,
        "python": sys.version,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "upstream_model": "Qwen/Qwen3-0.6B",
        "upstream_model_revision": REVISION,
        "load_format": "dummy (decode); exported sharded_state (verify controls)",
        "seed": 0,
        "cases": [],
    }
    for name, shape in (
        ("decode", "2*1@64"),
        ("verify", "2*4@64:3"),
        ("verify_configured", "2*4@64:3"),
    ):
        case_dir = data_dir / name
        case_dir.mkdir()
        cmd = [
            sys.executable,
            "/audit-driver.py",
            "bench",
            "forward",
            "--model",
            "Qwen/Qwen3-0.6B" if name == "decode" else str(model_dir),
            "--tokenizer",
            "Qwen/Qwen3-0.6B",
            "--revision",
            REVISION,
            "--load-format",
            "dummy" if name == "decode" else "sharded_state",
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
            "--case",
            f"{name}={shape}",
            "--output-json",
            str(case_dir / "output.json"),
        ]
        if name == "verify_configured":
            cmd += [
                "--speculative-config",
                '{"method":"ngram_gpu","num_speculative_tokens":3}',
            ]
        env = {
            **os.environ,
            "PYTHONPATH": "/audit-src",
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_AUDIT_CAPTURE_DIR": str(case_dir),
        }
        if name == "decode":
            env["VLLM_AUDIT_EXPORT_MODEL"] = str(model_dir)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=220
            )
            item = {
                "name": name,
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_s": time.monotonic() - started,
            }
            (case_dir / "stdout.log").write_text(proc.stdout)
            (case_dir / "stderr.log").write_text(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            item = {"name": name, "command": cmd, "timeout": True}
            (case_dir / "stdout.log").write_bytes(exc.stdout or b"")
            (case_dir / "stderr.log").write_bytes(exc.stderr or b"")
        result["cases"].append(item)
        print(json.dumps(item), flush=True)
        if name == "decode" and item.get("returncode") != 0:
            raise RuntimeError((case_dir / "stderr.log").read_text()[-8000:])

    metadata = Path(
        snapshot_download(
            "Qwen/Qwen3-0.6B",
            revision=REVISION,
            allow_patterns=["*.json", "*.txt", "*.jinja", "LICENSE", "merges.txt"],
        )
    )
    for file in metadata.iterdir():
        if file.is_file() and not (model_dir / file.name).exists():
            shutil.copy2(file, model_dir / file.name)
    (model_dir / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-0.6B\n"
        "tags:\n- vllm\n- debugging\n---\n\n"
        "# Random dummy checkpoint for a vLLM forward-verify reproducer\n\n"
        "These are random dummy weights exported from the native vLLM runner, "
        "not Qwen's pretrained weights. They have no useful text-generation quality. "
        "Load with vLLM `--load-format sharded_state` (TP=1), not AutoModel.\n\n"
        f"Configuration/tokenizer source: Qwen/Qwen3-0.6B at `{REVISION}`. "
        "Initialization seed: 0.\n\n"
        f"Evidence: https://huggingface.co/datasets/{DATA_REPO}\n"
    )
    (data_dir / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n\n# vLLM forward-verify reproduction\n\n"
        "Synthetic token IDs, exact runner capacity, raw stdout/stderr, "
        "and measurements from three native benchmark processes on one L4.\n\n"
        "Python source is pinned independently of the compiled image runtime; "
        "both versions are recorded in results.json. "
        "No model-quality or serving-performance claim.\n"
    )
    hashes = {}
    for file in model_dir.glob("*.safetensors"):
        with file.open("rb") as stream:
            hashes[file.name] = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": file.stat().st_size,
            }
    result["model_weights"] = hashes
    result["model_repo"] = MODEL_REPO
    result["dataset_repo"] = DATA_REPO
    api = HfApi()
    model_commit = api.upload_folder(
        repo_id=MODEL_REPO,
        folder_path=model_dir,
        commit_message="Preserve random checkpoint for the forward verify reproducer",
    )
    result["model_commit"] = model_commit.oid
    (data_dir / "results.json").write_text(json.dumps(result, indent=2))
    dataset_commit = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=data_dir,
        commit_message="Preserve exact inputs, raw logs, and GPU measurements",
    )
    result["dataset_commit"] = dataset_commit.oid
    return result


@app.local_entrypoint()
def publish():
    result = preserve.remote()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = AUDIT_DIR / f"published-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved", path)
