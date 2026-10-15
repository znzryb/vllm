# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = Path(
    os.environ.get(
        "VLLM_AUDIT_DIR",
        SCRIPT_DIR.parent.parent / ".local/bug-audits/forward-verify",
    )
)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "4d42798d27104deb507639250e7b8ad6b0250f97c3d4276038ee2fb7ac9e367e"
)
app = modal.App("vllm-forward-verify-audit")
runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE)
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python", "python --version")
    .env({"VLLM_NO_USAGE_STATS": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
image = runtime_image.add_local_dir(AUDIT_DIR / "upstream/vllm", "/audit-src/vllm")


def source_with_extensions():
    import vllm

    installed = Path(vllm.__file__).parent
    source = Path("/audit-src/vllm")
    for extension in installed.rglob("*.so"):
        target = source / extension.relative_to(installed)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(extension)
    for generated in ("_version.py", "_build_info.py"):
        if (installed / generated).exists() and not (source / generated).exists():
            (source / generated).symlink_to(installed / generated)
    for generated in (installed / "third_party").rglob("*"):
        if generated.is_file():
            target = source / generated.relative_to(installed)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(generated)
    return source


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def reproduce():
    import hashlib
    import os
    import subprocess
    import sys
    import time

    import torch

    import vllm

    source = source_with_extensions()
    result = {
        "vllm_version": vllm.__version__,
        "python": sys.version,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "cases": [],
    }
    result["source_commit"] = "c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a"
    result["forward_source"] = str(source / "benchmarks/forward/main.py")
    result["forward_sha256"] = hashlib.sha256(
        Path(result["forward_source"]).read_bytes()
    ).hexdigest()
    base = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "forward",
        "--model",
        "Qwen/Qwen3-0.6B",
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
    for name, shape in [
        ("decode", "2*1@64"),
        ("verify", "2*4@64:3"),
        ("verify_configured", "2*4@64:3"),
    ]:
        cmd = base + ["--case", f"{name}={shape}", "--output-json", f"/tmp/{name}.json"]
        if name == "verify_configured":
            cmd += [
                "--speculative-config",
                '{"method":"ngram_gpu","num_speculative_tokens":3}',
            ]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=170,
                env={
                    **os.environ,
                    "VLLM_USE_RUST_FRONTEND": "0",
                    "PYTHONPATH": "/audit-src",
                },
            )
            item = {
                "name": name,
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_s": time.monotonic() - started,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            path = Path(f"/tmp/{name}.json")
            if path.exists():
                item["output"] = json.loads(path.read_text())
        except subprocess.TimeoutExpired as exc:
            item = {
                "name": name,
                "timeout": True,
                "stdout": str(exc.stdout),
                "stderr": str(exc.stderr),
            }
        result["cases"].append(item)
        print(
            name,
            json.dumps(
                {k: v for k, v in item.items() if k not in ("stdout", "stderr")}
            ),
        )
        if name == "decode" and item.get("returncode") != 0:
            print(item.get("stderr", "")[-7000:])
            break
    return result


@app.local_entrypoint()
def main():
    result = reproduce.remote()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = AUDIT_DIR / f"modal-forward-result-{stamp}.json"
    with path.open("x") as output:
        json.dump(result, output, indent=2)
    print("Saved", path)
    for case in result.get("cases", []):
        print(
            case["name"],
            "returncode=",
            case.get("returncode"),
            case.get("stderr", "")[-7000:],
        )
