# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download the pinned upstream snapshot into ignored project storage."""

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

SOURCE_COMMIT = "c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a"
script_dir = Path(__file__).resolve().parent
work = Path(
    os.environ.get(
        "VLLM_AUDIT_DIR", script_dir.parents[1] / ".local/bug-audits/forward-verify"
    )
)
work.mkdir(parents=True, exist_ok=True)
source = work / "upstream"
if not source.exists():
    archive = work / f"upstream-{SOURCE_COMMIT}.tar.gz"
    with archive.open("wb") as stream:
        subprocess.run(
            ["gh", "api", f"repos/vllm-project/vllm/tarball/{SOURCE_COMMIT}"],
            stdout=stream,
            check=True,
        )
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter="data")
        folders = list(Path(temporary).iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise RuntimeError("Unexpected source archive layout")
        folders[0].rename(source)

main_file = source / "vllm/benchmarks/forward/main.py"
actual = hashlib.sha256(main_file.read_bytes()).hexdigest()
expected = "a33eac4641dab02a44d2e402974bf4daa2f92102a535d0042f26d18f2aacc545"
if actual != expected:
    raise RuntimeError(f"Unexpected forward source hash: {actual}")
(work / "source-provenance.json").write_text(
    json.dumps(
        {
            "source_commit": SOURCE_COMMIT,
            "forward_main_sha256": actual,
        },
        indent=2,
    )
)
print(source)
