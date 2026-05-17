# Remote Bash Setup

Copy this into a bash terminal from the repository root on the remote machine.
It mirrors the package and FFmpeg/libvmaf setup from `getting_started.ipynb`.

```bash
set -euo pipefail

cd "${VCR_BENCH_REPO:-$PWD}"

python3 -m pip install --upgrade pip

# CUDA PyTorch wheel, matching getting_started.ipynb.
# The wheel bundles CUDA runtime libraries, but the host still needs a working
# NVIDIA driver.
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project, notebook/display helpers, and research/metric extras.
python3 -m pip install -e . matplotlib pickleshare
python3 -m pip install -e ".[research]"

# FFmpeg with libvmaf. Most system FFmpeg builds do not include this filter.
FFMPEG_DIR="$PWD/notebook_tools/ffmpeg"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
ARCHIVE_PATH="$FFMPEG_DIR/ffmpeg-linux64-gpl.tar.xz"

mkdir -p "$FFMPEG_DIR"

ffmpeg_bin="$(find "$FFMPEG_DIR" -type f -path '*/bin/ffmpeg' 2>/dev/null | sort | head -n 1 || true)"
if [[ -z "$ffmpeg_bin" ]]; then
    echo "Downloading $FFMPEG_URL"
    python3 - <<PY
import urllib.request
urllib.request.urlretrieve("$FFMPEG_URL", "$ARCHIVE_PATH")
PY
    tar -xf "$ARCHIVE_PATH" -C "$FFMPEG_DIR"
    ffmpeg_bin="$(find "$FFMPEG_DIR" -type f -path '*/bin/ffmpeg' | sort | head -n 1)"
fi

chmod +x "$ffmpeg_bin" || true

if ! "$ffmpeg_bin" -hide_banner -filters | grep -q ' libvmaf '; then
    echo "ERROR: downloaded FFmpeg does not expose libvmaf: $ffmpeg_bin" >&2
    exit 1
fi

export FFMPEG_BIN="$ffmpeg_bin"
export VMAF_BACKEND=ffmpeg
export VMAF_TIMEOUT_SEC="${VMAF_TIMEOUT_SEC:-180}"

echo "Using FFmpeg: $FFMPEG_BIN"

python3 - <<'PY'
import importlib.util
import os
import subprocess

required = ["torch", "torchvision", "matplotlib", "pandas", "IQA_pytorch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing packages: {missing}")

import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

ffmpeg_bin = os.environ["FFMPEG_BIN"]
filters = subprocess.run(
    [ffmpeg_bin, "-hide_banner", "-filters"],
    check=True,
    capture_output=True,
    text=True,
).stdout
print("ffmpeg:", ffmpeg_bin)
print("libvmaf:", "available" if " libvmaf " in filters else "missing")
PY
```

For `scripts/batch_attack.sh` and remote attack jobs, the important part is that
the binary is cached under `notebook_tools/ffmpeg`. The job bootstrap checks
system FFmpeg first, then this cache, and exports `FFMPEG_BIN`/`VMAF_BACKEND`
inside the Slurm task.

To force a specific binary for manual runs in the current shell:

```bash
export FFMPEG_BIN="$(find "$PWD/notebook_tools/ffmpeg" -type f -path '*/bin/ffmpeg' | sort | head -n 1)"
export VMAF_BACKEND=ffmpeg
export VMAF_TIMEOUT_SEC=180
```

To disable FFmpeg bootstrapping inside remote attack jobs:

```bash
export VCR_BENCH_BOOTSTRAP_FFMPEG=0
```
