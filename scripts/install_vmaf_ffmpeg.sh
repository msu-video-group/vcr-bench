#!/usr/bin/env bash
# Downloads a GPL FFmpeg build with libvmaf from BtbN FFmpeg-Builds and
# sets the env vars VCR-Bench needs to enable VMAF scoring.
#
# Usage:
#   bash scripts/install_vmaf_ffmpeg.sh              # installs to notebook_tools/ffmpeg/
#   bash scripts/install_vmaf_ffmpeg.sh /custom/path
#   source scripts/install_vmaf_ffmpeg.sh            # also exports vars into current shell

set -euo pipefail

DEST="${1:-notebook_tools/ffmpeg}"
mkdir -p "$DEST"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS-$ARCH" in
  linux-x86_64|linux-amd64)
    URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
    ARCHIVE="$DEST/ffmpeg-linux64-gpl.tar.xz"
    EXTRACT_CMD="tar -xf"
    FFMPEG_NAME="ffmpeg"
    ;;
  darwin-*)
    echo "macOS: install FFmpeg with libvmaf via Homebrew:"
    echo "  brew install ffmpeg"
    echo "Then export:"
    echo "  export FFMPEG_BIN=\$(which ffmpeg)"
    echo "  export VMAF_BACKEND=ffmpeg"
    exit 0
    ;;
  *)
    echo "Unsupported platform: $OS $ARCH"
    echo "On Windows, run the 'Install FFmpeg With libvmaf' cell in getting_started.ipynb instead."
    exit 1
    ;;
esac

FFMPEG_BIN=$(find "$DEST" -name "$FFMPEG_NAME" -type f 2>/dev/null | head -1)

if [[ -z "$FFMPEG_BIN" ]]; then
    echo "Downloading $URL ..."
    curl -L --progress-bar -o "$ARCHIVE" "$URL"
    echo "Extracting ..."
    $EXTRACT_CMD "$ARCHIVE" -C "$DEST"
    FFMPEG_BIN=$(find "$DEST" -name "$FFMPEG_NAME" -type f | head -1)
fi

chmod +x "$FFMPEG_BIN"

if ! "$FFMPEG_BIN" -hide_banner -filters 2>/dev/null | grep -q ' libvmaf '; then
    echo "ERROR: downloaded FFmpeg does not expose libvmaf: $FFMPEG_BIN"
    exit 1
fi

export FFMPEG_BIN
export VMAF_BACKEND=ffmpeg
export VMAF_TIMEOUT_SEC="${VMAF_TIMEOUT_SEC:-180}"

echo "FFmpeg with libvmaf ready: $FFMPEG_BIN"
echo ""
echo "To persist across shells, add to your ~/.bashrc or job script:"
echo "  export FFMPEG_BIN=\"$(realpath "$FFMPEG_BIN")\""
echo "  export VMAF_BACKEND=ffmpeg"
