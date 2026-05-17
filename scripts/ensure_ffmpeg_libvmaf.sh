#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

FFMPEG_DIR="${VCR_BENCH_FFMPEG_DIR:-${REPO_ROOT}/notebook_tools/ffmpeg}"
FFMPEG_URL="${VCR_BENCH_FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz}"
ARCHIVE_PATH="${FFMPEG_DIR}/ffmpeg-linux64-gpl.tar.xz"
LOCK_DIR="${FFMPEG_DIR}/.install.lock"

log() {
    echo "[ensure_ffmpeg_libvmaf] $*" >&2
}

supports_libvmaf() {
    local ffmpeg_bin="$1"
    [[ -x "${ffmpeg_bin}" ]] || return 1
    "${ffmpeg_bin}" -hide_banner -filters 2>/dev/null | grep -q ' libvmaf '
}

find_cached_ffmpeg() {
    find "${FFMPEG_DIR}" -type f -path '*/bin/ffmpeg' 2>/dev/null | sort | head -n 1
}

system_ffmpeg="$(command -v ffmpeg || true)"
if [[ -n "${system_ffmpeg}" ]] && supports_libvmaf "${system_ffmpeg}"; then
    echo "${system_ffmpeg}"
    exit 0
fi

cached_ffmpeg="$(find_cached_ffmpeg)"
if [[ -n "${cached_ffmpeg}" ]] && supports_libvmaf "${cached_ffmpeg}"; then
    echo "${cached_ffmpeg}"
    exit 0
fi

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64|Linux-amd64)
        ;;
    *)
        log "unsupported platform $(uname -s) $(uname -m); set FFMPEG_BIN manually"
        exit 1
        ;;
esac

mkdir -p "${FFMPEG_DIR}"
for _ in $(seq 1 120); do
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
        break
    fi
    cached_ffmpeg="$(find_cached_ffmpeg)"
    if [[ -n "${cached_ffmpeg}" ]] && supports_libvmaf "${cached_ffmpeg}"; then
        echo "${cached_ffmpeg}"
        exit 0
    fi
    sleep 1
done

if [[ ! -d "${LOCK_DIR}" ]]; then
    log "timed out waiting for ffmpeg install lock"
    exit 1
fi

cached_ffmpeg="$(find_cached_ffmpeg)"
if [[ -n "${cached_ffmpeg}" ]] && supports_libvmaf "${cached_ffmpeg}"; then
    echo "${cached_ffmpeg}"
    exit 0
fi

log "downloading ${FFMPEG_URL}"
tmp_archive="${ARCHIVE_PATH}.tmp.$$"
FFMPEG_URL="${FFMPEG_URL}" TMP_ARCHIVE="${tmp_archive}" python3 - <<'PY'
import os
import urllib.request

urllib.request.urlretrieve(os.environ["FFMPEG_URL"], os.environ["TMP_ARCHIVE"])
PY
mv "${tmp_archive}" "${ARCHIVE_PATH}"

log "extracting ${ARCHIVE_PATH}"
tar -xf "${ARCHIVE_PATH}" -C "${FFMPEG_DIR}"

cached_ffmpeg="$(find_cached_ffmpeg)"
if [[ -z "${cached_ffmpeg}" ]]; then
    log "ffmpeg binary was not found after extraction"
    exit 1
fi
chmod +x "${cached_ffmpeg}" || true

if ! supports_libvmaf "${cached_ffmpeg}"; then
    log "downloaded ffmpeg does not expose libvmaf: ${cached_ffmpeg}"
    exit 1
fi

echo "${cached_ffmpeg}"
