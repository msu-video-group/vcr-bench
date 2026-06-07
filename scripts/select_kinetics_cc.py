#!/usr/bin/env python
"""Select Creative-Commons Kinetics clips into a redistributable demo set.

Kinetics clips are excerpts of YouTube videos; some of those videos are uploaded under the
*Creative Commons* (CC BY 3.0) licence, which makes those specific clips redistributable
(with attribution). This script:

1. Joins locally-downloaded Kinetics clips to their ground-truth labels (annotation CSVs).
2. Looks up each clip's YouTube licence via the YouTube Data API (status.license).
3. Keeps only `creativeCommon` clips and balances them across classes.
4. Re-encodes them to a uniform format and packages the `kinetics400 / demo` subset, with
   real labels (no pseudo-labelling) plus PROVENANCE.csv and ATTRIBUTION.md (CC BY 3.0).

Needs a YouTube Data API key (free): pass --api-key or set $YT_API_KEY. See docs.

Example:
    set YT_API_KEY=...   # Windows;  export YT_API_KEY=... on Linux/Mac
    python scripts/select_kinetics_cc.py \
        --video-dir data/kinetics400/k400_val/k400_val/val \
        --annotations data/kinetics400/k400_val/k400_val/annotations/k400_val.csv \
        --video-dir data/kinetics400/k400_test/extracted \
        --annotations data/kinetics400/k400_test/k400_test.csv \
        --per-class 5 --target-total 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YT_API = "https://www.googleapis.com/youtube/v3/videos"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
# Kinetics filename: <ytid>_<start6>_<end6>.ext  (ytid itself may contain underscores)
NAME_RE = re.compile(r"^(.*)_(\d{6})_(\d{6})$")
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return _TAG_RE.sub("", text or "").replace("\n", " ").strip()


def find_tool(name: str, override: str | None) -> str:
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    vendored = list(REPO_ROOT.glob(f"notebook_tools/ffmpeg/**/bin/{name}.exe"))
    if vendored:
        return str(vendored[0])
    raise FileNotFoundError(f"{name} not found; install it or pass --{name}")


def load_annotations(paths: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    mapping[row[1]] = row[0]  # ytid -> label_name
    return mapping


def ytid_from_name(stem: str) -> str | None:
    m = NAME_RE.match(stem)
    return m.group(1) if m else None


def yt_licenses(ids: list[str], api_key: str, cache: dict[str, dict]) -> dict[str, dict]:
    """Return {ytid: {license, channel, title}} using the API, with a persistent cache."""
    todo = [i for i in ids if i not in cache]
    for start in range(0, len(todo), 50):
        batch = todo[start : start + 50]
        params = {
            "part": "status,snippet",
            "id": ",".join(batch),
            "key": api_key,
            "maxResults": "50",
        }
        url = YT_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "vcr-bench/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise SystemExit(f"YouTube API error {e.code}: {body[:400]}")
        returned = set()
        for item in data.get("items", []):
            vid = item.get("id")
            returned.add(vid)
            cache[vid] = {
                "license": item.get("status", {}).get("license", "unknown"),
                "channel": item.get("snippet", {}).get("channelTitle", ""),
                "title": item.get("snippet", {}).get("title", ""),
            }
        for vid in batch:  # deleted/private => not returned
            if vid not in returned:
                cache[vid] = {"license": "unavailable", "channel": "", "title": ""}
        print(f"  queried {min(start + 50, len(todo))}/{len(todo)} new ids", flush=True)
    return cache


def yt_licenses_ytdlp(ids: list[str], cache: dict[str, dict], ytdlp: str, cache_path: Path) -> dict[str, dict]:
    """Look up licences via yt-dlp (no API key). Slower than the API but key-free.

    yt-dlp's `license` field is the human string 'Creative Commons Attribution license
    (reuse allowed)' for CC videos and 'NA'/standard otherwise.
    """
    todo = [i for i in ids if i not in cache]
    for start in range(0, len(todo), 60):
        batch = todo[start : start + 60]
        urls = [f"https://www.youtube.com/watch?v={i}" for i in batch]
        cmd = [ytdlp, "--skip-download", "--no-warnings", "--ignore-errors",
               "--socket-timeout", "20",
               "--print", "%(id)s\t%(license)s\t%(channel)s\t%(title)s"] + urls
        res = subprocess.run(cmd, capture_output=True, text=True)
        seen = set()
        for line in (res.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            vid, lic = parts[0], parts[1]
            ch = parts[2] if len(parts) > 2 else ""
            ti = parts[3] if len(parts) > 3 else ""
            seen.add(vid)
            kind = "creativeCommon" if "creative commons" in lic.lower() else "youtube"
            cache[vid] = {"license": kind, "channel": ch, "title": ti}
        for vid in batch:
            if vid not in seen:  # deleted/private/geo-blocked
                cache[vid] = {"license": "unavailable", "channel": "", "title": ""}
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")  # incremental
        n_cc = sum(1 for v in cache.values() if v.get("license") == "creativeCommon")
        print(f"  checked {min(start + 60, len(todo))}/{len(todo)} new  (CC so far: {n_cc})", flush=True)
    return cache


def reencode(ffmpeg: str, src: Path, dst: Path, seconds: float, fps: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src), "-t", f"{seconds:.2f}"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-vf", "scale=-2:256", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-loglevel", "error", str(dst)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0 and dst.exists()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-dir", action="append", required=True, help="Kinetics video dir (repeatable)")
    p.add_argument("--annotations", action="append", required=True, help="Annotation CSV (repeatable)")
    p.add_argument("--label-map", default="data/kinetics400/k400_val/k400_val/annotations/k400_label_map_k400.txt")
    p.add_argument("--api-key", default=os.environ.get("YT_API_KEY"))
    p.add_argument("--backend", choices=["auto", "api", "yt-dlp"], default="auto",
                   help="License lookup backend. 'auto' uses the API if a key is set, else yt-dlp.")
    p.add_argument("--ytdlp", default=None, help="Path to yt-dlp (default: auto-detect)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "k400_demo_cc0_build"))
    p.add_argument("--cache", default=str(REPO_ROOT / "data" / "_meta" / "yt_license_cache.json"))
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--target-total", type=int, default=100)
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--dry-run", action="store_true", help="Only report CC counts; do not re-encode/package")
    args = p.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    backend = args.backend
    if backend == "auto":
        backend = "api" if args.api_key else "yt-dlp"
    if backend == "api" and not args.api_key:
        raise SystemExit("api backend needs a key: pass --api-key or set $YT_API_KEY")
    print(f"licence backend: {backend}")

    ann = load_annotations(args.annotations)
    # Gather clips -> (path, ytid, label)
    clips: list[tuple[Path, str, str]] = []
    for vd in args.video_dir:
        for f in Path(vd).rglob("*"):
            if f.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            yt = ytid_from_name(f.stem)
            if not yt:
                continue
            label = ann.get(yt)
            if label:
                clips.append((f, yt, label))
    print(f"local labelled clips: {len(clips)}", flush=True)

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    ids = sorted({yt for _, yt, _ in clips})
    print(f"looking up YouTube licences for {len(ids)} unique ids (backend={backend}) ...", flush=True)
    if backend == "api":
        cache = yt_licenses(ids, args.api_key, cache)
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    else:
        ytdlp = find_tool("yt-dlp", args.ytdlp)
        cache = yt_licenses_ytdlp(ids, cache, ytdlp, cache_path)

    cc = [(f, yt, lab) for (f, yt, lab) in clips if cache.get(yt, {}).get("license") == "creativeCommon"]
    by_class: dict[str, list] = defaultdict(list)
    for item in cc:
        by_class[item[2]].append(item)
    print(f"creativeCommon clips: {len(cc)} across {len(by_class)} classes")

    if args.dry_run:
        for lab in sorted(by_class, key=lambda k: -len(by_class[k]))[:30]:
            print(f"  {len(by_class[lab]):3d}  {lab}")
        print("dry run complete.")
        return

    # Balanced round-robin selection up to per-class cap and target total.
    selected: list = []
    pools = {k: list(v) for k, v in by_class.items()}
    counts: dict[str, int] = defaultdict(int)
    progress = True
    while progress and len(selected) < args.target_total:
        progress = False
        for lab in sorted(pools):
            if pools[lab] and counts[lab] < args.per_class and len(selected) < args.target_total:
                selected.append(pools[lab].pop())
                counts[lab] += 1
                progress = True
    print(f"selected {len(selected)} clips across {len(counts)} classes "
          f"(<= {args.per_class}/class, target {args.target_total})")

    # Build the archive layout (same as the kinetics400 'demo' subset expects).
    out_dir = Path(args.out_dir)
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_tool("ffmpeg", args.ffmpeg)
    labels_master = [l.strip() for l in open(args.label_map, encoding="utf-8") if l.strip()]

    rows = []
    written = []
    for src, yt, label in selected:
        vid = uuid.uuid4().hex[:11]
        dst = out_dir / f"demo_{vid}_000000_{int(args.clip_seconds):06d}.mp4"
        if not reencode(ffmpeg, src, dst, args.clip_seconds, args.fps):
            continue
        meta = cache.get(yt, {})
        rows.append({"vid": vid, "label": label, "yt": yt,
                     "channel": strip_html(meta.get("channel", "")), "title": strip_html(meta.get("title", ""))})
        written.append(dst)
    print(f"re-encoded {len(written)} clips")

    with (ann_dir / "k400_demo_cc0.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["label"], r["vid"]])
    with (ann_dir / "k400_label_map_k400.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(labels_master) + "\n")

    with (out_dir / "PROVENANCE.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "youtube_id", "youtube_url", "channel", "license"])
        for r in rows:
            w.writerow([r["vid"], r["label"], r["yt"],
                        f"https://www.youtube.com/watch?v={r['yt']}", r["channel"], "CC BY 3.0"])

    with (out_dir / "ATTRIBUTION.md").open("w", encoding="utf-8") as f:
        f.write("# Attribution\n\n")
        f.write(f"{len(rows)} clips, each a Creative Commons (CC BY 3.0) YouTube video, "
                f"trimmed and re-encoded. Attribution per clip:\n\n")
        for r in rows:
            f.write(f"- **{r['label']}** — \"{r['title']}\" by {r['channel'] or 'Unknown'} "
                    f"(https://www.youtube.com/watch?v={r['yt']}), CC BY 3.0\n")

    archive = out_dir.parent / "k400_demo_cc0.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in written:
            zf.write(d, arcname=d.name)
        zf.write(ann_dir / "k400_demo_cc0.csv", arcname="annotations/k400_demo_cc0.csv")
        zf.write(ann_dir / "k400_label_map_k400.txt", arcname="annotations/k400_label_map_k400.txt")
        zf.write(out_dir / "PROVENANCE.csv", arcname="PROVENANCE.csv")
        zf.write(out_dir / "ATTRIBUTION.md", arcname="ATTRIBUTION.md")
    print(f"archive: {archive}")
    print("Upload: huggingface-cli upload maxv65/vcr-bench "
          f"{archive} k400_demo_cc0.zip")


if __name__ == "__main__":
    main()
