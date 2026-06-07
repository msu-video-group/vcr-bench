#!/usr/bin/env python
"""Build a small, redistributable demo dataset for VCR-Bench.

Pipeline
--------
1. For each target action in ``scripts/demo_sources.toml``, query Wikimedia Commons and
   keep only files whose license is CC0 / public domain (configurable).
2. Download candidates and trim a short clip from each with ffmpeg.
3. Pseudo-label every clip with a classifier ensemble (top Kinetics-400 models in this
   repo). Models are loaded **one at a time** (memory-safe on small GPUs); a clip is
   accepted only if the models that scored it agree on the top-1 class above a confidence
   threshold.
4. Package the accepted clips into a ZIP laid out for the ``kinetics400`` adapter, plus a
   ``PROVENANCE.csv`` recording the source/author/license of every clip.

The produced archive is meant to be uploaded as the ``kinetics400 / demo`` subset
(``redistributable = true`` in ``configs/datasets.toml``).

IMPORTANT: the labels are *pseudo-labels* produced by models, not human ground truth. The
demo set is for demonstrating attacks/defences, not for reproducing benchmark numbers.

Examples
--------
    # Dry run: only search Commons and print license-clean candidates (no models, no GPU)
    python scripts/build_demo_dataset.py --dry-run

    # Full build on GPU, then inspect data/k400_demo_cc0_build/
    python scripts/build_demo_dataset.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # allow `python scripts/...` without editable install
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VCR-Bench-demo-builder/1.0 (https://github.com/; research dataset tooling)"


# ---------------------------------------------------------------------------
# Wikimedia Commons
# ---------------------------------------------------------------------------
def _api_get(params: dict[str, str]) -> dict[str, Any]:
    params = {**params, "format": "json"}
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def commons_search(term: str, limit: int) -> list[str]:
    """Return File: titles matching ``term`` restricted to video files."""
    data = _api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": f"{term} filetype:video",
            "srnamespace": "6",  # File namespace
            "srlimit": str(limit),
        }
    )
    hits = data.get("query", {}).get("search", [])
    return [h["title"] for h in hits if "title" in h]


def commons_imageinfo(titles: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch url + license + size/duration metadata for a batch of File: titles."""
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(titles), 50):  # API accepts <=50 titles per call
        batch = titles[start : start + 50]
        data = _api_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mediatype|dimensions",
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata", {}) or {}
            out[page.get("title", "")] = {
                "url": info.get("url"),
                "mediatype": info.get("mediatype"),
                "size": info.get("size"),
                "duration": info.get("duration"),
                "width": info.get("width"),
                "height": info.get("height"),
                "license": (ext.get("License", {}) or {}).get("value", ""),
                "license_short": (ext.get("LicenseShortName", {}) or {}).get("value", ""),
                "usage_terms": (ext.get("UsageTerms", {}) or {}).get("value", ""),
                "artist": (ext.get("Artist", {}) or {}).get("value", ""),
                "descriptionurl": info.get("descriptionurl"),
            }
    return out


import re

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return _TAG_RE.sub("", text or "").replace("\n", " ").strip()


def classify_license(meta: dict[str, Any], cc0_tokens: list[str], allow_cc_by: bool) -> str | None:
    """Return 'cc0' for public-domain/CC0, 'cc-by' for plain CC BY (when allowed), else None.

    Rejects ShareAlike / NonCommercial / NoDerivatives even if 'CC BY' appears as a substring.
    """
    hay = " ".join(
        str(meta.get(k, "")).lower() for k in ("license", "license_short", "usage_terms")
    )
    if any(tok.lower() in hay for tok in cc0_tokens):
        return "cc0"
    restricted = any(
        t in hay for t in (
            "noncommercial", "non-commercial", "-nc", " nc ",
            "noderiv", "no deriv", "-nd", " nd ",
            "sharealike", "share alike", "-sa", " sa ",
        )
    )
    if restricted:
        return None
    if allow_cc_by and ("cc by" in hay or "cc-by" in hay or "attribution" in hay):
        return "cc-by"
    return None


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def find_tool(name: str, override: str | None) -> str:
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    vendored = list(REPO_ROOT.glob(f"notebook_tools/ffmpeg/**/bin/{name}.exe"))
    if vendored:
        return str(vendored[0])
    raise FileNotFoundError(
        f"{name} not found. Install it or pass --{name}. On Windows see "
        "scripts/install_vmaf_ffmpeg.sh / notebook_tools/ffmpeg."
    )


def probe_duration(ffprobe: str | None, path: Path) -> float | None:
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def trim_clip(ffmpeg: str, src: Path, dst: Path, start: float, seconds: float, fps: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = "scale=-2:256"
    cmd = [ffmpeg, "-y", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{seconds:.2f}"]
    if fps:
        cmd += ["-r", str(fps)]  # standardise frame rate (Kinetics-like ~10s x 30fps)
    cmd += ["-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-loglevel", "error", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def mean_satmax(ffmpeg: str, path: Path) -> float | None:
    """Average peak chroma (SATMAX) across ~1 fps samples.

    Black & white clips have a low SATMAX even when dim colour scenes keep a low *average*
    saturation, so SATMAX discriminates B&W far more reliably than SATAVG (empirically B&W
    ~4-9 vs colour ~40-99).
    """
    cmd = [
        ffmpeg, "-i", str(path), "-vf", "fps=1,signalstats,metadata=print:file=-",
        "-an", "-f", "null", "-", "-loglevel", "error",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    vals = []
    for line in (res.stdout or "").splitlines():
        if "SATMAX" in line:
            try:
                vals.append(float(line.split("=")[-1].strip()))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None


def download(url: str, dst: Path, max_bytes: int | None) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180) as resp, dst.open("wb") as out:
            read = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                read += len(chunk)
                if max_bytes and read > max_bytes:
                    return False  # too large; caller discards
                out.write(chunk)
        return True
    except Exception as exc:
        print(f"    ! download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------
def load_label_map(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_label_map(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        REPO_ROOT / "mmaction2" / "tools" / "data" / "kinetics" / "label_map_k400.txt",
        REPO_ROOT / "data" / "kinetics400" / "demo" / "k400_demo_cc0" / "annotations" / "k400_label_map_k400.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the HF copy used by the dataset subsets.
    print("  label map not found locally; fetching k400_label_map_k400.txt from HF ...")
    from huggingface_hub import hf_hub_download

    dst = REPO_ROOT / "data" / "_meta"
    dst.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="maxv65/vcr-bench", filename="k400_label_map_k400.txt",
        repo_type="model", local_dir=str(dst),
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=str(REPO_ROOT / "scripts" / "demo_sources.toml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "k400_demo_cc0_build"))
    p.add_argument("--label-map", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    p.add_argument("--limit-classes", type=int, default=None, help="Only process the first N classes")
    p.add_argument("--dry-run", action="store_true", help="Only search Commons + list license-clean candidates")
    p.add_argument("--no-package", action="store_true", help="Skip building the final ZIP")
    args = p.parse_args()

    # Commons titles/authors contain non-ASCII; avoid cp1251 console crashes on Windows.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    with open(args.sources, "rb") as f:
        cfg = tomllib.load(f)
    settings = cfg.get("settings", {})
    cc0_tokens = [str(x) for x in settings.get("license_allow", ["cc0", "public domain"])]
    allow_cc_by = bool(settings.get("allow_cc_by", False))
    per_class = int(settings.get("per_class_target", 5))
    max_cand = int(settings.get("max_candidates_per_class", 12))
    clip_seconds = float(settings.get("clip_seconds", 10.0))
    output_fps = int(settings.get("output_fps", 30))
    min_src = float(settings.get("min_source_seconds", 4.0))
    max_source_mb = float(settings.get("max_source_mb", 200.0))
    min_height = int(settings.get("min_source_height", 480))
    min_satmax = float(settings.get("min_color_satmax", 20.0))
    threshold = float(settings.get("confidence_threshold", 0.3))
    unanimous = bool(settings.get("require_unanimous", True))
    require_target = bool(settings.get("require_target_match", False))
    ensemble_specs = cfg.get("ensemble", [])
    sources = cfg.get("sources", [])
    if args.limit_classes:
        sources = sources[: args.limit_classes]
    max_source_bytes = int(max_source_mb * 1024 * 1024) if max_source_mb else None

    ffmpeg = find_tool("ffmpeg", args.ffmpeg)
    ffprobe = None if args.dry_run else find_tool("ffprobe", args.ffprobe)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_raw"
    clips_dir = out_dir  # videos live at the archive root (video_root = ".")
    ann_dir = out_dir / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1+2: search & trim candidates ------------------------------------------------
    candidates: list[dict[str, Any]] = []
    for src in sources:
        target = src["target_class"]
        term = src.get("search", target)
        print(f"[search] {target!r}  <= {term!r}", flush=True)
        try:
            titles = commons_search(term, max_cand * 3)
            infos = commons_imageinfo(titles)
        except Exception as exc:
            print(f"  ! Commons query failed: {exc}", flush=True)
            continue
        kept = 0
        for title, meta in infos.items():
            if kept >= max_cand:
                break
            lic_kind = classify_license(meta, cc0_tokens, allow_cc_by)
            if not meta.get("url") or lic_kind is None:
                continue
            size = meta.get("size") or 0
            if max_source_bytes and size and size > max_source_bytes:
                continue
            h = meta.get("height") or 0
            w = meta.get("width") or 0
            if min_height and h and w and min(h, w) < min_height:
                continue  # too low-res to resemble Kinetics
            api_dur = meta.get("duration")
            if api_dur and api_dur < min_src:
                continue
            if args.dry_run:
                print(f"    OK  {title}  [{meta.get('license_short')}]  {meta.get('descriptionurl')}")
                kept += 1
                continue
            ext = os.path.splitext(urllib.parse.urlparse(meta["url"]).path)[1] or ".webm"
            vid = uuid.uuid4().hex[:11]
            src_path = raw_dir / f"{vid}{ext}"
            if not download(meta["url"], src_path, max_source_bytes):
                src_path.unlink(missing_ok=True)
                continue
            dur = api_dur or probe_duration(ffprobe, src_path)
            if dur is not None and dur < min_src:
                src_path.unlink(missing_ok=True)
                continue
            start = max(0.0, (dur / 2 - clip_seconds / 2)) if dur else 0.0
            clip_path = clips_dir / f"demo_{vid}_000000_{int(clip_seconds):06d}.mp4"
            ok = trim_clip(ffmpeg, src_path, clip_path, start, clip_seconds, output_fps)
            src_path.unlink(missing_ok=True)
            if not ok:
                continue
            satmax = mean_satmax(ffmpeg, clip_path)
            if min_satmax and satmax is not None and satmax < min_satmax:
                clip_path.unlink(missing_ok=True)  # black & white / near-grayscale
                continue
            candidates.append({
                "clip_path": clip_path, "video_id": vid, "target": target, "title": title,
                "url": meta.get("descriptionurl") or meta.get("url"),
                "license": meta.get("license_short") or meta.get("license"),
                "license_kind": lic_kind,
                "artist": strip_html(str(meta.get("artist", ""))),
            })
            kept += 1
        print(f"  collected {kept} license-clean candidate(s)")

    if args.dry_run:
        print(f"\nDry run complete. {len(sources)} classes queried.")
        return

    if not candidates:
        print("No candidates collected; nothing to label.", file=sys.stderr)
        sys.exit(1)

    # 3: pseudo-label (one model in VRAM at a time) --------------------------------
    import torch
    from vcr_bench.models import create_model

    labels = load_label_map(resolve_label_map(args.label_map))

    scores: dict[str, list[tuple[int, float] | None]] = {c["video_id"]: [] for c in candidates}
    models_used = 0
    for spec in ensemble_specs:
        name = spec["model"]
        print(f"\n[label] loading {name} ({spec.get('backbone')}) on {args.device} ...", flush=True)
        try:
            model = create_model(
                name, backbone=spec.get("backbone"),
                weights_dataset=spec.get("weights_dataset", "kinetics400"),
                device=args.device,
            ).eval()
        except Exception as exc:
            print(f"  ! skipping {name}: {exc}")
            continue
        ok = 0
        for c in candidates:
            try:
                with torch.no_grad():
                    bundle = model.predict(model.load_video(str(c["clip_path"])), return_full=True)
                probs = bundle.probs.reshape(-1)
                idx = int(torch.argmax(probs).item())
                scores[c["video_id"]].append((idx, float(probs[idx].item())))
                ok += 1
            except Exception as exc:
                scores[c["video_id"]].append(None)
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
        models_used += 1
        print(f"  scored {ok}/{len(candidates)} clips with {name}")
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if models_used == 0:
        print("No ensemble model could be loaded; aborting.", file=sys.stderr)
        sys.exit(1)
    need_agreement = models_used >= 2

    accepted: list[dict[str, Any]] = []
    per_class_count: dict[str, int] = {}
    for c in candidates:
        if per_class_count.get(c["target"], 0) >= per_class:
            c["clip_path"].unlink(missing_ok=True)
            continue
        votes = [v for v in scores[c["video_id"]] if v is not None]
        if not votes:
            c["clip_path"].unlink(missing_ok=True)
            continue
        idxs = [v[0] for v in votes]
        confs = [v[1] for v in votes]
        if not all(cf >= threshold for cf in confs):
            c["clip_path"].unlink(missing_ok=True)
            continue
        if unanimous and need_agreement and len(set(idxs)) != 1:
            c["clip_path"].unlink(missing_ok=True)
            continue
        label_idx = max(set(idxs), key=idxs.count)
        if not (0 <= label_idx < len(labels)):
            c["clip_path"].unlink(missing_ok=True)
            continue
        label_name = labels[label_idx]
        if require_target and label_name != c["target"]:
            c["clip_path"].unlink(missing_ok=True)
            continue
        c["label_name"] = label_name
        c["confidence"] = sum(confs) / len(confs)
        accepted.append(c)
        per_class_count[c["target"]] = per_class_count.get(c["target"], 0) + 1
        print(f"  + {c['video_id']}  ->  {label_name}  (conf {c['confidence']:.2f}, {len(votes)} votes)")

    if not accepted:
        print("No clips passed the ensemble agreement filter.", file=sys.stderr)
        sys.exit(1)

    # 4: write annotations / provenance / archive ----------------------------------
    label_map_dst = ann_dir / "k400_label_map_k400.txt"
    with label_map_dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")

    ann_csv = ann_dir / "k400_demo_cc0.csv"
    with ann_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for c in accepted:
            w.writerow([c["label_name"], c["video_id"]])

    prov_csv = out_dir / "PROVENANCE.csv"
    with prov_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "pseudo_label", "target_class", "source_title", "source_url", "license", "artist"])
        for c in accepted:
            w.writerow([c["video_id"], c["label_name"], c["target"],
                        c["title"], c["url"], c["license"], c["artist"]])

    # CC BY clips legally require attribution shipped with the dataset.
    cc_by = [c for c in accepted if c.get("license_kind") == "cc-by"]
    n_cc0 = len(accepted) - len(cc_by)
    attribution_md = out_dir / "ATTRIBUTION.md"
    with attribution_md.open("w", encoding="utf-8") as f:
        f.write("# Attribution\n\n")
        f.write(f"This demo set contains {n_cc0} CC0 / public-domain clips (no attribution "
                f"required) and {len(cc_by)} CC BY clips listed below.\n")
        f.write("All clips were trimmed and re-encoded from Wikimedia Commons sources.\n\n")
        if cc_by:
            f.write("## CC BY clips\n\n")
            for c in cc_by:
                artist = c["artist"] or "Unknown author"
                f.write(f"- **{c['label_name']}** — \"{c['title']}\" by {artist}; "
                        f"{c['license']}; {c['url']}\n")

    print(f"\nAccepted {len(accepted)} clips across {len(per_class_count)} classes "
          f"(ensemble size used: {models_used}).")
    print(f"  annotations: {ann_csv}")
    print(f"  provenance:  {prov_csv}")

    if not args.no_package:
        archive = out_dir.parent / "k400_demo_cc0.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for c in accepted:
                zf.write(c["clip_path"], arcname=c["clip_path"].name)
            zf.write(ann_csv, arcname="annotations/k400_demo_cc0.csv")
            zf.write(label_map_dst, arcname="annotations/k400_label_map_k400.txt")
            zf.write(prov_csv, arcname="PROVENANCE.csv")
            zf.write(attribution_md, arcname="ATTRIBUTION.md")
        print(f"  archive:     {archive}")
        print("\nUpload it as the `kinetics400 / demo` subset (configs/datasets.toml):")
        print(f"  huggingface-cli upload maxv65/vcr-bench {archive} k400_demo_cc0.zip")


if __name__ == "__main__":
    main()
