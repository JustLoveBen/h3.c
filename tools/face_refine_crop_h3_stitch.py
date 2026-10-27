#!/usr/bin/env python3
"""Minimal FaceRefine for h3.c eval clips: crop → (H3 regen | classical) → stitch.

Distant-face mush on 480p is a head-size-in-frame pixel-budget problem. Community
FaceRefine raises face pixels by cropping the face region, regenerating /
enhancing at a larger canvas, then feather-stitching back. Ref2VA / --ref-image
is identity conditioning, NOT this path.

Subcommands:
  detect   Track largest face (YuNet), write track.json + first-frame JPEG +
           optional crop-panel mp4 (face region upscaled to --refine-size).
  stitch   Paste a refined face video (same frame count) back onto the source
           using track.json with a feathered elliptical mask.
  enhance  Classical baseline: Lanczos upscale face crop + unsharp, stitch
           (no H3). Useful A/B when GFPGAN/CodeFormer weights are unavailable.

Gaps (documented, not blockers for this slice):
  - H3 path uses official --first-frame F2VA on a face-CU canvas; motion is
    regenerated, not pixel-locked img2img denoise. Temporal identity can drift
    vs body — expected for minimal FaceRefine without latent composite.
  - No CodeFormer/GFPGAN weights wired yet (download optional later).
  - Single-subject largest-face tracker only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_yunet(model: Path, w: int, h: int) -> cv2.FaceDetectorYN:
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.4, 0.3, 5000)
    return det


def detect_faces(det: cv2.FaceDetectorYN, bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    if faces is None:
        return out
    for row in faces:
        x, y, bw, bh = [float(v) for v in row[:4]]
        score = float(row[-1])
        out.append((int(x), int(y), int(bw), int(bh), score))
    return out


def pad_square_box(
    x: int, y: int, bw: int, bh: int, W: int, H: int, pad: float
) -> tuple[int, int, int, int]:
    cx = x + bw * 0.5
    cy = y + bh * 0.5
    side = max(bw, bh) * (1.0 + pad)
    # keep even
    side = max(32.0, side)
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x1 = int(round(cx + side * 0.5))
    y1 = int(round(cy + side * 0.5))
    # clamp and re-square if clipped
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    side_i = min(x1 - x0, y1 - y0)
    # recenter inside clamp
    cx_i = (x0 + x1) // 2
    cy_i = (y0 + y1) // 2
    x0 = max(0, min(W - side_i, cx_i - side_i // 2))
    y0 = max(0, min(H - side_i, cy_i - side_i // 2))
    return x0, y0, side_i, side_i


def smooth_track(
    boxes: list[tuple[int, int, int, int] | None], win: int = 5
) -> list[tuple[int, int, int, int] | None]:
    """Temporal median on present boxes; forward/back fill gaps."""
    n = len(boxes)
    # fill gaps from nearest
    filled: list[tuple[int, int, int, int] | None] = list(boxes)
    last = None
    for i in range(n):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    last = None
    for i in range(n - 1, -1, -1):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    # median smooth
    out: list[tuple[int, int, int, int] | None] = [None] * n
    half = win // 2
    for i in range(n):
        chunk = [filled[j] for j in range(max(0, i - half), min(n, i + half + 1)) if filled[j]]
        if not chunk:
            out[i] = None
            continue
        arr = np.array(chunk, dtype=np.float32)
        med = np.median(arr, axis=0)
        out[i] = tuple(int(round(v)) for v in med)  # type: ignore
    return out


def feather_mask(h: int, w: int, feather: float = 0.22) -> np.ndarray:
    """Elliptical alpha mask [0,1], float32."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    ry, rx = max(1.0, h * 0.5), max(1.0, w * 0.5)
    # normalized ellipse radius
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    # soft falloff near edge
    edge0 = 1.0 - feather
    alpha = np.clip((1.0 - r) / max(1e-6, feather), 0.0, 1.0)
    # smootherstep
    alpha = alpha * alpha * (3 - 2 * alpha)
    # zero outside ellipse
    alpha = np.where(r <= 1.0, alpha, 0.0).astype(np.float32)
    # boost interior
    alpha = np.where(r < edge0, 1.0, alpha).astype(np.float32)
    return alpha


def cmd_detect(args: argparse.Namespace) -> int:
    video = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}", file=sys.stderr)
        return 2
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    det = load_yunet(Path(args.yunet), W, H)

    raw_boxes: list[tuple[int, int, int, int] | None] = []
    frames_bgr: list[np.ndarray] = []
    face_areas: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(det, frame)
        if faces:
            # largest by area
            faces.sort(key=lambda t: t[2] * t[3], reverse=True)
            x, y, bw, bh, score = faces[0]
            box = pad_square_box(x, y, bw, bh, W, H, args.pad)
            raw_boxes.append(box)
            face_areas.append(float(bw * bh))
        else:
            raw_boxes.append(None)
            face_areas.append(0.0)
        frames_bgr.append(frame)
        i += 1
    cap.release()
    n = len(frames_bgr)
    if n == 0:
        print("no frames", file=sys.stderr)
        return 2
    track = smooth_track(raw_boxes, win=5)
    hit = sum(1 for b in track if b is not None)
    if hit == 0:
        print("no faces detected", file=sys.stderr)
        return 3

    # pick clearest (largest raw face area) for first-frame identity
    best_i = int(np.argmax(np.array(face_areas)))
    if track[best_i] is None:
        best_i = next(i for i, b in enumerate(track) if b is not None)
    bx, by, bw, bh = track[best_i]  # type: ignore
    crop = frames_bgr[best_i][by : by + bh, bx : bx + bw]
    refine = int(args.refine_size)
    if refine % 32 != 0:
        print(f"refine_size must be %32==0, got {refine}", file=sys.stderr)
        return 2
    face_cu = cv2.resize(crop, (refine, refine), interpolation=cv2.INTER_LANCZOS4)
    first_jpg = out_dir / "first_frame_face.jpg"
    cv2.imwrite(str(first_jpg), face_cu, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # optional crop panel video (source face upscaled — input to classical / debug)
    panel_path = out_dir / "face_crop_panel.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(panel_path), fourcc, fps, (refine, refine))
    for fi, box in enumerate(track):
        if box is None:
            writer.write(np.zeros((refine, refine, 3), dtype=np.uint8))
            continue
        x, y, w, h = box
        c = frames_bgr[fi][y : y + h, x : x + w]
        writer.write(cv2.resize(c, (refine, refine), interpolation=cv2.INTER_LANCZOS4))
    writer.release()

    # also dump early/late face crops for evidence
    for tag, ti in [("early", 0), ("mid", n // 2), ("best", best_i), ("late", n - 1)]:
        box = track[ti]
        if box is None:
            continue
        x, y, w, h = box
        c = frames_bgr[ti][y : y + h, x : x + w]
        cv2.imwrite(
            str(out_dir / f"crop_{tag}_f{ti:04d}.png"),
            cv2.resize(c, (refine, refine), interpolation=cv2.INTER_NEAREST),
        )

    meta = {
        "video": str(video),
        "width": W,
        "height": H,
        "fps": fps,
        "n_frames": n,
        "n_hint": n_hint,
        "pad": args.pad,
        "refine_size": refine,
        "best_frame": best_i,
        "best_face_area_px": float(face_areas[best_i]),
        "detect_hit_frames": hit,
        "first_frame_jpeg": str(first_jpg),
        "face_crop_panel": str(panel_path),
        "track": [
            None if b is None else {"x": b[0], "y": b[1], "w": b[2], "h": b[3]} for b in track
        ],
        "raw_face_area_px": face_areas,
    }
    track_path = out_dir / "track.json"
    track_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"detect: frames={n} hits={hit} best_f={best_i} "
        f"best_area={face_areas[best_i]:.0f}px refine={refine} → {track_path}"
    )
    return 0


def load_track(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paste_face(
    canvas: np.ndarray, face_bgr: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = box
    if w < 4 or h < 4:
        return canvas
    face = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    alpha = feather_mask(h, w)
    roi = canvas[y : y + h, x : x + w].astype(np.float32)
    f = face.astype(np.float32)
    a = alpha[..., None]
    blended = roi * (1.0 - a) + f * a
    out = canvas.copy()
    out[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def read_all_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    # Write via ffmpeg for CRF14 h264 (opencv mp4v is for intermediate only)
    tmp = path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    for fr in frames:
        wr.write(fr)
    wr.release()
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(tmp),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.check_call(cmd)
    tmp.unlink(missing_ok=True)


def cmd_stitch(args: argparse.Namespace) -> int:
    meta = load_track(Path(args.track))
    src_frames, fps = read_all_frames(Path(args.video))
    ref_frames, _ = read_all_frames(Path(args.refined))
    track = meta["track"]
    n = min(len(src_frames), len(ref_frames), len(track))
    if n == 0:
        print("empty", file=sys.stderr)
        return 2
    if len(src_frames) != len(ref_frames):
        print(
            f"warn: src_frames={len(src_frames)} refined={len(ref_frames)} using n={n}",
            file=sys.stderr,
        )
    out_frames = []
    for i in range(n):
        box = track[i]
        fr = src_frames[i]
        if box is None:
            out_frames.append(fr)
            continue
        b = (int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"]))
        out_frames.append(paste_face(fr, ref_frames[i], b))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(out, out_frames, fps if fps > 1 else float(meta.get("fps") or 24))
    print(f"stitch: n={n} → {out}")
    return 0


def cmd_enhance(args: argparse.Namespace) -> int:
    """Classical crop-upscale-unsharp-stitch (no generative)."""
    meta = load_track(Path(args.track))
    src_frames, fps = read_all_frames(Path(args.video))
    track = meta["track"]
    refine = int(meta["refine_size"])
    n = min(len(src_frames), len(track))
    out_frames = []
    for i in range(n):
        box = track[i]
        fr = src_frames[i]
        if box is None:
            out_frames.append(fr)
            continue
        b = (int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"]))
        x, y, w, h = b
        crop = fr[y : y + h, x : x + w]
        up = cv2.resize(crop, (refine, refine), interpolation=cv2.INTER_LANCZOS4)
        # unsharp
        blur = cv2.GaussianBlur(up, (0, 0), 1.2)
        sharp = cv2.addWeighted(up, 1.45, blur, -0.45, 0)
        out_frames.append(paste_face(fr, sharp, b))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(out, out_frames, fps if fps > 1 else float(meta.get("fps") or 24))
    print(f"enhance: n={n} → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("detect", help="Track face + export first-frame JPEG")
    d.add_argument("--video", required=True)
    d.add_argument("--yunet", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--pad", type=float, default=0.65, help="box pad fraction")
    d.add_argument("--refine-size", type=int, default=544, help="H3 canvas (%%32==0)")
    d.set_defaults(func=cmd_detect)

    s = sp.add_parser("stitch", help="Stitch refined face video onto source")
    s.add_argument("--video", required=True)
    s.add_argument("--refined", required=True)
    s.add_argument("--track", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_stitch)

    e = sp.add_parser("enhance", help="Classical Lanczos+unsharp FaceRefine baseline")
    e.add_argument("--video", required=True)
    e.add_argument("--track", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_enhance)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
