#!/usr/bin/env python3
"""F32 layer oracle: torch LatentResizer3D vs C h3_lbh_oracle.

Does not import Comfy. Architecture copied from
third_party/accel_survey/comfyui-SelfLift/h3_upscaler.py.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalization(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward_until(self, x, scale, size, stage):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)
        if stage == 0:
            return x
        x = self.conv_in(x)
        if stage == 1:
            return x
        if stage == 2:
            b0 = self.in_blocks[0]
            if isinstance(b0, ResBlockEmb3D):
                return b0(x, emb.expand(x.shape[0], -1))
            return b0(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        if stage == 3:
            return x
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        if stage == 4:
            return x
        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


def detect_arch(sd):
    import re
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
           "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = sd["conv_in.weight"].shape[1]
        cfg["channels"] = sd["conv_in.weight"].shape[0]
    in_ids, out_ids = set(), set()
    for k in sd:
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        if k.endswith("dwconv.weight"):
            cfg["temporal_kernel"] = sd[k].shape[2]
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    return cfg


def write_f32(path, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().astype("float32").tofile(path)


def read_f32(path, shape):
    import numpy as np
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape).copy())


def stats(a, b, name):
    d = (a.float() - b.float()).abs()
    mx = float(d.max())
    rmse = float((d * d).mean().sqrt())
    mean = float(d.mean())
    print(f"{name:16s}  max={mx:.6g}  rmse={rmse:.6g}  mean={mean:.6g}  "
          f"a={float(a.abs().mean()):.5g} b={float(b.abs().mean()):.5g}")
    return mx, rmse


def run_c(bin_path, args):
    cmd = [bin_path] + [str(x) for x in args]
    env = os.environ.copy()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"C oracle failed: {' '.join(cmd)}")
    sys.stdout.write(r.stdout)
    return r.stdout.strip()


def kernel_tests(bin_path, tmp):
    torch.manual_seed(7)
    worst = 0.0
    # conv3d NCHW 1x8x2x4x4 → 16, k=3 pad=1
    x = torch.randn(1, 8, 2, 4, 4)
    w = torch.randn(16, 8, 3, 3, 3)
    b = torch.randn(16)
    y = F.conv3d(x, w, b, padding=1)
    write_f32(f"{tmp}/c3_in.bin", x)
    write_f32(f"{tmp}/c3_w.bin", w)
    write_f32(f"{tmp}/c3_b.bin", b)
    run_c(bin_path, [
        "conv3d", 1, 8, 16, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1,
        f"{tmp}/c3_in.bin", f"{tmp}/c3_w.bin", f"{tmp}/c3_b.bin",
        f"{tmp}/c3_out.bin",
    ])
    y_c = read_f32(f"{tmp}/c3_out.bin", y.shape)
    mx, _ = stats(y, y_c, "conv3d")
    worst = max(worst, mx)

    # depthwise
    x = torch.randn(1, 8, 4, 3, 3)
    w = torch.randn(8, 1, 5, 1, 1)
    b = torch.randn(8)
    y = F.conv3d(x, w, b, padding=(2, 0, 0), groups=8)
    write_f32(f"{tmp}/dw_in.bin", x)
    write_f32(f"{tmp}/dw_w.bin", w)
    write_f32(f"{tmp}/dw_b.bin", b)
    run_c(bin_path, [
        "conv3d", 1, 8, 8, 4, 3, 3, 5, 1, 1, 2, 0, 0, 8,
        f"{tmp}/dw_in.bin", f"{tmp}/dw_w.bin", f"{tmp}/dw_b.bin",
        f"{tmp}/dw_out.bin",
    ])
    y_c = read_f32(f"{tmp}/dw_out.bin", y.shape)
    mx, _ = stats(y, y_c, "conv3d-dw")
    worst = max(worst, mx)

    gn = nn.GroupNorm(4, 8)
    x = torch.randn(1, 8, 2, 4, 4)
    y = gn(x)
    write_f32(f"{tmp}/gn_in.bin", x)
    write_f32(f"{tmp}/gn_g.bin", gn.weight.data)
    write_f32(f"{tmp}/gn_b.bin", gn.bias.data)
    run_c(bin_path, [
        "gn", 1, 8, 2, 4, 4, 4,
        f"{tmp}/gn_in.bin", f"{tmp}/gn_g.bin", f"{tmp}/gn_b.bin",
        f"{tmp}/gn_out.bin",
    ])
    y_c = read_f32(f"{tmp}/gn_out.bin", y.shape)
    mx, _ = stats(y, y_c, "groupnorm")
    worst = max(worst, mx)

    gn = nn.GroupNorm(32, 512)
    x = torch.randn(1, 512, 2, 4, 4)
    y = gn(x)
    write_f32(f"{tmp}/gn512_in.bin", x)
    write_f32(f"{tmp}/gn512_g.bin", gn.weight.data)
    write_f32(f"{tmp}/gn512_b.bin", gn.bias.data)
    run_c(bin_path, [
        "gn", 1, 512, 2, 4, 4, 32,
        f"{tmp}/gn512_in.bin", f"{tmp}/gn512_g.bin", f"{tmp}/gn512_b.bin",
        f"{tmp}/gn512_out.bin",
    ])
    y_c = read_f32(f"{tmp}/gn512_out.bin", y.shape)
    mx, _ = stats(y, y_c, "groupnorm512")
    worst = max(worst, mx)

    x = torch.randn(1, 4, 2, 3, 5)
    y = F.interpolate(x, size=(2, 6, 10), mode="trilinear", align_corners=False)
    write_f32(f"{tmp}/tr_in.bin", x)
    run_c(bin_path, [
        "trilinear", 1, 4, 2, 3, 5, 6, 10,
        f"{tmp}/tr_in.bin", f"{tmp}/tr_out.bin",
    ])
    y_c = read_f32(f"{tmp}/tr_out.bin", y.shape)
    mx, _ = stats(y, y_c, "trilinear")
    worst = max(worst, mx)
    return worst


def load_model(ckpt):
    try:
        from safetensors.torch import load_file
        sd = load_file(ckpt)
    except Exception:
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    cfg = detect_arch(sd)
    model = LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"], channels=cfg["channels"],
        dropout=cfg["dropout"], attn=False,
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"])
    model.load_state_dict(sd, strict=True)
    return model.float().eval().requires_grad_(False), cfg


def stage_tests(bin_path, ckpt, tmp):
    model, cfg = load_model(ckpt)
    print("arch", cfg)
    w = model.conv_in.weight.detach().float().contiguous().view(-1)[:8]
    print("py conv_in.weight[0:8]", " ".join(f"{float(v):.8g}" for v in w))
    run_c(bin_path, ["whead", ckpt])

    torch.manual_seed(11)
    t, h, w = 2, 4, 8
    oh, ow = 8, 16
    z = torch.randn(1, 24, t, h, w)
    mean = torch.linspace(-0.4, 0.4, 24)
    std = torch.linspace(0.5, 1.5, 24)
    write_f32(f"{tmp}/z.bin", z)
    write_f32(f"{tmp}/mean.bin", mean)
    write_f32(f"{tmp}/std.bin", std)
    mean5 = mean.view(1, 24, 1, 1, 1)
    std5 = std.view(1, 24, 1, 1, 1)
    x = (z - mean5) / std5
    scale = (oh / h + ow / w) / 2.0
    size = (t, oh, ow)
    names = {
        0: "norm_in",
        1: "conv_in",
        2: "res0",
        3: "in_blocks",
        4: "trilinear",
        5: "full_lift",
    }
    worst = 0.0
    with torch.no_grad():
        for stage in range(6):
            if stage == 5:
                y = model.forward_until(x, scale, size, 5)
                y = y * std5 + mean5
            elif stage == 0:
                y = x
            else:
                y = model.forward_until(x, scale, size, stage)
            run_c(bin_path, [
                "stage", stage, ckpt, f"{tmp}/mean.bin", f"{tmp}/std.bin",
                t, h, w, oh, ow, f"{tmp}/z.bin", f"{tmp}/s{stage}.bin",
            ])
            y_c = read_f32(f"{tmp}/s{stage}.bin", y.shape)
            mx, _ = stats(y, y_c, names[stage])
            worst = max(worst, mx)
    return worst


def fullsize_lift(bin_path, ckpt, tmp, t, h, w, oh, ow):
    """Same F32 contract as generate: VAE-native in, VAE-native out, real 768P low grid."""
    model, cfg = load_model(ckpt)
    print("fullsize arch", cfg, f"in=24x{t}x{h}x{w} out=24x{t}x{oh}x{ow}")
    torch.manual_seed(11)
    z = torch.randn(1, 24, t, h, w)
    mean = torch.linspace(-0.4, 0.4, 24)
    std = torch.linspace(0.5, 1.5, 24)
    write_f32(f"{tmp}/z.bin", z)
    write_f32(f"{tmp}/mean.bin", mean)
    write_f32(f"{tmp}/std.bin", std)
    mean5 = mean.view(1, 24, 1, 1, 1)
    std5 = std.view(1, 24, 1, 1, 1)
    x = (z - mean5) / std5
    scale = (oh / h + ow / w) / 2.0
    size = (t, oh, ow)
    with torch.no_grad():
        y = model.forward_until(x, scale, size, 5)
        y = y * std5 + mean5
    run_c(bin_path, [
        "stage", 5, ckpt, f"{tmp}/mean.bin", f"{tmp}/std.bin",
        t, h, w, oh, ow, f"{tmp}/z.bin", f"{tmp}/full.bin",
    ])
    y_c = read_f32(f"{tmp}/full.bin", y.shape)
    mx, rmse = stats(y, y_c, f"full_{t}x{h}x{w}")
    print(f"fullsize_maxabs={mx:.6g} rmse={rmse:.6g}")
    return mx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--skip-full-blocks", action="store_true")
    p.add_argument("--full-size", action="store_true",
                   help="also compare full lift at 1s 768P low grid 7x24x42→48x84")
    args = p.parse_args()
    os.makedirs("/tmp/h3_lbh_oracle", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp/h3_lbh_oracle") as tmp:
        k = kernel_tests(args.bin, tmp)
        print(f"kernel_worst_maxabs={k:.6g}")
        s = 0.0 if args.skip_full_blocks else stage_tests(args.bin, args.ckpt, tmp)
        if not args.skip_full_blocks:
            print(f"stage_worst_maxabs={s:.6g}")
        f = 0.0
        if args.full_size:
            f = fullsize_lift(args.bin, args.ckpt, tmp, 7, 24, 42, 48, 84)
        worst = max(k, s, f)
        print(f"ORACLE_WORST_MAXABS={worst:.6g}")
        if worst > 1e-3:
            sys.exit(2)


if __name__ == "__main__":
    main()
