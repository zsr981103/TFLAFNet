# -*- coding: utf-8 -*-
"""
直达波前保留原始噪声 / 在 residual 中替换直达波前噪声段。

符号约定（vmd 为降噪结果）：
- 残差：sea - vmd。
- preserve_sea：直达波前 sea，直达波后 vmd（降噪记录）。
- preserve_sea_post_diff：直达波前 sea，直达波后 sea - vmd（差值剖面）。

典型用法：
1) 差值剖面：--mode preserve_sea_post_diff
2) 前 sea 后 vmd：--mode preserve_sea
3) replace_noise：改 noise 后 vmd+noise；可选 synthetic

依赖：numpy；可选 scipy 用于更平滑的滤波。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from get_patches import detect_direct_wave


def pick_onsets(
    data: np.ndarray,
    trace_axis: int,
    method: Literal["jump", "sta_lta"],
    **kwargs,
) -> np.ndarray:
    """兼容别名：调用 get_patches.detect_direct_wave。"""
    return detect_direct_wave(data, trace_axis=trace_axis, method=method, **kwargs)


def merge_preserve_sea_before_onset(
    sea: np.ndarray,
    vmd: np.ndarray,
    onsets: np.ndarray,
    trace_axis: int,
) -> np.ndarray:
    """直达波前 = sea，直达波后 = 降噪 vmd。该段残差噪声为 (sea - vmd)。"""
    sea = np.asarray(sea, dtype=np.float64)
    vmd = np.asarray(vmd, dtype=np.float64)
    out = np.array(vmd, copy=True)
    if trace_axis == 0:
        for i, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[1]))
            out[i, :t0] = sea[i, :t0]
    else:
        for j, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[0]))
            out[:t0, j] = sea[:t0, j]
    return out


def merge_preserve_sea_post_diff(
    sea: np.ndarray,
    vmd: np.ndarray,
    onsets: np.ndarray,
    trace_axis: int,
) -> np.ndarray:
    """直达波前 = sea，直达波后 = sea - vmd（逐道）。"""
    sea = np.asarray(sea, dtype=np.float64)
    vmd = np.asarray(vmd, dtype=np.float64)
    diff = sea - vmd
    out = np.array(diff, copy=True)
    if trace_axis == 0:
        for i, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[1]))
            out[i, :t0] = sea[i, :t0]
    else:
        for j, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[0]))
            out[:t0, j] = sea[:t0, j]
    return out


def merge_replace_noise_before_onset(
    sea: np.ndarray,
    vmd: np.ndarray,
    onsets: np.ndarray,
    trace_axis: int,
    synthetic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    noise = sea - vmd；对每道将 t < onset 的 noise 替换为 (sea - synthetic) 或仍为 sea - vmd。
    返回 (merged_noise, recombined = vmd + merged_noise)。
    """
    sea = np.asarray(sea, dtype=np.float64)
    vmd = np.asarray(vmd, dtype=np.float64)
    noise = sea - vmd
    if synthetic is not None:
        syn = np.asarray(synthetic, dtype=np.float64)
        if syn.shape != sea.shape:
            raise ValueError(f"synthetic 形状 {syn.shape} 与 sea {sea.shape} 不一致")
        pre = sea - syn
    else:
        pre = sea - vmd

    merged = np.array(noise, copy=True)
    if trace_axis == 0:
        for i, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[1]))
            merged[i, :t0] = pre[i, :t0]
    else:
        for j, t0 in enumerate(onsets):
            t0 = int(np.clip(t0, 0, sea.shape[0]))
            merged[:t0, j] = pre[:t0, j]

    return merged, vmd + merged


def load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return np.load(str(path))


def main() -> None:
    p = argparse.ArgumentParser(description="直达波初至检测 + 直达波前噪声处理")
    p.add_argument("--sea", type=str, default="sea_data.npy", help="原始含噪数据")
    p.add_argument("--vmd", type=str, default="vmd_data.npy", help="VMD/降噪结果")
    p.add_argument(
        "--synthetic",
        type=str,
        default="",
        help="可选：无噪/合成记录；用于直达波前段 noise=sea-synthetic",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=("preserve_sea", "preserve_sea_post_diff", "replace_noise"),
        default="preserve_sea_post_diff",
        help="preserve_sea: 前 sea 后 vmd；preserve_sea_post_diff: 前 sea 后 sea-vmd；replace_noise: vmd+合并noise",
    )
    p.add_argument(
        "--trace_axis",
        type=int,
        default=-1,
        help="震道维：0 表示 shape=(n_trace, n_time)；-1 表示 time 在 axis=-1",
    )
    p.add_argument(
        "--onset",
        type=str,
        default="jump",
        choices=("jump", "sta_lta"),
        help="初至自动拾取方法",
    )
    p.add_argument("--onset_file", type=str, default="", help="可选：每道初至索引 npy，形状 (n_trace,)")
    p.add_argument("--out", type=str, default="merged_output.npy", help="输出重组数据")
    p.add_argument("--out_noise", type=str, default="", help="保存噪声数组（replace 时为合并后；preserve 时为 sea-vmd）")
    p.add_argument(
        "--write_residual",
        action="store_true",
        help="同时将 sea-vmd 存为 与 --out 同目录、文件名加后缀 _residual_sea_minus_vmd.npy",
    )
    args = p.parse_args()

    sea = load_npy(Path(args.sea))
    vmd = load_npy(Path(args.vmd))
    if sea.shape != vmd.shape:
        raise ValueError(f"sea {sea.shape} 与 vmd {vmd.shape} 不一致")

    trace_axis = args.trace_axis
    if trace_axis < 0:
        trace_axis = sea.ndim + trace_axis

    syn = None
    if args.synthetic.strip():
        syn = load_npy(Path(args.synthetic))
        if syn.shape != sea.shape:
            raise ValueError(f"synthetic {syn.shape} 与 sea {sea.shape} 不一致")

    if args.onset_file.strip():
        onsets = np.load(args.onset_file).astype(np.int64).ravel()
        n_expect = sea.shape[0] if trace_axis == 0 else sea.shape[-1]
        if onsets.size != n_expect:
            raise ValueError(f"onset_file 长度 {onsets.size} 与道数 {n_expect} 不符")
    else:
        method = "sta_lta" if args.onset == "sta_lta" else "jump"
        onsets = detect_direct_wave(sea, trace_axis=trace_axis, method=method)

    if args.mode == "preserve_sea":
        out = merge_preserve_sea_before_onset(sea, vmd, onsets, trace_axis)
        noise_out = sea - vmd
    elif args.mode == "preserve_sea_post_diff":
        out = merge_preserve_sea_post_diff(sea, vmd, onsets, trace_axis)
        noise_out = sea - vmd
    else:
        noise_out, out = merge_replace_noise_before_onset(
            sea, vmd, onsets, trace_axis, synthetic=syn
        )

    np.save(args.out, out.astype(sea.dtype, copy=False))
    if args.out_noise.strip():
        np.save(args.out_noise, noise_out.astype(sea.dtype, copy=False))
    if args.write_residual:
        outp = Path(args.out)
        res_path = outp.with_name(outp.stem + "_residual_sea_minus_vmd.npy")
        residual = (sea - vmd).astype(sea.dtype, copy=False)
        np.save(str(res_path), residual)
        print("  残差噪声 sea-vmd:", res_path)

    np.save(str(Path(args.out).with_suffix("")) + "_onsets.npy", onsets)

    print("完成。")
    print("  sea:", sea.shape, sea.dtype)
    print("  初至统计: min/mean/max =", int(onsets.min()), f"{onsets.mean():.1f}", int(onsets.max()))
    print("  已保存:", args.out)
    print("  模式:", args.mode)
    if args.mode == "preserve_sea_post_diff":
        print("  组合: 直达波前=sea，直达波后=sea-vmd")
    elif args.mode == "preserve_sea":
        print("  组合: 直达波前=sea，直达波后=vmd；残差噪声文件仍为 sea-vmd")
    if args.out_noise.strip():
        print("  noise 文件:", args.out_noise)


if __name__ == "__main__":
    main()
