"""vis_mask.py

更直观地可视化 CheXWorld 里的 mask collator 输出。

背景：
- 这些 mask 都是按 patch token (H/patch_size * W/patch_size) 的索引生成的。
- 在 JEPA/MAE 语境里，通常：
  - `masks_enc` 表示 encoder 可见（keep/context）的 token indices
  - `masks_pred` 表示需要 predictor 预测的 token indices（对 encoder 来说可理解为 masked 区域）

本脚本的目标是让你“一眼看懂哪些 patch 被 mask/哪些可见”，因此默认采用**离散分类图**：
- 0: neither（不在 enc 也不在 pred）
- 1: pred-only（masked / to-predict）
- 2: enc-only（visible / context）
- 3: overlap（enc 与 pred 重叠；若策略不允许重叠通常为 0）

用法示例：

1) 画一种（并保存）
    python vis_mask.py --mask-type multiblock --save

2) 批量画全部（并保存）
    python vis_mask.py --all --save

3) 想让网格更粗：减小输入尺寸或增大 patch_size
    python vis_mask.py --mask-type multiblock --input-size 128 --patch-size 16 --save

4) 想对比旧版 3 图（pred/enc/overlay）
    python vis_mask.py --mask-type multiblock --plot three --save

提示：
- `--pixel-scale` 控制单个 patch 显示时放大倍数，越大越清楚。
- `--grid` 可以绘制 patch 网格线，进一步帮助定位。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, List

import matplotlib

# 默认用非交互式后端，避免远程/终端环境弹窗问题。
# 如果你希望弹窗显示，可以运行时加：--show
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from masks import build_mask_collator


@dataclass
class Args:
    """模拟训练时 build_mask_collator 所需参数。"""

    mask_type: str = "multiblock"
    input_size: Any = (224, 224)
    patch_size: int = 16

    pred_mask_scale: tuple = (0.2, 0.6)
    enc_mask_scale: tuple = (0.2, 0.6)

    mask_nenc: int = 2
    mask_npred: int = 2

    mask_min_keep: int = 4
    mask_max_keep: int | None = None
    mask_rand_keep: bool = False

    mask_merge: bool = False


def indices_to_grid(indices: torch.Tensor, h: int, w: int) -> np.ndarray:
    """把 flatten 后的 patch indices -> (h,w) 0/1 网格"""
    grid = torch.zeros(h * w, dtype=torch.float32)
    if indices.numel() > 0:
        grid[indices.long()] = 1.0
    return grid.view(h, w).cpu().numpy()


def upsample_grid(arr: np.ndarray, pixel_scale: int) -> np.ndarray:
    """把 (h,w) 或 (h,w,3) 的网格用 kron 放大成更易观察的像素图。"""
    if pixel_scale <= 1:
        return arr

    if arr.ndim == 2:
        return np.kron(arr, np.ones((pixel_scale, pixel_scale), dtype=arr.dtype))

    if arr.ndim == 3 and arr.shape[2] == 3:
        chs = [np.kron(arr[..., c], np.ones((pixel_scale, pixel_scale), dtype=arr.dtype)) for c in range(3)]
        return np.stack(chs, axis=-1)

    raise ValueError(f"Unsupported array shape for upsample: {arr.shape}")


def draw_patch_grid(ax: plt.Axes, h: int, w: int, pixel_scale: int, *, color: str = "k", lw: float = 0.25) -> None:
    """在 imshow 的图上画 patch 边界网格线。"""
    if pixel_scale <= 1:
        return

    H = h * pixel_scale
    W = w * pixel_scale

    # imshow 的像素中心坐标是整数，边界线在 n-0.5 处。
    for y in range(0, H + 1, pixel_scale):
        ax.axhline(y - 0.5, color=color, linewidth=lw, alpha=0.35)
    for x in range(0, W + 1, pixel_scale):
        ax.axvline(x - 0.5, color=color, linewidth=lw, alpha=0.35)


def build_discrete_mask_cmap() -> tuple[ListedColormap, BoundaryNorm, list[Patch]]:
    """离散 4 类配色 + norm + legend patches."""

    # 0 neither: 浅灰
    # 1 pred-only(masked): 红
    # 2 enc-only(visible): 蓝
    # 3 overlap: 紫
    colors = ["#f0f0f0", "#ff3b30", "#007aff", "#af52de"]
    cmap = ListedColormap(colors, name="mask_discrete")
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)

    legend = [
        Patch(facecolor=colors[0], edgecolor="k", label="neither"),
        Patch(facecolor=colors[1], edgecolor="k", label="pred-only (masked/to-predict)"),
        Patch(facecolor=colors[2], edgecolor="k", label="enc-only (visible/context)"),
        Patch(facecolor=colors[3], edgecolor="k", label="overlap"),
    ]
    return cmap, norm, legend


def _normalize_mask_groups(masks_obj) -> List[List[torch.Tensor]]:
    """把不同 collator 返回的 masks 统一成 groups: List[List[Tensor]]。

    现实里各个 collator 的返回形态不完全一致：

    1) 常见：Tensor
        - [B, K, L]  (K 个 mask)
        - [B, L]     (单个 mask)

    2) multi_multiblock_list：List[group]
        - pred 通常是 Tensor [B, K, L]
        - enc 的 group 可能是 List[Tensor]（例如 [Tensor[B, L]]），这是因为它构造时用了 list 包了一层。

    这里做一个“容错/兼容”的解析：
    - 输出固定为：List[Group]，每个 Group 是 List[mask_tensor]，mask_tensor shape = [L]
    - 只取 batch 的第 0 个样本用于可视化。
    """

    def group_to_mask_list(group) -> List[torch.Tensor]:
        # Tensor: [B,K,L] or [B,L]
        if torch.is_tensor(group):
            if group.ndim == 3:
                return [group[0, i] for i in range(group.shape[1])]
            if group.ndim == 2:
                return [group[0]]
            raise TypeError(f"Unexpected tensor shape for masks: {tuple(group.shape)}")

        # list/tuple: 可能是 [Tensor] / [Tensor, Tensor] / 更深嵌套
        if isinstance(group, (list, tuple)):
            if len(group) == 0:
                return []

            if all(torch.is_tensor(x) for x in group):
                # 常见情况：list 里每个元素代表一个 mask
                # - 若元素形状是 [B,L]，说明一个元素就是一个 mask
                # - 若元素形状是 [B,K,L]，说明这个元素里又包含 K 个 mask（少见，但也支持）
                out: List[torch.Tensor] = []
                for x in group:
                    out.extend(group_to_mask_list(x))
                return out

            # 嵌套 list：拍平
            out: List[torch.Tensor] = []
            for sub in group:
                out.extend(group_to_mask_list(sub))
            return out

        raise TypeError(f"Unexpected group type: {type(group)}")

    # masks_obj: list => 多组；tensor => 单组
    if isinstance(masks_obj, list):
        return [group_to_mask_list(g) for g in masks_obj]

    if torch.is_tensor(masks_obj):
        return [group_to_mask_list(masks_obj)]

    raise TypeError(f"Unsupported masks type: {type(masks_obj)}")


def _make_dummy_batch(input_size: Any) -> list[torch.Tensor]:
    if isinstance(input_size, tuple):
        H, W = input_size
    else:
        H = W = int(input_size)
    return [torch.randn(3, H, W)]


def visualize_one(
    mask_type: str,
    *,
    input_size: int,
    patch_size: int,
    pred_mask_scale: tuple[float, float],
    enc_mask_scale: tuple[float, float],
    nenc: int,
    npred: int,
    merge: bool,
    rand_keep: bool,
    pixel_scale: int,
    out_dir: str,
    show: bool,
    save: bool,
    group_index: int,
    plot: str,
    draw_grid: bool,
) -> None:
    args = Args(
        mask_type=mask_type,
        input_size=(input_size, input_size),
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        mask_nenc=nenc,
        mask_npred=npred,
        mask_merge=merge,
        mask_rand_keep=rand_keep,
    )

    # 一些类型不依赖这些参数，但传进去也无碍
    if mask_type == "random":
        args.mask_nenc = 1
        args.mask_npred = 1

    if mask_type == "data2vec":
        # build_mask_collator 里只读 nenc
        args.mask_npred = 1

    collator = build_mask_collator(args)

    batch = _make_dummy_batch(args.input_size)
    _, masks_enc, masks_pred = collator(batch)

    h = args.input_size[0] // args.patch_size
    w = args.input_size[1] // args.patch_size

    enc_groups = _normalize_mask_groups(masks_enc)
    pred_groups = _normalize_mask_groups(masks_pred)

    if group_index < 0 or group_index >= max(len(enc_groups), len(pred_groups)):
        raise ValueError(
            f"group_index={group_index} out of range; enc_groups={len(enc_groups)}, pred_groups={len(pred_groups)}"
        )

    enc_list = enc_groups[group_index] if group_index < len(enc_groups) else []
    pred_list = pred_groups[group_index] if group_index < len(pred_groups) else []

    enc_union = np.zeros((h, w), dtype=np.uint8)
    pred_union = np.zeros((h, w), dtype=np.uint8)

    for t in enc_list:
        enc_union = np.maximum(enc_union, indices_to_grid(t, h, w).astype(np.uint8))
    for t in pred_list:
        pred_union = np.maximum(pred_union, indices_to_grid(t, h, w).astype(np.uint8))

    # 离散 4 类：0 neither, 1 pred-only, 2 enc-only, 3 overlap
    cat = pred_union + 2 * enc_union

    # 旧 overlay（RGB）仍保留用于 three 模式
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[..., 0] = pred_union
    overlay[..., 2] = enc_union

    pred_show = upsample_grid(pred_union, pixel_scale=pixel_scale)
    enc_show = upsample_grid(enc_union, pixel_scale=pixel_scale)
    overlay_show = upsample_grid(overlay, pixel_scale=pixel_scale)
    cat_show = upsample_grid(cat, pixel_scale=pixel_scale)

    # 统计
    c0 = int((cat == 0).sum())
    c1 = int((cat == 1).sum())
    c2 = int((cat == 2).sum())
    c3 = int((cat == 3).sum())

    title_line = (
        f"mask_type={mask_type} | input={input_size} | patch={patch_size} | grid={h}x{w} | group={group_index}\n"
        f"nenc={nenc}, npred={npred}, merge={merge}, rand_keep={rand_keep}, pixel_scale={pixel_scale}"
    )

    if plot == "discrete":
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        cmap, norm, legend = build_discrete_mask_cmap()
        ax.imshow(cat_show, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(
            "discrete view\n"
            f"neither={c0}, pred-only(masked)={c1}, enc-only(visible)={c2}, overlap={c3}"
        )
        ax.axis("off")
        if draw_grid:
            draw_patch_grid(ax, h=h, w=w, pixel_scale=pixel_scale)
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        fig.suptitle(title_line)
        plt.tight_layout()

    elif plot == "three":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(pred_show, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title(f"pred union (masked/to-predict)\ncount={int(pred_union.sum())}, masks={len(pred_list)}")
        axes[0].axis("off")
        if draw_grid:
            draw_patch_grid(axes[0], h=h, w=w, pixel_scale=pixel_scale)

        axes[1].imshow(enc_show, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"enc union (visible/context)\ncount={int(enc_union.sum())}, masks={len(enc_list)}")
        axes[1].axis("off")
        if draw_grid:
            draw_patch_grid(axes[1], h=h, w=w, pixel_scale=pixel_scale)

        # 第三张换成离散分类图（更易读），而不是 RGB 混色
        cmap, norm, legend = build_discrete_mask_cmap()
        axes[2].imshow(cat_show, cmap=cmap, norm=norm, interpolation="nearest")
        axes[2].set_title(f"discrete overlay\nneither={c0}, pred-only={c1}, enc-only={c2}, overlap={c3}")
        axes[2].axis("off")
        if draw_grid:
            draw_patch_grid(axes[2], h=h, w=w, pixel_scale=pixel_scale)

        axes[2].legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)

        fig.suptitle(title_line)
        plt.tight_layout()

    else:
        raise ValueError(f"Unsupported --plot: {plot}")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"mask_{mask_type}_input{input_size}_patch{patch_size}_group{group_index}_{plot}.png",
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print("[saved]", out_path)

    if show:
        # 如果要弹窗显示，需要切回可交互后端；最简单是这里直接 plt.show()（有的环境也能弹）
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-type", type=str, default="multiblock")
    parser.add_argument("--all", action="store_true", help="依次可视化所有内置 mask_type")

    parser.add_argument("--input-size", type=int, default=128, help="建议先用小一点的，比如 128，更直观")
    parser.add_argument("--patch-size", type=int, default=16)

    parser.add_argument("--pred-mask-scale", type=float, nargs=2, default=(0.2, 0.6))
    parser.add_argument("--enc-mask-scale", type=float, nargs=2, default=(0.2, 0.6))

    parser.add_argument("--nenc", type=int, default=2)
    parser.add_argument("--npred", type=int, default=2)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--rand-keep", action="store_true")

    parser.add_argument("--pixel-scale", type=int, default=30, help="网格放大倍数，越大越清楚")
    parser.add_argument("--grid", action="store_true", help="绘制 patch 网格线")
    parser.add_argument(
        "--plot",
        type=str,
        default="discrete",
        choices=["discrete", "three"],
        help="discrete: 只画离散分类图(最清晰)；three: 画 pred/enc + 离散 overlay",
    )

    parser.add_argument("--out-dir", type=str, default="mask_vis_outputs")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true", help="尝试弹窗显示（可能受环境影响）")

    parser.add_argument(
        "--group-index",
        type=int,
        default=0,
        help="针对 multi_multiblock_list：选择展示第几组（0..nenc-1）。其它类型通常只有 0。",
    )

    args = parser.parse_args()

    mask_types = [args.mask_type]
    if args.all:
        mask_types = [
            "multiblock",
            "multi_multiblock",
            "multi_multiblock_v2",
            "multi_multiblock_list",
            "extra_multiblock",
            "data2vec",
            "random",
        ]

    for mt in mask_types:
        print("Visualizing:", mt)
        visualize_one(
            mt,
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=tuple(args.pred_mask_scale),
            enc_mask_scale=tuple(args.enc_mask_scale),
            nenc=args.nenc,
            npred=args.npred,
            merge=args.merge,
            rand_keep=args.rand_keep,
            pixel_scale=args.pixel_scale,
            out_dir=args.out_dir,
            show=args.show,
            save=args.save,
            group_index=args.group_index,
            plot=args.plot,
            draw_grid=args.grid,
        )


if __name__ == "__main__":
    main()
