"""Generative Modeling via Drifting — MLX.
Paper: https://arxiv.org/abs/2602.04770
"""

import argparse
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

# -- CLI ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drifting Model 2D (MLX)")
    p.add_argument(
        "-d", "--dataset", default="swiss_roll", choices=["swiss_roll", "checkerboard"]
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--x-batch-size", type=int, default=2048)
    p.add_argument("--y_batch_size", type=int, default=2048)
    p.add_argument("--in-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-snapshots", type=int, default=4)
    return p.parse_args()


# -- Samplers -----------------------------------------------------------


def sample_checkerboard(n: int, noise: float = 0.05) -> mx.array:
    b = mx.random.randint(0, 2, shape=(n,))
    i = mx.random.randint(0, 2, shape=(n,)) * 2 + b
    j = mx.random.randint(0, 2, shape=(n,)) * 2 + b
    u = mx.random.uniform(shape=(n,))
    v = mx.random.uniform(shape=(n,))
    pts = mx.stack([i + u, j + v], axis=1) - 2.0
    pts = pts * 0.5
    if noise > 0:
        pts = pts + noise * mx.random.normal(pts.shape)
    return pts


def sample_swiss_roll(n: int, noise: float = 0.03) -> mx.array:
    u = mx.random.uniform(shape=(n,))
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = mx.stack([t * mx.cos(t), t * mx.sin(t)], axis=1)
    pts = pts / (mx.max(mx.abs(pts)) + 1e-8)
    if noise > 0:
        pts = pts + noise * mx.random.normal(pts.shape)
    return pts


# -- Model --------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 256, out_dim: int = 2):
        super().__init__()
        self.layers = [
            nn.Linear(in_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, out_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers[:-1]:
            x = nn.silu(layer(x))
        return self.layers[-1](x)


# -- Drift field --------------------------------------------------------


def compute_drift(y: mx.array, x: mx.array, temp: float = 0.05) -> mx.array:
    """Contrastive mean-shift drift field.

    Args:
        y: Generated samples  [G, D] (negatives)
        x: Real data samples  [P, D] (positives)
        temp: Kernel temperature (Python float — no type promotion)

    Returns:
        V: Drift vectors  [G, D]
    """
    targets = mx.concatenate([y, x], axis=0)  # [G+P, D]
    g = y.shape[0]

    # Pairwise distances  [G, G+P]
    diff = y[:, None, :] - targets[None, :, :]  # [G, G+P, D]
    dist = mx.sqrt(mx.sum(diff * diff, axis=-1) + 1e-12)

    # Mask self-distances (y↔y diagonal) with large value
    mask = mx.pad(mx.eye(g), pad_width=[(0, 0), (0, targets.shape[0] - g)])
    dist = mx.where(mask, 1e6, dist)

    # Kernel  [G, G+P]
    kernel = mx.exp(-dist / temp)

    # Batch-normalise along both dims
    norm_row = mx.sum(kernel, axis=-1, keepdims=True)  # [G, 1]
    norm_col = mx.sum(kernel, axis=-2, keepdims=True)  # [1, G+P]
    normalizer = mx.sqrt(mx.maximum(norm_row * norm_col, 1e-12))
    kernel = kernel / normalizer

    # Positive (pull toward data) and negative (push from generated)
    k_pos = kernel[:, g:]  # [G, P]
    k_neg = kernel[:, :g]  # [G, G]

    pos_coeff = k_pos * mx.sum(k_neg, axis=-1, keepdims=True)  # [G, P]
    neg_coeff = k_neg * mx.sum(k_pos, axis=-1, keepdims=True)  # [G, G]

    pos_v = pos_coeff @ targets[g:]  # [G, D]
    neg_v = neg_coeff @ targets[:g]  # [G, D]

    return pos_v - neg_v


# -- Training -----------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    media = Path("media")
    media.mkdir(exist_ok=True)

    mx.random.seed(args.seed)

    sampler = sample_swiss_roll if args.dataset == "swiss_roll" else sample_checkerboard

    model = MLP(in_dim=args.in_dim, hidden=args.hidden, out_dim=2)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x: mx.array):
        def loss_fn(model: MLP):
            z = mx.random.normal((args.y_batch_size, args.in_dim))
            y = model(z)
            v = compute_drift(y, x, args.temp)
            target = mx.stop_gradient(y + v)
            return mx.mean((y - target) ** 2)

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Snapshot schedule: n-1 spaced up to 25%, then final step
    early = mx.linspace(2, args.steps * 0.25, args.n_snapshots - 1)
    snapshot_steps = sorted(set(list(early) + [args.steps]))
    snapshots: list[tuple[int, mx.array]] = []

    loss_history: list[float] = []
    ema = 0.0
    pbar = tqdm(range(1, args.steps + 1))

    for i in pbar:
        x = sampler(args.x_batch_size)
        mx.eval(x)

        loss = step(x)
        mx.eval(state)

        loss_val = loss.item()
        loss_history.append(loss_val)
        ema = loss_val if i == 1 else 0.96 * ema + 0.04 * loss_val
        pbar.set_postfix(loss=f"{ema:.2e}")

        if i in snapshot_steps:
            z = mx.random.normal((5000, args.in_dim))
            y = mx.array(model(z))
            mx.eval(mx.random.state)
            snapshots.append((i, y))

    gt = mx.array(sampler(5000))
    path = media / f"{args.dataset}-figure.png"
    _plot_figure(loss_history, gt, snapshots, path)


# -- Plotting -----------------------------------------------------------


def _plot_figure(loss_history, target, snapshots, path):
    panels = [(s, pts, "tab:orange", f"Step {s:,}") for s, pts in snapshots]
    panels.append((None, target, "k", "Target"))
    ncols = len(panels)

    all_pts = mx.concatenate([pts for _, pts, *_ in panels])
    lo, hi = float(all_pts.min()), float(all_pts.max())
    margin = (hi - lo) * 0.01
    lim = (lo - margin, hi + margin)

    fig = plt.figure(figsize=(2.0 * ncols, 4.5))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1, 1.75], hspace=0.3)

    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(loss_history, c="k", lw=0.8, alpha=0.85)
    ax_loss.set(
        xlabel="Step", ylabel="Loss", yscale="log" if "swiss" in str(path) else "linear"
    )
    ax_loss.spines[["top", "right"]].set_visible(False)

    y0 = ax_loss.get_ylim()[0]  # bottom of loss axes in data coords

    for i, (step, pts, color, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=0.6,
            alpha=0.4,
            edgecolors="none",
            rasterized=True,
        )
        ax.set(xlim=lim, ylim=lim, aspect="equal", title=title)
        ax.axis("off")

        # only draw arrows for snapshot panels (step is not None)
        if step is not None:
            con = ConnectionPatch(
                xyA=(step, y0),  # (x=step) on bottom of loss plot
                xyB=(0.5, 1.18),  # top-center of scatter axes
                coordsA="data",
                coordsB="axes fraction",
                axesA=ax_loss,
                axesB=ax,
                arrowstyle="->",
                lw=1,
                color="tab:blue",
                alpha=1,
                clip_on=False,  # helps prevent arrow clipping
            )
            ax_loss.axvline(step, color="tab:blue", lw=1, alpha=1)
            fig.add_artist(con)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -- Entry point --------------------------------------------------------


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
