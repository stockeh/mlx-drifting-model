# Generative Models via Drifting Models in MLX
paper: https://arxiv.org/html/2602.04770
mlx: https://github.com/ml-explore/mlx
mlx-docs: https://ml-explore.github.io/mlx/build/html/index.html

A generator is trained to output samples that are already at equilibrium under a drift field. Sampling requires only one forward pass.

## Core Idea

We match the model distribution q to the data distribution p using a kernel function of negative/positive. Consider the following anti-symmetric drift:

V_{p,q}(x) = V_p⁺(x) - V_q⁻(x)

such that when:

q == p => V_{p,q} = V_{q,p} = −V_{p,q} => V_{p,q} = 0

For each generated sample x:
-	Pull toward nearby real data
-	Push away from nearby generated samples

## Training Objective

Let: e ~ N(0, 1)

loss = MSE(f(e), stopgrad(f(e) + V(f(e))))

### Drift Definition

Let's define a kernel function:
- y⁺ = data sample (from p)
- y⁻ = generated sample (from q)
- k(x,y) = exp(- ||x - y|| / τ)

Then under contrastive mean-shift:

V_p⁺(x) = E[ k(x,y⁺) * (y⁺ - x) ] / Zp
V_q⁻(x) = E[ k(x,y⁻) * (y⁻ - x) ] / Zq

with normalization factors Zp, Zq gives:

V_{p,q}(x) = V_p⁺(x) - V_q⁻(x)

Note: we use a batch approximation to estimate to compute kernel similarities and normalize across batch to approximate Zp, Zq.

## Implementation

This is an astral (uv, ruff, ty) project. 
- Run code with `uv run main.py` and add dependencies with `uv add tqdm`.
- Use `uv run ruff format`, `uv run ruff check --fix`, and `uv run ty check` after modifying files.
