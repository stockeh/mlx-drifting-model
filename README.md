# Drifiting Models in MLX

A generative model that evolves the pushforward distribution during training for single-step inference implemented in [MLX](https://github.com/ml-explore/mlx). Based on [Generative Models via Drifting](https://arxiv.org/abs/2602.04770).

![Swiss Roll](media/swiss_roll-figure.png)

## Setup

```bash
# install and run
uv sync
uv run main.py

# cli arguments
uv run main.py --help
options:
  -h, --help
  -d, --dataset {swiss_roll,checkerboard}
  --steps STEPS
  --lr LR
  --temp TEMP
  --x-batch-size X_BATCH_SIZE
  --y_batch_size Y_BATCH_SIZE
  --in-dim IN_DIM
  --hidden HIDDEN
  --seed SEED
  --n-snapshots N_SNAPSHOTS
```
