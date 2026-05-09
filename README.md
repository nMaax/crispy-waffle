# crispy-waffle

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/mila-iqia/ResearchTemplate)

crispy-waffle: A new research project at VANDAL.

## Installation

```bash
uv sync
```

## Usage

```console
. .venv/bin/activate
python policy/main.py --help
```

or

```bash
uv run python policy/main.py --help
```


### Dataset and replay

```bash
uv run python -m mani_skill.utils.download_demo "StackCube-v1"
```

```bash
uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  -b "physx_cpu" \
  -c pd_ee_delta_pos \
  -o state \
  --save-traj
```

```bash
uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --use-first-env-state \
  -b "physx_cuda" \
  --save-traj
```

### Live rendering

 ```bash
  trainer.callbacks.rollout_evaluation.num_envs=1 \
  trainer.callbacks.rollout_evaluation.num_test_episodes=1 \
  trainer.callbacks.rollout_evaluation.video_dir=null \
  +trainer.callbacks.rollout_evaluation.render_mode="human"
```

### Pre-commit setup

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Pytest

```bash
uv run pytest --cov=policy --cov-fail-under=70
```
