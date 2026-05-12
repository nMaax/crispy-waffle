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

### Experiments naming converiont

`<Algorithm>__<Datamodule>__<Trainer>__<Phase>__[<Adapter>]__[<Extra>].yaml`

`<Phase>` can take `train`, `test`, `val`, `sanitycheck` or whatever you want.

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

### Rendering

 ```bash
uv run python policy/eval.py \
    experiment=DiffusionPolicy__StackCube-v1_EEDeltaPos_cuda__default__test \
    ckpt_path=logs/.../step_035000.ckpt \
    render=live # | video
```

### Motion Planning (`mplib`) Segmentation Fault


### Pre-commit setup

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Pytest

```bash
uv run pytest --cov=policy --cov-fail-under=70
```

---

### Offline Data Generation & Motion Planning (`mplib`) Setup

For tasks like `PlaceSphere-v1` where pre-collected demos might not be readily available, you can generate your own trajectories using the built-in motion planning. It is recommended to maintain a **cloned version of ManiSkill** as an isolated "Data Generator" to avoid dependency conflicts with your main crispy-waffle clone.

### Setup Maniskill source code

First, clone the ManiSkill repository and set up a development environment using `uv`. This allows you to run example scripts and motion planning solvers that are not always packaged in the standard pip release.

```bash
# Clone the repository
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill

# Install ManiSkill in editable/dev mode using uv
uv add --dev -e .
```

#### Troubleshooting Motion Planning Segmentation Faults

When running ManiSkill motion planning scripts (e.g., `PlaceSphere-v1`, `PickCube-v1`), the process silently crashes immediately. The progress bar stays at `0%`, and the OS throws a multiprocessing warning: `resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`. This is caused by a fatal C++ segmentation fault occurring during the initialization of the `mplib` planner, driven by two specific dependency updates:

1. `mplib` relies heavily on C++ bindings. If it is forced to interact with NumPy 2.0+, it triggers an instant segfault when passing arrays between Python and C++.
2. Newer versions of `mplib` (>= 0.2.0) introduce breaking API changes, explicitly requiring a custom `mplib.pymp.Pose` object instead of standard NumPy arrays for base poses. ManiSkill natively passes NumPy arrays, causing an `incompatible function arguments` crash.

To fix this, you must pin both `numpy` and `mplib` to their stable, legacy versions within your `uv` workspace.

1. Force `uv` to downgrade and lock the dependencies in your workspace (use the `--dev` flag if your `pyproject.toml` requires it):

```bash
uv add "numpy<2.0.0" "mplib==0.1.1" --dev
```

*(Alternatively, if just working inside a standard virtual environment without a project table: `uv pip install "numpy<2.0.0" "mplib==0.1.1"`)*

#### Generating and Replaying Demonstrations

Once dependencies are pinned, you can generate trajectories. The solver will decompose the task into pick-and-place waypoints and save the result as `.h5` files.

```bash
# Generate 100 successful trajectories for PlaceSphere-v1
uv run python -m mani_skill.examples.motionplanning.panda.run -e "PlaceSphere-v1" -n 100 --only-count-success

# (Optional) Visualize the motion planning solve live/mp4 file
uv run python -m mani_skill.examples.motionplanning.panda.run -e "PlaceSphere-v1" --vis # --video instead of --vis to render as mp4 files
```

Keep this patched ManiSkill clone strictly for data generation. Once your trajectories are generated in the `demos/` folder, simply copy the `.h5` and `.json` files to your main project. Your main project can then use the latest versions of NumPy and ManiSkill without `mplib` installed, as the motion planning logic is only required during the initial offline data collection phase.
