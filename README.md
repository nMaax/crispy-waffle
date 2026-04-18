# crispy-waffle

[![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
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

### Download dataset

```bash
uv run python -m mani_skill.utils.download_demo "StackCube-v1"
```

### Git Hooks Setup

To enable automatic linting and formatting on your machine, follow these steps:

1. Run `uv run pre-commit install` to link the hooks to your local Git repository
2. Verify the setup by running `uv run pre-commit run --all-files`
