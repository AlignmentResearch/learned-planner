# Learned Planner

![Learned Planner - Level18](https://github.com/user-attachments/assets/764939ec-1cb7-482d-a42d-72609aa76b23)


This repository contains the evaluation and interpretability code for the paper "Planning behavior in a recurrent neural network that plays Sokoban", from the ICML 2024 Mechanistic Interpretability Workshop. ([OpenReview](https://openreview.net/forum?id=T9sB3S2hok)) ([arXiv](https://arxiv.org/abs/2407.15421))

The [lp-training repository](https://github.com/AlignmentResearch/lp-training/) lets you train the neural networks on Sokoban. If you just want to train the DRC networks, you should go there.

## Installation

The repository can be installed with pip:

```bash
pip install -e .
```

We also provide a dockerfile for running the code:

```bash
docker build -t learned-planner .
docker run -it learned-planner
```

### Optional dependency: Envpool

We implemented a faster version of the Sokoban environment in C++ using the [Envpool](https://github.com/AlignmentResearch/envpool/) library. As per our testing, Envpool only
works on Linux as of now. We provide the python wheels for the library in the [Envpool](https://github.com/AlignmentResearch/envpool/) repository:

```bash
pip install https://github.com/AlignmentResearch/envpool/releases/download/v0.2.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl
```

To build the envpool library from source, follow the instructions in the original [documentation](https://envpool.readthedocs.io/en/latest/content/build.html) using our forked envpool version.

## Trained models

The trained DRC networks are available in our [huggingface model hub](https://huggingface.co/AlignmentResearch/learned-planner) which contains all the checkpoints for the `ResNet`, `DRC(1, 1)`, and `DRC(3, 3)` models trained with different hyperparameters. The best model for each of the model types are available at:

- DRC(3, 3):  [drc33/bkynosqi/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc33/bkynosqi/cp_2002944000)
- DRC(1, 1):  [drc11/eue6pax7/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc11/eue6pax7/cp_2002944000)
- ResNet:  [resnet/syb50iz7/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/resnet/syb50iz7/cp_2002944000)


## Loading the model

First, you will need to clone the Boxoban levels. We will use the `BOXOBAN_CACHE` environment variable to specify the directory

```
BOXOBAN_CACHE="/opt/sokoban_cache"  # change if desired
sudo mkdir -p "$BOXOBAN_CACHE"
sudo git clone https://github.com/google-deepmind/boxoban-levels \
  "$BOXOBAN_CACHE/boxoban-levels-master"
```

You can load the model using the following code:

```python
import pathlib
import os

from cleanba.environments import BoxobanConfig
from cleanba import cleanba_impala
from huggingface_hub import snapshot_download

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/" # DRC(3, 3) 2B checkpoint
MODEL_BASE_PATH = pathlib.Path(
    snapshot_download("AlignmentResearch/learned-planner", allow_patterns=[MODEL_PATH_IN_REPO + "*"]),
) # only download the specific model
MODEL_PATH = MODEL_BASE_PATH / MODEL_PATH_IN_REPO
BOXOBAN_CACHE=os.environ.get("BOXOBAN_CACHE", "/opt/sokoban_cache")

env = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=1,
    max_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
).make()
jax_policy, carry_t, jax_args, train_state, _ = cleanba_impala.load_train_state(MODEL_PATH, env)
```

The `jax_policy` loads the network using the JAX implementation of the DRC network in the [lp-training repository](https://github.com/AlignmentResearch/lp-training/).

This repository provides the PyTorch implementation of the DRC network compatible with [MambaLens](https://github.com/Phylliida/MambaLens/) for doing interpretability research. You can load the model using the following code:

```python
from learned_planner.interp.utils import load_jax_model_to_torch

cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, env)
```

## Reproducing paper results

The paper results can be reproduced using the `behavior_analysis.py` script:

```bash
python plot/behavior_analysis.py
```

This script will generate the plots in the `plots` directory.

The A* solutions for the [Boxoban levels](https://github.com/google-deepmind/boxoban-levels) can be found [here](https://huggingface.co/datasets/AlignmentResearch/boxoban-astar-solutions).

## Citation

If you use this code, please cite our work:

```bibtex
@inproceedings{garriga-alonso2024planning,
    title={Planning behavior in a recurrent neural network that plays Sokoban},
    author={Adri{\`a} Garriga-Alonso and Mohammad Taufeeque and Adam Gleave},
    booktitle={ICML 2024 Workshop on Mechanistic Interpretability},
    year={2024},
    url={https://openreview.net/forum?id=T9sB3S2hok}
}
```
