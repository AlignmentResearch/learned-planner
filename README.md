# Learned Planner & Interpreting Learned Search

![Learned Planner - Level18](https://github.com/user-attachments/assets/764939ec-1cb7-482d-a42d-72609aa76b23)


This repository contains the evaluation and interpretability code for the papers:
- "Planning behavior in a recurrent neural network that plays Sokoban". ([OpenReview-ICML-MI-Workshop](https://openreview.net/forum?id=T9sB3S2hok)) ([arXiv](https://arxiv.org/abs/2407.15421))
  - This paper shows that the DRC(3, 3) represents its plan causally, which can be found using linear probes trained to predict all future moves. We also show that the network *deliberately* paces around in cycles at the start of difficult levels to get a better plan using more compute. The network also generalizes to 3-4x level sizes and number of boxes.
- "Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban". ([arXiv](https://arxiv.org/abs/2506.10138))
  - This paper reverse-engineers the planning algorithm showing the network has an internal transition model and value function, and performs a bidirectional search, thus providing a concrete agentic example of a [*mesa-optimizer*](https://arxiv.org/abs/1906.01820).

The [lp-training repository](https://github.com/AlignmentResearch/lp-training/) lets you train the neural networks on Sokoban. If you just want to train the DRC networks, you should go there.

The code for the Interpreting Learned Search paper is in the `learned_planner/learned_search` directory. Please refer to the [`learned_planner/learned_search/README.md`](learned_planner/learned_search/README.md) file for more details.

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

We install `jax[cpu]` by default. JAX is only used to obtain the cache in the `plot/behavior_analysis.py` script. To run the script on a GPU, you can install JAX on CUDA:

```bash
pip uninstall jax
pip install jax[cuda]==0.4.34
```

### Optional dependency: Envpool

We implemented a faster version of the Sokoban environment in C++ using the [Envpool](https://github.com/AlignmentResearch/envpool/) library. As per our testing, Envpool only
works on Linux as of now. We provide the python wheels for the library in the [Envpool](https://github.com/AlignmentResearch/envpool/) repository:

```bash
pip install https://github.com/AlignmentResearch/envpool/releases/download/v0.3.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl
```

To build the envpool library from source, follow the instructions in the original [documentation](https://envpool.readthedocs.io/en/latest/content/build.html) using our forked envpool version.

## Trained models

The trained DRC networks are available in our [huggingface model hub](https://huggingface.co/AlignmentResearch/learned-planner) which contains all the checkpoints for the `ResNet`, `DRC(1, 1)`, and `DRC(3, 3)` models trained with different hyperparameters. The best model for each of the model types are available at:

- DRC(3, 3):  [drc33/bkynosqi/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc33/bkynosqi/cp_2002944000)
- DRC(1, 1):  [drc11/eue6pax7/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc11/eue6pax7/cp_2002944000)
- ResNet:  [resnet/syb50iz7/cp_2002944000](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/resnet/syb50iz7/cp_2002944000)

Probes and SAEs trained on the DRC(3, 3) model are available at the same [huggingface model hub](https://huggingface.co/AlignmentResearch/learned-planner) under the `probes` and `saes` directories.


## Loading the model

First, you will need to clone the Boxoban levels. We assume that the levels are stored in the `training/.sokoban_cache` directory. If you want to change the path to the directory, you can set a new path in the `learned_planner/__init__.py` file. You can clone the levels using the following commands:

```
BOXOBAN_CACHE="training/.sokoban_cache"  # change if desired
mkdir -p "$BOXOBAN_CACHE"
git clone https://github.com/google-deepmind/boxoban-levels \
  "$BOXOBAN_CACHE/boxoban-levels-master"
```

You can load the model using the following code:

```python
import pathlib
import os

from cleanba import cleanba_impala

from learned_planner.policies import download_policy_from_huggingface
from learned_planner.interp.utils import get_boxoban_cfg

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/" # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

env_cfg = get_boxoban_cfg().make()
jax_policy, carry_t, jax_args, train_state, _ = cleanba_impala.load_train_state(MODEL_PATH, env_cfg)
```

The `jax_policy` loads the network using the JAX implementation of the DRC network in the [lp-training repository](https://github.com/AlignmentResearch/lp-training/).

This repository provides the PyTorch implementation of the DRC network compatible with [MambaLens](https://github.com/taufeeque9/MambaLens/) for doing interpretability research. You can load the model using the following code:

```python
from learned_planner.interp.utils import load_jax_model_to_torch

cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, env_cfg)
```

## Reproducing behavioral results

The behavioral results from the paper can be reproduced using the `behavior_analysis.py` script:

```bash
python plot/behavior_analysis.py
```

The script uses CPU or GPU depending on the type of JAX library installed. See the installation section for more details. This script will generate the plots in the `{output_base_path}/{model_name}/plots` directory.

The A* solutions for the [Boxoban levels](https://github.com/google-deepmind/boxoban-levels) can be found [here](https://huggingface.co/datasets/AlignmentResearch/boxoban-astar-solutions).

## Probes and SAEs

### Collecting the activations

For training the probes, we first need to generate the dataset of model activations. The `learned_planner/interp/collect_dataset.py` script can be used to cache the activations. Activations of each level are stored in a separate pickle object of the class `learned_planner.interp.collect_dataset.DatasetStore`. The `DatasetStore` The script uses the [DRC(3, 3)](https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc33/bkynosqi/cp_2002944000) model by default. See the script for additional options.

```bash
python learned_planner/interp/collect_dataset.py --boxoban_cache {BOXOBAN_CACHE} --output_path {activation_cache_path}
```

### Creating the dataset for the probes

The `learned_planner/interp/save_ds.py` script takes the activations path using `--dataset_path` and `--labels_type` to create the dataset for the probe with the specified type and saves the torch dataset `learned_planner.interp.train_probes.ActivationsDataset` in the same dataset path. See the script for additional options.

```bash
python learned_planner/interp/save_ds.py --dataset_path {activation_cache_path} --labels_type {labels_type}
```

### Training the probes or SAEs

The files provided in `experiments/probes/` defines the hyperparameter search space for different probes. Running the files will train a probe with each hyperparameter configuration. The `plot/interp/probes/probe_hp_search.py` script can be used to plot the results of the hyperparameter search and pick the best probe on the validation set. The scripts in experiments directory run the appropriate shell command to train the probes. Alternatively, you can directly train the probes using the command below. The default config is available in `learned_planner/configs/train_probe.py`. You can overwrite arguments in the config using the `cmd.{argument}={value}` syntax. 
```bash
WANDB_MODE=disabled python -m learned_planner --from-py-fn=learned_planner.configs.train_probe:train_local cmd.train_on.layer={layer} cmd.train_on.dataset_name={dataset_name} cmd.dataset_path={dataset_path}
```

The files provided in `experiments/sae/` defines the hyperparameter search space for training the SAEs.

### Evaluating the probes or SAEs

The `plot/interp/probes/` directory contains the scripts to evaluate the different types of probes in multiple ways. These are main scripts used to evaluate the results in the paper:
- `evaluate_probe`: evaluates the precision, recall, and F1 scores of the probes on a dataset.
- `ci_score_direction_probe`: computes the causal intervention score for box or agent direction probes by modifying one single direction (move) in the plan using the probe and checking whether the agent follows the modified plan.
- `ci_score_box_target_probe`: computes the causal intervention score for next_box or next_target probes.
- `ci_score_from_csv`: The above scripts save the results in a CSV file. This script can be used to compute the average and best case CI scores from the CSV files.
- `measure_plan_quality`: computes the plan quality of the boxes directions probe across thinking steps.
- `measure_plan_recall`: computes the plan recall of the boxes directions probe across thinking steps.

The `plot/interp/evaluate_features.py` script can be used to find interpretable features in channels, SAE feature neurons. Probes can also be evaluated in this script.


### Visualizations

The `plot/interp/save_{probe/sae}_videos.py` script can be used to save the videos of probes / SAEs features.


## Citation

If you use this code, please cite our work:

```bibtex
@misc{taufeeque2025planningrecurrentneuralnetwork,
      title={Planning in a recurrent neural network that plays Sokoban}, 
      author={Mohammad Taufeeque and Philip Quirke and Maximilian Li and Chris Cundy and Aaron David Tucker and Adam Gleave and Adrià Garriga-Alonso},
      year={2025},
      eprint={2407.15421},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15421}, 
}

@misc{taufeeque2025interpretinglearnedsearchfinding,
      title={Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban}, 
      author={Mohammad Taufeeque and Aaron David Tucker and Adam Gleave and Adrià Garriga-Alonso},
      year={2025},
      eprint={2506.10138},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10138}, 
}
```
