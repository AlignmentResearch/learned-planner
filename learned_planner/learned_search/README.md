## Reproducing results from the Interpreting learned search paper

"Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban". ([arXiv](https://arxiv.org/abs/2506.10138))

### Channel groups

We interpret and group all the channels in DRC(3, 3) into five categories based on their functionality: `box`, `agent`, `Misc plan`, `entity`, and `no-label`. These groups are defined in the `learned_planner/interp/channel_group.py` file, that contains channel description, whether they store long- or short-term information, and their sign of activation. Each channel in the model is spatially-offset and we provide the offset values in the `learned_planner/interp/offset_fns.py` file.

### Causal intervention on channel groups

We verify our interpretation of the channel groups by performing causal intervention on the channel groups. The script `ci_score_direction_channel.py` performs the causal intervention on the channel groups. `--channel_type` can be `box`, `agent`, `box_agent`, `nfa`, or `mpa`. See its arguments for more details.

```bash
python learned_planner/learned_search/ci_score_direction_channel.py --channel_type box_agent
```

### Short and Long-term channel AUC

We verify the short- and long-term channel storing future actions at different horizon length by checking their AUC score for predicting actions `N` steps in the future. The script `future_accuracy_channels.py` performs this analysis.

```bash
python learned_planner/learned_search/future_accuracy_channels.py
```

### Kernel visualization

The kernel visualization is done using the `kernel_visualization.py` script. The script will generate the plots for the linear, turn plan extension kernels and the winner takes all kernel figure in the `new_plots` directory.

```bash
python learned_planner/learned_search/kernel_visualization.py
```

### Turn stabilization ablation

The headline figure with the two paths and the figure showing the ablation of turn kernels is generated using the `turn_stabiliization_ablation.py` script.

```bash
python learned_planner/learned_search/turn_stabiliization_ablation.py
```

### Backtracking mechanism

The backtracking mechanism is visualized using the `backtrack_mechanism.py` script.
The quantitative results are generated using the `backtrack_quant.py` script.

```bash
python learned_planner/learned_search/backtrack_mechanism.py
python learned_planner/learned_search/backtrack_quant.py
```

### Plan stopping

The plan stopping signals are visualized using the `plan_stopping.py` script.

```bash
python learned_planner/learned_search/plan_stopping.py
```

## Citation

If you use this code, please cite our work:

```bibtex
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
