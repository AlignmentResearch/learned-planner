"""Ablate non-planning channels with 1-step cache and calculate accuracy of action predictions and logits MSE.

Performs 3 forward passes: 1 for clean, 1 for getting 1-step cache, and 1 for non-planning channels ablation.
The metrics are calculated by comparing the predictions of clean and ablated models on solved levels.
"""

# %%
import argparse
import warnings
from functools import partial

import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from scipy.stats import bootstrap
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from learned_planner import BOXOBAN_CACHE
from learned_planner.interp.channel_group import get_group_channels, split_by_layer
from learned_planner.interp.utils import (
    PlayLevelOutput,
    get_cache_and_probs,
    load_jax_model_to_torch,
    run_fn_with_cache,
)
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


# %%
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

# Example command:
# For forget gate ablation:
# python long_term_plan_channels.py --ablation_channels all --ablation_gate f
parser = argparse.ArgumentParser(
    description="Ablate non-planning channels with 1-step cache and calculate accuracy of action predictions and logits MSE."
)
parser.add_argument(
    "--ablation_channels",
    type=str,
    default="non-plan",
    help="Channels to ablate. Options: 'non-plan', 'all', 'none', 'random', 'plan', 'plan-random', 'box'",
)
parser.add_argument(
    "--exclude_channels",
    type=str,
    default="",
    help="Channels to exclude from ablation. ',' separated list of channel names with layer separated by '-'. E.g. '16-1,15-3,4'",
)
parser.add_argument("--seq_len", type=int, default=120, help="Sequence length")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--ablation_gate",
    type=str,
    default="",
    help="Ablate the specified gate channels. Options: '', i, j, f, o",
)
parser.add_argument(
    "--ablation_method",
    type=str,
    default="zero",
    help="Ablation method. Options: 'zero', 'mean', 'single-step'. Default: 'cache'",
)
parser.add_argument(
    "--ablation_ticks",
    type=str,
    default="0",
    help="Ablation performed at the specified internal ticks. ',' separated list of ticks. Only used for zero-ablation on a gate.",
)
parser.add_argument(
    "--action_accuracy",
    action="store_true",
    help="Compute action accuracy on the distribution of original network's trajectories",
)
parser.add_argument("--split", type=str, default="valid", help="Dataset split.")
parser.add_argument("--difficulty", type=str, default="medium", help="Dataset difficulty.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

if __name__ == "__main__":  # Running as a script
    args = parser.parse_args()
    seq_len = args.seq_len
    num_envs = args.num_envs
    batch_size = min(args.batch_size, num_envs)
    ablation_method = args.ablation_method
    play = not args.action_accuracy
    ablation_channels = args.ablation_channels
    ablation_gate = args.ablation_gate
    exclude_channels = args.exclude_channels
    ablation_ticks = list(map(int, args.ablation_ticks.strip().split(",")))
else:
    args = parser.parse_args([])
    seq_len = 120
    num_envs = 1024
    batch_size = 128
    ablation_method = "single-step"
    play = True
    clean = True
    ablation_channels = "non-plan"  # options: 'non-plan', 'all', 'random', 'plan'
    ablation_gate = ""
    exclude_channels = ""
    ablation_ticks = [0]

clean = ablation_channels == "none"
rng = np.random.default_rng(args.seed)

boxo_cfg = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=batch_size,
    max_episode_steps=120,
    min_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
    difficulty=args.difficulty,
    split=None if args.split.lower() in ["", "none"] else args.split,
    seed=args.seed,
)
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
# check num of cuda devices
device = "cpu"
if th.cuda.is_available() and th.cuda.device_count() < 8:  # 8 devices shown on cpu nodes
    model = model.cuda()
    print("Model on cuda")
    device = "cuda"

# %%


def run_policy_reset(num_steps, envs, policy, pad=False):
    new_obs, _ = envs.reset()
    new_obs = th.as_tensor(new_obs)
    obs = [new_obs]
    carry = policy.recurrent_initial_state(envs.num_envs)
    all_false = th.zeros(envs.num_envs, dtype=th.bool)
    eps_done = np.zeros(envs.num_envs, dtype=bool)
    eps_solved = np.zeros(envs.num_envs, dtype=bool)
    episode_lengths = np.zeros(envs.num_envs, dtype=np.int32)
    all_log_probs = []
    for _ in range(num_steps - 1):
        dist, carry = policy.get_distribution(new_obs, carry, all_false)
        log_probs = dist.distribution.logits
        all_log_probs.append(log_probs)
        action = log_probs.argmax(-1)
        new_obs, _, term, trunc, _ = envs.step(action.detach().cpu().numpy())
        # assert not (np.any(term) or np.any(trunc))
        new_obs = th.as_tensor(new_obs)
        obs.append(new_obs)
        eps_done |= term | trunc
        episode_lengths[~eps_solved] += 1
        eps_solved |= term
        if np.all(eps_done):
            break
    episode_lengths = th.as_tensor(episode_lengths)
    if pad:
        obs = th.nn.utils.rnn.pad_sequence(obs, batch_first=True)
        all_log_probs = th.nn.utils.rnn.pad_sequence(all_log_probs, batch_first=True)
    else:
        obs = th.stack(obs)[:, episode_lengths == num_steps - 1]
        all_log_probs = th.stack(all_log_probs)[:, episode_lengths == num_steps - 1]
    print("Number of envs:", obs.shape[1])
    return obs, th.as_tensor(episode_lengths), th.as_tensor(eps_solved), all_log_probs


envs = boxo_cfg.make()


def metrics(clean_probs, patched_probs, clean_length, print_metrics=True):
    # take until clean_length
    unpadded_clean_probs = [clean_probs[:l, i] for i, l in enumerate(clean_length)]
    unpadded_patched_probs = [patched_probs[:l, i] for i, l in enumerate(clean_length)]

    correct, total = 0, 0
    eps_wise_acc = []
    mse_sum, norm_sum = 0.0, 0.0

    # Calculate all metrics in a single loop through episodes
    for i, (clean, patched) in enumerate(zip(unpadded_clean_probs, unpadded_patched_probs)):
        # Overall accuracy calculation
        ep_correct = (clean.argmax(-1) == patched.argmax(-1)).sum().item()
        correct += ep_correct
        total += len(clean)

        # Episode-wise accuracy
        ep_acc = ep_correct / len(clean) if len(clean) > 0 else 0.0
        eps_wise_acc.append(th.tensor(ep_acc))

        # MSE components
        ep_mean = clean.mean(0, keepdim=True)  # Mean per action probability
        ep_mse = ((clean - patched) ** 2).sum(-1).mean().item()
        ep_norm = ((clean - ep_mean) ** 2).sum(-1).mean().item()

        mse_sum += ep_mse * len(clean)
        norm_sum += ep_norm * len(clean)

    acc = correct / total
    eps_wise_acc = th.stack(eps_wise_acc)
    argsort_acc = th.argsort(eps_wise_acc)
    norm_mse = mse_sum / norm_sum if norm_sum > 0 else 0.0

    if print_metrics:
        print("Total steps:", total)

    # Calculate 95% confidence interval using bootstrap
    def calculate_mean(data):
        return data.mean().item()

    bootstrap_result = bootstrap(
        (eps_wise_acc.numpy(),),
        statistic=calculate_mean,
        confidence_level=0.95,
        random_state=args.seed,
        n_resamples=1000,
        method="percentile",
    )

    ci_low, ci_high = bootstrap_result.confidence_interval
    delta = max(ci_high - acc, acc - ci_low)
    if print_metrics:
        print(f"Accuracy: {acc * 100:.2f}% ± {delta * 100:.2f}%")
        print(f"Bottom 10 acc: {eps_wise_acc[argsort_acc[:10]]}")
        print(f"Bottom 10 envs: {argsort_acc[:10]}")
        print(f"Normalized MSE: {norm_mse}")

    return acc, delta, norm_mse


# %% zero-out state at every 0th tick or use local (1-step) cache for non-plan channels

mean_cache = None  # mean cache computed below


def zero_ablation_hook(inputs: th.Tensor, hook: HookPoint, channels):
    inputs[:, channels] = 0
    return inputs


def mean_ablation_hook(inputs: th.Tensor, hook: HookPoint, channels):
    if mean_cache is not None:
        hook_name, pos, _ = hook.name.rsplit(".", 2)
        inputs[:, channels] = mean_cache[hook.name][channels]
    else:
        inputs[:, channels] = inputs[:, channels].mean(dim=0, keepdim=True)  # mean over batch
    return inputs


def ablation_with_cache(inputs: th.Tensor, hook: HookPoint, channels):
    hook_name, pos, _ = hook.name.rsplit(".", 2)  # type: ignore
    pos = int(pos) - 1
    # print(hook_name, pos)
    if pos < 0:
        return inputs  # nothing to ablate at the first step
    hook_name = hook_name.replace("input_", "")
    inputs[:, channels] = th.tensor(reset_cache[hook_name][pos * 3 + 2][:, channels], device=inputs.device)
    return inputs


layer_wise_ablation_channels = []
if ablation_channels == "all":
    layer_wise_ablation_channels = [list(range(32)) for _ in range(3)]
    # layer_wise_ablation_channels = [list(range(32)), list(range(32)), []]
elif ablation_channels in ["plan", "non-plan", "box", "agent", "box_agent", "no-label", "T", "nfa_mpa", "nfa", "mpa"]:
    plan_grp = get_group_channels(group_names=ablation_channels, exclude_nfa_mpa=True)
    layer_wise_ablation_channels = split_by_layer(plan_grp)
elif ablation_channels == "random":
    num_channels = sum(len(channels) for channels in get_group_channels(group_names="non-plan"))
    random_channels = rng.choice(np.arange(32 * 3), num_channels, replace=False)
    random_channels = [[(c // 32, c % 32) for c in random_channels]]
    layer_wise_ablation_channels = split_by_layer(random_channels)
elif ablation_channels.endswith("-random"):
    non_plan_grp = get_group_channels(group_names="non-plan")
    num_channels = sum(len(channels) for channels in non_plan_grp)
    plan_grp = get_group_channels(group_names=ablation_channels.split("-")[0])
    plan_grp = [c for g in plan_grp for c in g]
    num_channels = min(num_channels, len(plan_grp))
    random_channels = rng.choice(np.arange(len(plan_grp)), num_channels, replace=False)
    random_channels = [[plan_grp[c_idx] for c_idx in random_channels]]
    layer_wise_ablation_channels = split_by_layer(random_channels)
elif ablation_channels == "none":
    layer_wise_ablation_channels = [[] for _ in range(3)]
else:
    raise ValueError(f"Unknown ablation_channels value: {ablation_channels}")

if exclude_channels:
    try:
        exclude_channel_list = exclude_channels.strip().split("-")
        exclude_channel_list = [list(map(int, channels.strip().split(","))) for channels in exclude_channel_list]
        layer_wise_ablation_channels = [
            [c for c in channels if c not in exclude_channel_list[layer]]
            for layer, channels in enumerate(layer_wise_ablation_channels)
        ]
    except ValueError:
        raise ValueError("Invalid format for exclude_channels. Should be 'ch1,ch2,...-ch1,ch2,...-ch1,ch2,...'")


print("Number of", ablation_channels, "channels ablated:", sum(len(channels) for channels in layer_wise_ablation_channels))

ablation_fn = None
if ablation_method == "single-step":
    ablation_fn = ablation_with_cache
elif ablation_method == "zero":
    ablation_fn = zero_ablation_hook
elif ablation_method == "mean":
    ablation_fn = mean_ablation_hook
else:
    raise ValueError(f"Unknown ablation method: {ablation_method}")


if ablation_gate:
    # assert ablation_gate in ["i", "j", "f", "o"]
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.hook_{ablation_gate}.{pos}.{tick}",
            partial(zero_ablation_hook, channels=channels),
        )
        for pos in range(seq_len)
        for tick in ablation_ticks
        for layer, channels in enumerate(layer_wise_ablation_channels)
    ]
else:
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{tick}",
            partial(ablation_fn, channels=channels),
        )
        for pos in range(1, seq_len)
        for tick in [0]
        for layer, channels in enumerate(layer_wise_ablation_channels)
        for hook_type in ["hook_input_h", "hook_input_c"]
    ]
    fwd_hooks += [
        (
            f"features_extractor.cell_list.0.hook_prev_layer_hidden.{pos}.0",
            partial(
                ablation_fn, channels=layer_wise_ablation_channels[-1]
            ),  # last layer's channels are fed in the first layer
        )
        for pos in range(1, seq_len)
    ]

if not play:
    clean_obs, clean_length, clean_solved, clean_probs = run_policy_reset(seq_len, envs, model, pad=True)
    clean_obs = clean_obs[:, clean_solved]
    clean_length = clean_length[clean_solved]
    clean_probs = clean_probs[:, clean_solved]
    clean_solved = clean_solved[clean_solved]
    assert clean_solved.all(), "All envs should be solved"
    print("Solved envs:", len(clean_solved))
    num_eps = clean_obs.shape[1]
    zero_carry = model.recurrent_initial_state(num_eps)
    eps_start = th.zeros((seq_len, num_eps), dtype=th.bool)
    eps_start[0, :] = True

    if ablation_method == "single-step":
        reset_cache, reset_probs = get_cache_and_probs(clean_obs, model, reset_state_at_every_step=True)
        print("Got 1-step cache")
        acc, delta, norm_mse = metrics(clean_probs, reset_probs, clean_length, print_metrics=False)
        print("Acc, delta, mse with 1-step cache:", acc, delta, norm_mse)

    patched_cache, patched_probs = get_cache_and_probs(clean_obs, model, fwd_hooks)

    metrics(clean_probs, patched_probs, clean_length)

# %%


def state_patch(state, one_step_state):
    for d in range(len(state)):
        state[d][0][:, :, layer_wise_ablation_channels[d]] = one_step_state[d][0][:, :, layer_wise_ablation_channels[d]]
        state[d][1][:, :, layer_wise_ablation_channels[d]] = one_step_state[d][1][:, :, layer_wise_ablation_channels[d]]
    return state


def play_level(
    env,
    policy_th,
    reset_opts={},
    probes=[],
    probe_train_ons=[],
    probe_logits=False,
    sae=None,
    thinking_steps=0,
    max_steps=120,
    internal_steps=False,
    fwd_hooks=None,
    hook_steps: list[int] | int = -1,
    names_filter=None,
    obs_reference=None,  # updates current observation to this variable. Used to get base features for interpretable_forward
    use_interpretable_forward: bool = False,
    re_hook_filter: str = "",  # empty string means no filter
    return_cache=False,
    clean: bool = True,
    single_step_cache_ablation: bool = False,
) -> PlayLevelOutput:
    """Execute the policy on the environment and the probe on the policy's activations.

    Args:
        env (gymnasium.Env): Environment to play the level in.
        policy_th (torch.nn.Module): Policy to play the level with.
        reset_opts (dict): Options to reset the environment with. Useful for custom-built levels
            or providing the `level_file_idx` and `level_idx` of a level in Boxoban.
        probes (list[sklearn.linear_model.LogisticRegression]): Probes to run on the activations of the policy.
        probe_train_ons (list[ProbeTrainOn]): Correponding configuration of the probe.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Observations, actions, rewards, all probe outputs.
    """
    assert len(probe_train_ons) == len(probes)
    try:
        start_obs, info = env.reset(options=reset_opts)
    except:  # noqa
        start_obs, info = env.reset()
    device = policy_th.device
    start_obs = th.tensor(start_obs, device=device)
    all_obs = [start_obs]
    all_act_dist = []
    all_acts = []
    all_logits = []
    all_rewards = []
    all_cache = []
    all_sae_outs = []

    all_probe_outs = [[] for _ in probes]
    N = start_obs.shape[0]
    eps_done = th.zeros(N, dtype=th.bool)
    eps_solved = th.zeros(N, dtype=th.bool)
    episode_lengths = th.zeros(N, dtype=th.int32)

    state = policy_th.recurrent_initial_state(N, device=device)
    one_step_state = policy_th.recurrent_initial_state(N, device=device)
    obs = start_obs
    r, d, t = [0.0], th.tensor([False] * N, dtype=th.bool, device=device), th.tensor([False] * N, dtype=th.bool, device=device)
    for i in range(max_steps):
        if obs_reference is not None:
            obs_reference[:] = obs.cpu()

        if (not clean) and single_step_cache_ablation:
            state = state_patch(state, one_step_state)
            resets = th.tensor([1.0] * N, dtype=th.bool, device=device)
            _, one_step_state = policy_th.get_distribution(obs, one_step_state, resets)
        (distribution, state), cache = run_fn_with_cache(
            policy_th,
            "get_distribution",
            obs,
            state,
            th.tensor([0.0] * N, dtype=th.bool, device=device),
            fwd_hooks=fwd_hooks if (hook_steps == -1) or (i in hook_steps) else None,
            names_filter=names_filter if return_cache else [],
        )
        best_act = distribution.get_actions(deterministic=True)
        all_act_dist.append(distribution.distribution.probs.detach())
        all_acts.append(best_act)
        all_logits.append(distribution.distribution.logits.detach())
        if return_cache:
            all_cache.append(cache)
        if i >= thinking_steps:
            try:
                obs, r, d, t, _ = env.step(best_act.cpu().numpy())
            except ValueError as e:
                if str(e) == "Output array is the wrong shape":
                    warnings.warn(str(e))
                    episode_lengths[~eps_solved] += 1
                    # if single env, then set the episode as done as this error
                    # occurs on a new level with a different shape
                    eps_done |= th.ones(N, dtype=th.bool) if N == 1 else th.zeros(N, dtype=th.bool)
                    break
                else:
                    raise e
            d, t = th.tensor(d), th.tensor(t)
            obs = th.tensor(obs, device=device)
            eps_done |= d | t
            episode_lengths[~eps_solved] += 1
            eps_solved |= d

            all_rewards.append(r)

        if eps_done.all().item() or i == max_steps - 1:
            break
        all_obs.append(obs)
    return PlayLevelOutput(
        obs=th.stack(all_obs).cpu(),
        act_dist=th.stack(all_act_dist),
        acts=th.stack(all_acts),
        logits=th.stack(all_logits),
        rewards=th.tensor(np.array(all_rewards)),
        lengths=episode_lengths,
        solved=eps_solved,
        cache=all_cache if return_cache else None,
        info=info,
        probe_outs=[np.stack(probe_out) for probe_out in all_probe_outs],
        sae_outs=th.stack(all_sae_outs) if sae else None,
    )


if ablation_gate:
    if ablation_method == "single-step":
        raise ValueError("Single-step cache ablation cannot be done with gate ablation. Use zero or mean ablation.")
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.hook_{ablation_gate}.{pos}.{tick}",
            partial(ablation_fn, channels=channels),
        )
        for pos in range(1)
        for tick in ablation_ticks
        for layer, channels in enumerate(layer_wise_ablation_channels)
    ]
elif ablation_method == "single-step":
    fwd_hooks = None
else:
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{tick}",
            partial(ablation_fn, channels=channels),
        )
        for pos in range(1)
        for tick in [0]
        for layer, channels in enumerate(layer_wise_ablation_channels)
        for hook_type in ["hook_input_h", "hook_input_c"]
    ]
    fwd_hooks += [
        (
            f"features_extractor.cell_list.0.hook_prev_layer_hidden.{pos}.0",
            partial(
                ablation_fn, channels=layer_wise_ablation_channels[-1]
            ),  # last layer's channels are fed in the first layer
        )
        for pos in range(1)
    ]


def mean_with_ci(successes):
    mean = successes.mean()
    ci = bootstrap(
        (successes,),
        statistic=lambda x: x.mean(),
        confidence_level=0.95,
        random_state=42,
        n_resamples=1000,
        method="percentile",
    )
    delta = max(ci.confidence_interval[1] - mean, mean - ci.confidence_interval[0])
    return mean, delta


use_local_cache = ablation_method == "single-step"

if ablation_method == "mean":
    envs = boxo_cfg.make()
    # for batch_idx in tqdm(range(0, num_envs // batch_size)):
    #     envs = boxo_cfg.make()
    #     for _ in range(batch_idx):
    #         envs.reset()
    print("Collecting mean cache")
    clean_play = play_level(envs, model, fwd_hooks=None, clean=True, return_cache=True)
    mean_cache = {k: th.stack([d[k] for d in clean_play.cache]).mean((0, 1)) for k in clean_play.cache[0].keys()}
    del clean_play

if play:
    envs = boxo_cfg.make()

    patched_successes = []
    for batch_idx in tqdm(range(0, num_envs // batch_size)):
        envs = boxo_cfg.make()
        for _ in range(batch_idx):
            envs.reset()
        patched_play = play_level(envs, model, fwd_hooks=fwd_hooks, clean=clean, single_step_cache_ablation=use_local_cache)
        patched_successes.append(patched_play.solved)
    patched_successes = th.cat(patched_successes).numpy()
    # mean with 95% confidence interval
    patched_mean, patched_delta = mean_with_ci(patched_successes)

    prefix = f"{ablation_gate} gate" if ablation_gate else ""
    prefix += f" {ablation_channels} abl" if ablation_channels != "none" else "Original network"
    print(f"{prefix}: {patched_mean * 100:.2f}% ± {patched_delta * 100:.2f}%")

# %%
